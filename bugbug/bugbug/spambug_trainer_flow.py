import requests
from metaflow import FlowSpec, step, kubernetes, card, current
from metaflow.plugins.cards.card_modules.components import Markdown, Image

from bugbug import db
from bugbug.metaflow_utils import get_confusion_matrix_component
from bugbug.models.spambug import SpamBugModel


class SpamBugTrainerFlow(FlowSpec, SpamBugModel):
    def __init__(self):
        # multiple inheritance  also see  https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        SpamBugModel.__init__(self)  # This must go before the FlowSpec init due to a Metaflow quirk
        FlowSpec.__init__(self)

    @step
    def start(self):
        self.setup()  # Rjr this is needed because there are pickle serialization issues with
        # objects that are set in the init function. May be metaflow related
        self.next(self.flow_collect_training_data)

    @step
    def flow_collect_training_data(self):
        # TODO -- I assume we will be able to query the version ID of the latest bugubug data
        # and store that as an artifact so that training is consistent between steps
        for required_db in self.training_dbs:
            assert db.download(required_db)

        with open('data/bugs.json.version', 'r') as f:
            self.bugzilla_version_id = f.read()

        with open('data/commits.json.version', 'r') as f:
            self.commits_version_id = f.read()
        self.setup_data_and_splits()
        self.next(self.flow_train)

    @step
    def flow_train(self):
        """
        This step trains the BugBug spambug model
        """
        self.train()
        self.next(self.flow_validate_model)

    @card
    @step
    def flow_validate_model(self):
        self.tracking_metrics = self.evaluate_training()
        current.card.append(
            Markdown("# BugBug Train Summary")
        )

        stats = self.tracking_metrics["report"]["average"]
        current.card.append(
            Markdown("## F1 {:.2f}\n"
                     "##Precision {:.2f}\n"
                     "##Recall {:.2f}".format(stats['f1'], stats['precision'], stats['recall']))
        )
        current.card.append(get_confusion_matrix_component(self.tracking_metrics["confusion_matrix"]))

        current.card.append(
            Image(
                requests.get("https://i.makeagif.com/media/7-06-2015/DkY6g2.gif").content,
                "Training Process"
            )
        )

        #        current.card.append(
        #            Image.from_pil_image(PIL.Image.open("feature_importance.png"), "Feature Importance")
        #        )
        #
        current.card.append(
            Markdown(f"** raw stats ** \n *********** \n {self.tracking_metrics}")
        )
        self.next(self.end)

    @step
    def end(self):
        """
        Finished training
        """
        print("Training completed for BugBug spambug")


if __name__ == "__main__":
    SpamBugTrainerFlow()
