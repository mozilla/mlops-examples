# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import sys
from logging import INFO, basicConfig, getLogger

from bugbug import db
from bugbug.models import MODELS, get_model_class
from bugbug.utils import CustomJsonEncoder, zstd_compress
from metaflow import FlowSpec, step

MODELS_WITH_TYPE = ("component",)

basicConfig(level=INFO)
logger = getLogger(__name__)


class ExtractedTrainer(FlowSpec):
    """
    A flow to train SpamBug. main() has been updated to create a Trainer() which launches the 
    FlowSpec

    The flow performs the following steps:
    1) Uses Trainer extending FlowSpec
    """
    @step
    def start(self):
        """
        Not doing anything
        """

        self.next(self.download_data)

    @step
    def download_data(self):
        # Download datasets that were built by bugbug_data.
        os.makedirs("data", exist_ok=True)

        args = parse_args(["spambug"])

        model_class = get_model_class(args.model)
        parameter_names = set(inspect.signature(model_class.__init__).parameters)
        parameters = {
            key: value for key, value in vars(args).items() if key in parameter_names
        }
        model_obj = model_class(**parameters)

        for required_db in model_obj.training_dbs:
            assert db.download(required_db)
        else:
            logger.info("Skipping download of the databases")

        self.model_obj = model_obj
        self.args = args
        self.next(self.train)

    @step
    def train(self):
        """
        This step trains the BugBug spambug model
        """
        logger.info("Training *%s* model", self.args.model)

        metrics = self.model_obj.train(limit=self.args.limit)
        self.metrics = metrics

        metric_file_path = "metrics.json"
        with open(metric_file_path, "w") as metric_file:
            json.dump(self.metrics, metric_file, cls=CustomJsonEncoder)

        logger.info("Training done")

        model_file_name = f"{self.model_name}model"
        assert os.path.exists(model_file_name)
        zstd_compress(model_file_name)

        logger.info("Model compressed")

        if self.model_obj.store_dataset:
            assert os.path.exists(f"{model_file_name}_data_X")
            zstd_compress(f"{model_file_name}_data_X")
            assert os.path.exists(f"{model_file_name}_data_y")
            zstd_compress(f"{model_file_name}_data_y")

        self.next(self.end)

    @step
    def end(self):
        """
        Finished training
        """
        print("Training completed for BugBug spambug")


def parse_args(args):
    description = "Train the models"
    main_parser = argparse.ArgumentParser(description=description)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only train on a subset of the data, used mainly for integrations tests",
    )
    parser.add_argument(
        "--no-download",
        action="store_false",
        dest="download_db",
        help="Do not download databases, uses whatever is on disk",
    )
    parser.add_argument(
        "--download-eval",
        action="store_true",
        dest="download_eval",
        help="Download databases and database support files required at runtime "
             "(e.g. if the model performs custom evaluations)",
    )
    parser.add_argument(
        "--lemmatization",
        help="Perform lemmatization (using spaCy)",
        action="store_true",
    )
    parser.add_argument(
         "--classifier",
         help="Type of the classifier. Only used for component classification.",
         choices=["default", "nn"],
         default="default",
    )

    subparsers = main_parser.add_subparsers(title="model", dest="model", required=True)

    for model_name in MODELS:
        subparser = subparsers.add_parser(
            model_name, parents=[parser], help=f"Train {model_name} model"
        )

        try:
            model_class_init = get_model_class(model_name).__init__
        except ImportError:
            continue

    for parameter in inspect.signature(model_class_init).parameters.values():
        if parameter.name == "self":
            continue

        # Skip parameters handled by the base class
        # (TODO: add them to the common argparser and skip them automatically without
        #  hardcoding by inspecting the base class)
        if parameter.name == "lemmatization":
            continue

        parameter_type = parameter.annotation
        if parameter_type == inspect._empty:
            parameter_type = type(parameter.default)
        assert parameter_type is not None

        if parameter_type == bool:
            subparser.add_argument(
                f"--{parameter.name}"
                if parameter.default is False
                else f"--no-{parameter.name}",
                action="store_true"
                if parameter.default is False
                else "store_false",
                dest=parameter.name,
            )
        else:
            subparser.add_argument(
                f"--{parameter.name}",
                default=parameter.default,
                dest=parameter.name,
                type=int,
            )

    return main_parser.parse_args(args)


if __name__ == "__main__":
    ExtractedTrainer()