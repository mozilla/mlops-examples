import pandas as pd
from metaflow.plugins.cards.card_modules.components import Image, Table
from sklearn import metrics
import matplotlib.pyplot as plt
def get_roc_curve_image(actual, predicted):
    fpr, tpr, _ = metrics.roc_curve(actual, predicted)
    auc = metrics.roc_auc_score(actual, predicted)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    return Image.from_matplotlib(plt)
def get_confusion_matrix_component(confusion_matrix):
    df = pd.DataFrame(data={"1 Predicted": confusion_matrix[:][0], "0 Predicted": confusion_matrix[:][1]})
    df.insert(0, '', ["1 Actual", "0 Actual"])
    return Table.from_dataframe(df)
