import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Prints classification report and accuracy
    """

    print(f"\n{model_name} Evaluation")
    print("-" * 40)
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", cmap="Blues"):
    """
    Plots confusion matrix heatmap
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
