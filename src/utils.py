import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_correlation_heatmap(df):
    """Plots and displays a correlation heatmap of the dataframe."""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt=".2f")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plots and displays a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def print_model_scores(results_df):
    """Prints the model tuning results."""
    print("\nModel Tuning Results:")
    print(results_df)
