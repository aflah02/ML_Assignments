import matplotlib.pyplot as plt

def plot_RMSEs(RMSEs, title, x_label, y_label):
    """
    Implementation to plot the RMSEs
    Args:
        RMSEs: RMSEs to plot
        title: Title of the plot
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        save_path: Path to save the plot
        fileName: Name of the file to save the plot
    Returns:
        None
    """
    for RMSE in RMSEs:
        plt.plot(RMSE)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend([f"Fold {i}" for i in range(1, len(RMSEs)+1)])
    return plt