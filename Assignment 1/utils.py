import matplotlib.pyplot as plt

def EarlyStopping(ls_loss, patience, min_delta):
    """
    Implementation of Early Stopping to automate manual oversight of the training process by stopping the training process

    Args:
        ls_loss: A list containing the loss values over the epochs
        patience: Number of epochs to check for improvement and stop the training process if no improvement is observed
        min_delta: Change in loss value which is considered as an improvement

    Returns:
        A bool value indicating whether to stop the training process or not
    """
    if len(ls_loss) > patience:
        if ls_loss[len(ls_loss)-1] - ls_loss[len(ls_loss)-patience] > min_delta:
            return True
    return False

def plot_RMSEs(RMSEs, title, x_label, y_label):
    """
    Implementation to plot the RMSEs
    Args:
        RMSEs: RMSEs to plot
        title: Title of the plot
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        split: Split of the data
    Returns:
        plot of the RMSEs
    """
    for RMSE in RMSEs:
        plt.plot(RMSE)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend([f"Fold {i}" for i in range(1, len(RMSEs)+1)])
    return plt