import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

def k_fold_cross_validation(X, y, k, random_seed=0):
    """
    Implementation of K-Fold Cross Validation

    Args:
        X: Features
        y: Targets
        k: Number of folds
        random_seed: A seed to make the randomization deterministic and reproducible

    Returns:
        A list containing k tuples of the form (X_train, y_train, X_val, y_val) corresponding to the k folds
    """
    # np.random.seed(random_seed)
    # np.random.shuffle(X)
    # np.random.shuffle(y)
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    folds = []
    for i in range(k):
        x_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
        x_val = X_folds[i]
        y_val = y_folds[i]
        folds.append((x_train, y_train, x_val, y_val))
    return folds

def ComputeLoss(y_target, y_pred, params, bias, loss_fn, regularization, regularization_scaling_factor, average=True):
    """
    Wrapper function which wraps different loss functions (Only L2 Loss Implemented Here as Linear Regression used Least Squares Optimization)

    Args:
        y_target: Target Values
        y_pred: Predicted Values
        params: Parameters of the Model (Needed for Regularization)
        bias: Bias of the Model (Needed for Regularization)
        loss_fn: Loss Function to use (L2)
        regularization: Regularization to use (No, Lasso, Ridge)
        regularization_scaling_factor: Scaling Factor for Regularization
        average: A bool value indicating whether to average the loss or not

    Returns:
        A float value containing the loss
    """
    if loss_fn == "L2":
        return L2_loss(y_target, y_pred, regularization, params, bias, regularization_scaling_factor, average)

def L2_loss(y_target, y_pred, regularization, params, bias, regularization_scaling_factor, average=True):
    """
    Implementation of L2 Loss

    Args:
        y_target: Target Values
        y_pred: Predicted Values
        regularization: Regularization to use ('No', Lasso, Ridge)
        params: Parameters of the Model (Needed for Regularization)
        bias: Bias of the Model (Needed for Regularization)
        regularization_scaling_factor: Scaling Factor for Regularization
        average: A bool value indicating whether to average the loss or not

    Returns:
        A float value containing the total loss (Not Mean Loss)
    """
    if regularization == 'Lasso':
        Lasso_loss = np.sum(np.square(y_target - y_pred)) + regularization_scaling_factor*(np.sum(np.abs(params)) + np.sum(np.abs(bias)))
        if average:
            return Lasso_loss / len(y_target)
        return Lasso_loss
    elif regularization == 'Ridge':
        Ridge_loss = np.sum(np.square(y_target - y_pred)) + regularization_scaling_factor*(np.sum(np.square(params)) + np.sum(np.square(bias)))
        if average:
            return Ridge_loss / len(y_target)
        return Ridge_loss
    else:
        loss = np.sum(np.square(y_target - y_pred))
        if average:
            return loss / len(y_target)
        return loss

def RMSE(y_target, y_pred):
    """
    Implementation of Root Mean Squared Error

    Args: 
        y_target: Target Values
        y_pred: Predicted Values
    
    Returns:
        A float value containing the RMSE
    """
    return np.sqrt(np.mean(np.square(y_target - y_pred)))

def plot_RMSE(ls_RMSE, split, fold, show, total_folds, learning_rate, regularization, regularization_scaling_factor):
    """
    Implementation of a function to plot and save the RMSE trends over the epochs and optionally display it

    Args:
        ls_RMSE: A list containing the RMSE values over the epochs
        split: The Split Name
        fold: The fold number
        show: A bool value indicating whether to show the plot or not
        total_folds: Total number of folds
        learning_rate: Learning Rate used for training

    Returns:
        None
    """
    plt.clf()
    plt.plot(ls_RMSE)
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Epochs for {split} Fold {fold} of Total Folds {total_folds} (Learning Rate = {learning_rate}, Regularization = {regularization}, Regularization Scaling Factor = {'NA' if regularization == 'No' else regularization_scaling_factor})", loc='center', wrap=True)
    if show:
        plt.show()
    plt.savefig(f"plots/split_{split}_RMSE_lr={lr}_totalFolds_{total_folds}_fold_{fold}_reg={regularization}_regScalingFactor={regularization_scaling_factor}.png")

def linear_regression_solve_normal_form(x_train, 
                                        y_train, 
                                        x_val, 
                                        y_val):
    """
    Implementation of Linear Regression using Normal Form

    Args:
        x_train: Training Data
        y_train: Training Labels
        x_val: Validation Data
        y_val: Validation Labels

    Returns:
        A tuple containing the parameters, bias, training RMSE, validation RMSE
    """
    params = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    bias = np.mean(y_train - x_train.dot(params))
    y_train_pred = x_train.dot(params) + bias
    y_val_pred = x_val.dot(params) + bias
    train_RMSE = RMSE(y_train, y_train_pred)
    val_RMSE = RMSE(y_val, y_val_pred)
    return params, bias, train_RMSE, val_RMSE

def linear_regression_with_gradient_descent(x_train, 
                                            y_train, 
                                            x_val, 
                                            y_val, 
                                            epochs, 
                                            lr, 
                                            regularization='No', 
                                            regularization_scaling_factor=1.0,
                                            useEarlyStopping=False, 
                                            EarlyStopping_patience=10,
                                            EarlyStopping_min_delta=0.001,
                                            random_seed=0, 
                                            verbose=1):
    """
    Implementation of Linear Regression with Gradient Descent

    Args:
        x_train: Training Features
        y_train: Training Targets
        x_val: Validation Features
        y_val: Validation Targets
        epochs: Number of epochs to train the model
        lr: Learning Rate
        regularization: Regularization to use (No, Lasso, Ridge)
        regularization_scaling_factor: Scaling Factor for Regularization
        useEarlyStopping: A bool value indicating whether to use Early Stopping or not
        EarlyStopping_patience: Number of epochs to wait before stopping if the validation loss does not decrease
        EarlyStopping_min_delta: Minimum change in validation loss to qualify as an improvement
        random_seed: A seed to make the randomization deterministic and reproducible
        verbose: A flag to control the verbosity of the training process
    
    Returns:
        A tuple containing the parameters and bias of the final model
    """
    num_params = len(x_train[0])
    params = np.random.RandomState(random_seed).normal(size=num_params)
    bias = np.random.RandomState(random_seed).normal(size=1)

    # Derivatives of y with respect to params and bias
    # y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_n * x_n
    # y = bias + params[0] * x_1 + params[1] * x_2 + ... + params[n] * x_n
    # y = bias + np.dot(params, x)
    # dy/dbias = 1
    # dy/dparams[0] = x_1
    # dy/dparams[1] = x_2
    # ...
    # dy/dparams[n] = x_n
    #0.0003122935124619811
    ls_loss = []
    ls_RMSE_train = []
    ls_RMSE_val = [] 
    earlyStopping = False
    epoch = 0
    print(regularization)
    if verbose == 2 or verbose == 3:
        open('log.txt', 'w').close()
    while epoch < epochs and not earlyStopping:
        y_pred = np.dot(x_train, params) + bias
        loss = ComputeLoss(y_train, y_pred, params, bias, "L2", regularization, regularization_scaling_factor)
        for i in range(num_params):
            if regularization == 'No':
                params[i] -= lr * np.sum(x_train[:, i] * (y_pred - y_train))/len(x_train)
            elif regularization == 'Lasso':
                # print(lr * (np.sum(x_train[:, i] * (y_pred - y_train)) + regularization_scaling_factor*np.sign(params[i]))/len(x_train))
                params[i] -= lr * (np.sum(x_train[:, i] * (y_pred - y_train)) + regularization_scaling_factor*np.sign(params[i]))/len(x_train)
            elif regularization == 'Ridge':
                params[i] -= lr * (np.sum(x_train[:, i] * (y_pred - y_train))+ regularization_scaling_factor*2*params[i])/len(x_train) 
        if regularization == 'No':
            bias -= lr * np.sum((y_pred - y_train))/len(x_train)
        elif regularization == 'Lasso':
            bias -= lr * (np.sum((y_pred - y_train)) + regularization_scaling_factor*np.sign(bias))/len(x_train)
        elif regularization == 'Ridge':
            bias -= lr * (np.sum((y_pred - y_train)) + regularization_scaling_factor*2*bias)/len(x_train)
        y_pred_val = np.dot(x_val, params) + bias
        loss_val = ComputeLoss(y_val, y_pred_val, params, bias, "L2", regularization, regularization_scaling_factor)
        ls_loss.append(loss_val)
        if useEarlyStopping:
            earlyStopping = EarlyStopping(ls_loss, patience=EarlyStopping_patience, min_delta=EarlyStopping_min_delta)
        RMSE_train = RMSE(y_train, y_pred)
        RMSE_val = RMSE(y_val, y_pred_val)
        ls_RMSE_train.append(RMSE_train)
        ls_RMSE_val.append(RMSE_val)
        message = f"Epoch {epoch}: loss_train={loss:.3f}, loss_val={loss_val:.3f}, RMSE_train={RMSE_train:.3f}, RMSE_val={RMSE_val:.3f}"
        if verbose == 1:
            print(message)
        elif verbose == 2:
            log_message(message, "log.txt")
        elif verbose == 3:
            print(message)
            log_message(message, "log.txt")
        else:
            pass
        epoch += 1

    return params, bias, ls_RMSE_train, ls_RMSE_val

df = pd.read_csv('Real estate.csv')
# Since the dataset was released 4 years ago I assume the dates are relative to that and hence break the transaction date into years since last sold
df['Year last sold'] = df['X1 transaction date'].apply(lambda x: int(str(x).split('.')[0]))
df['Years since last sold'] = 2018 - df['Year last sold']
X = df.drop(columns=['Y house price of unit area', 'X1 transaction date', 'No', 'Year last sold']).to_numpy()
y = df['Y house price of unit area'].to_numpy()
X = min_max_normalization(X)
# X_train = X
# y_train = y
X_train, X_test = make_test_set(X, 0.1)
y_train, y_test = make_test_set(y, 0.1)
# # solve using sklearn
# from sklearn.linear_model import LinearRegression
# reg = LinearRegression().fit(X_train, y_train)
# print(reg.coef_)
# print(reg.intercept_)
# # print RMSE
# y_pred = reg.predict(X_test)
# print(RMSE(y_test, y_pred))
# # Train RMSE
# y_pred = reg.predict(X_train)
# print(RMSE(y_train, y_pred))
# exit()
all_results_grad_descent = {}
average_results_grad_descent = {}

all_results_normal_equation = {}
average_results_normal_equation = {}

learning_rates = [0.01]
regularization_scaling_factors = [1]
for regularization in ['No']:
    for regularization_scaling_factor in regularization_scaling_factors:
        for lr in learning_rates:
            print(f"Learning Rate: {lr}")
            print()
            for k in range(4,5):
                folds = k_fold_cross_validation(X, y, k, random_seed=0)
                print(f"K-Fold Cross Validation with k={k}")

                train_RMSEs_grad_descent = []
                val_RMSEs_grad_descent = []
                test_RMSEs_grad_descent = []
                train_RMSEs_normal_equation = []
                val_RMSEs_normal_equation = []
                test_RMSEs_normal_equation = []

                for i, (x_train, y_train, x_val, y_val) in enumerate(folds):
                    print(f"Fold {i+1}")
                    print()
                    X_test = x_val
                    y_test = y_val
                    print("Normal Form Solution")
                    params, bias, train_RMSE_for_normal_equation, val_RMSE_for_normal_equation = linear_regression_solve_normal_form(x_train, y_train, x_val, y_val)
                    # print(params, bias)
                    test_RMSE_for_normal_equation = RMSE(y_test, np.dot(X_test, params) + bias)
                    key = f"Fold {i+1} Total Folds {k} - Normal Form Solution"
                    all_results_normal_equation[key] = [train_RMSE_for_normal_equation, val_RMSE_for_normal_equation, test_RMSE_for_normal_equation]
                    train_RMSEs_normal_equation.append(train_RMSE_for_normal_equation)
                    val_RMSEs_normal_equation.append(val_RMSE_for_normal_equation)
                    test_RMSEs_normal_equation.append(test_RMSE_for_normal_equation)
                    print(f'Train RMSE: {train_RMSE_for_normal_equation:.3f}, Validation RMSE: {val_RMSE_for_normal_equation:.3f}, Test RMSE: {test_RMSE_for_normal_equation:.3f}')

                    print("Gradient Descent Solution")
                    print("Regularization:", regularization)
                    print("Regularization Scaling Factor:", regularization_scaling_factor)
                    params, bias, ls_RMSE_train, ls_RMSE_val = linear_regression_with_gradient_descent(x_train, y_train, x_val, y_val, epochs=100000, lr=lr, 
                                                                            regularization=regularization, regularization_scaling_factor = regularization_scaling_factor, useEarlyStopping=True,
                                                                            EarlyStopping_patience=5, EarlyStopping_min_delta=1e-1,
                                                                            random_seed=42, verbose=1)
                    print(f'Params: {params}, Bias: {bias}')

                    plot_RMSE(ls_RMSE_train, "train", i, False, k, lr, regularization, regularization_scaling_factor)
                    plot_RMSE(ls_RMSE_val, "val", i, False, k, lr, regularization, regularization_scaling_factor)

                    y_pred = np.dot(X_test, params) + bias
                    # print(ls_RMSE_train)
                    # print(ls_RMSE_val)
                    train_RMSE_for_grad_descent = ls_RMSE_train[-1]
                    val_RMSE_for_grad_descent  = ls_RMSE_val[-1]
                    test_RMSE_for_grad_descent  = RMSE(y_test, y_pred)

                    print(f'RMSE_train: {train_RMSE_for_grad_descent:.3f}, RMSE_val: {val_RMSE_for_grad_descent:.3f}, RMSE_test: {test_RMSE_for_grad_descent:.3f}')
                    print()

                    key = f"lr={lr}, k={k}, fold={i+1}, regularization={regularization}, regularization_scaling_factor={regularization_scaling_factor}"
                    all_results_grad_descent[key] = [train_RMSE_for_grad_descent, val_RMSE_for_grad_descent, test_RMSE_for_grad_descent]

                    train_RMSEs_grad_descent.append(train_RMSE_for_grad_descent)
                    val_RMSEs_grad_descent.append(val_RMSE_for_grad_descent)
                    test_RMSEs_grad_descent.append(test_RMSE_for_grad_descent)

                print(f"Average RMSE_train: {np.mean(train_RMSEs_grad_descent):.3f}, Average RMSE_val: {np.mean(val_RMSEs_grad_descent):.3f}, Average RMSE_test: {np.mean(test_RMSEs_grad_descent):.3f}")
                key = f"lr={lr}, k={k}, regularization={regularization}, regularization_scaling_factor={regularization_scaling_factor}"
                average_results_grad_descent[key] = [np.mean(train_RMSEs_grad_descent), np.mean(val_RMSEs_grad_descent), np.mean(test_RMSEs_grad_descent)]
                key = f"k={k} - Normal Form Solution"
                average_results_normal_equation[key] = [np.mean(train_RMSEs_normal_equation), np.mean(val_RMSEs_normal_equation), np.mean(test_RMSEs_normal_equation)]
        if regularization == 'No':
            break

save_to_json(average_results_grad_descent, "average_results_grad_descent.json")
save_to_json(average_results_normal_equation, "average_results_normal_equation.json")
save_to_json(all_results_grad_descent, "all_results_grad_descent.json")
save_to_json(all_results_normal_equation, "all_results_normal_equation.json")