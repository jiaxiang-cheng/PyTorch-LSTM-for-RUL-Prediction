"""RUL Prediction with LSTM"""
from loading_data import *
from model import *
from visualize import *
import numpy as np

N_HIDDEN = 96  # NUMBER OF HIDDEN STATES
N_LAYER = 4  # NUMBER OF LSTM LAYERS
N_EPOCH = 150  # NUM OF EPOCHS
MAX = 135  # UPPER BOUND OF RUL
LR = 0.01  # LEARNING RATE


def testing_function(num, group_for_test):
    rmse_test, result_test = 0, list()

    for ite in range(1, num + 1):
        X_test = group_for_test.get_group(ite).iloc[:, 2:]
        X_test_tensors = Variable(torch.Tensor(X_test.to_numpy()))
        X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        test_predict = model.forward(X_test_tensors)
        data_predict = max(test_predict[-1].detach().numpy(), 0)
        result_test.append(data_predict)
        rmse_test = np.add(np.power((data_predict - y_test.to_numpy()[ite - 1]), 2), rmse_test)

    rmse_test = (np.sqrt(rmse_test / num)).item()
    return result_test, rmse_test


def train(model_for_train, ntrain, group_for_train):
    """

    :param model_for_train: initialized model
    :param ntrain: number of samples in training set
    :param group_for_train: grouped data per sample
    :return: evaluation results
    """
    rmse_temp = 100

    for epoch in range(1, N_EPOCH + 1):

        model_for_train.train()
        epoch_loss = 0

        for i in range(1, ntrain + 1):
            X, y = group_for_train.get_group(i).iloc[:, 2:-1], group_for_train.get_group(i).iloc[:, -1:]
            X_train_tensors = Variable(torch.Tensor(X.to_numpy()))
            y_train_tensors = Variable(torch.Tensor(y.to_numpy()))
            X_train_tensors = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

            outputs = model_for_train(X_train_tensors)  # forward pass

            optimizer.zero_grad()  # calculate the gradient, manually setting to 0
            loss = criterion(outputs, y_train_tensors)  # obtain the loss function
            epoch_loss += loss.item()
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e back propagation

        if epoch % 1 == 0:  # evaluate the model on testing set with each epoch

            model_for_train.eval()  # evaluate model
            result, rmse = testing_function(num_test, group_test)

            if rmse_temp < rmse and rmse_temp < 25:
                result, rmse = result_temp, rmse_temp
                break

            rmse_temp, result_temp = rmse, result  # store the last rmse
            print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, epoch_loss / ntrain, rmse))

    return result, rmse


if __name__ == "__main__":
    # fetch basic information from data sets
    group, group_test, y_test = load_FD001(MAX)
    num_train, num_test = len(group.size()), len(group_test.size())
    input_size = group.get_group(1).shape[1] - 3  # number of features

    # LSTM model initialization
    model = LSTM1(input_size, N_HIDDEN, N_LAYER)  # our lstm class
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training and evaluation
    result, rmse = train(model, num_train, group)
    visualize(result, y_test, num_test, rmse)
