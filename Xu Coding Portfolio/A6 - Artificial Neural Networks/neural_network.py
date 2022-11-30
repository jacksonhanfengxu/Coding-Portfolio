# Author: Hanfeng Xu
# Date: Nov 29, 2021
# Introduction: classify all the Iris in the Fisher’s Iris database with a multi-layer neural network.
#               forward propagation and backward propagation to train the model and perform predictions.
#               To build a neural network classifier that can adapt to large data sets, hyperparameter
#               tuning by cross-validation are used to avoid underfitting or overfitting. Also, by using
#               the Sigmoid function as the activation function, the calculation of the associated gradient
#               was simplified.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LAMBDA = 0.1
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_in_width']
classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']


def sigmoid(v):
    """
    Sigmoid activation function

    :param v: the input value v
    :return: sigmoid(v)
    """
    return 1 / (1 + np.exp(-v))


def forward_propagation(input_x, w, n_layers):
    """
    Forward propagation routine in neural network

    :param input_x: the initial values for each neuron
    :param w: weights for edges
    :param n_layers: number of layers
    :return: the output of the terminal neurons
    """
    outputs = [input_x]
    for l in range(n_layers):
        outputs.append(sigmoid(np.dot(input_x, w[l].T)))
        input_x = np.append(1, outputs[-1])  # add bias
    return outputs


def backward_propagation(y_expect, y_actual, w, n_layers):
    """
    Backward propagation to train the model

    :param y_expect: the expected y values
    :param y_actual: the actual y values
    :param w: weights for edges
    :param n_layers: number of layers
    :return: weights for edges after training
    """
    err = np.array(y_expect - y_actual[-1]).reshape(1, -1)
    for i in range(n_layers, 0, -1):
        yi = y_actual[i]
        if i <= 1:
            b = y_actual[0]
        else:
            b = y_actual[i - 1]
            b = np.append([1], b)
        gradient = err * yi * (1 - yi)  # calculate gradient
        delta_w = LAMBDA * gradient.transpose() * b
        w[i - 1] += delta_w
        err = np.dot(gradient, (np.delete(w[i - 1], [0], axis=1)))
    return w


def random_initial_weights(layers):
    """
    Randomly initialize weights for edges. Each edge is assigned with a
    number between -1 and 1.

    :param layers: the number of neurons in each layer
    :return: weights for edges
    """
    w = []
    for i in range(1, len(layers)):
        weights = []
        for j in range(layers[i]):
            weights.append([np.random.uniform(-1, 1) for _ in range(layers[i - 1] + 1)])
        w.append(np.array(weights))
    return w


def train_helper(x_train, y_train, w):
    """
    Helper routine for training the model.
    :param x_train: train data
    :param y_train: train label
    :param w: weights for edges
    :return: weights for edges after training
    """
    n_layers = len(w)
    for i in range(len(x_train)):
        xi = x_train[i]
        yi = y_train[i]
        xi = np.array(np.append(1, xi))
        b = forward_propagation(xi, w, n_layers)
        w = backward_propagation(yi, b, w, n_layers)
    return w


def train(x_train, y_train, x_valid, y_valid, n_iteration, neurons, print_message):
    """
    Train the model on the given data and do cross validation in order to tune the
    model. The print_message parameter is used for the process of tuning the hyperparameters
    such as n_iteration and the number of neurons in each hidden leyer.

    :param x_train: train data
    :param y_train: train label
    :param x_valid: validation data
    :param y_valid: validation label
    :param n_iteration: the number of iteration to train the model
    :param neurons: information for each layer
    :param print_message: whether to print accuracy or not
    :return: weights for edges after training
    """
    w = random_initial_weights(neurons)
    y_actual_tr = []
    y_actual_val = []
    for i in range(n_iteration):
        w = train_helper(x_train, y_train, w)
    for j in range(len(x_valid)):
        y_pred = predict(x_valid[j], w)
        y_actual_val.append(y_pred)
    for j in range(len(x_train)):
        y_pred = predict(x_train[j], w)
        y_actual_tr.append(y_pred)
    if print_message:
        print(f"The accuracy on validation data is: {get_accuracy(y_valid, y_actual_val)}")
        print(f"The accuracy on train data is: {get_accuracy(y_train, y_actual_tr)}")
    return w


def predict(features_in, w):
    """
    Given the features for Iris, predict its class.

    :param features_in: features for Iris
    :param w: weights for edges
    :return: weights for edges after training
    """
    f = np.append(1, features_in)
    n_layers = len(w)
    pred_vals = forward_propagation(f, w, n_layers)[-1]
    pred_class = np.argmax(pred_vals)
    res = [0.0 for _ in range(len(pred_vals))]
    res[pred_class] = 1.0
    return res


def get_accuracy(y_expect, y_actual):
    """
    Get the accuracy of prediction.

    :param y_expect: the expected y values
    :param y_actual: the actual y values
    :return: the accuracy of prediction
    """
    total = len(y_expect)
    comp = [1 for i in range(total) if list(y_expect[i]) == y_actual[i]]
    return sum(comp) / total


def preprocessing():
    """
    Read data and do some pre-processing.

    :return: None
    """
    df_iris = pd.read_csv('ANN - Iris data.txt', sep=",", header=None)
    df_iris.columns = features + ["class"]
    x = df_iris[features]
    y = df_iris['class']
    y_mat = []
    for i in range(len(y)):
        if y.iloc[i] == 'Iris-setosa':
            y_mat.append([1, 0, 0])
        elif y.iloc[i] == 'Iris-versicolor':
            y_mat.append([0, 1, 0])
        else:
            y_mat.append([0, 0, 1])
    return np.array(x), np.array(y_mat)


def run_model(x, y, n_iters, print_message):
    """
    Run the neural network model.

    :param x: features
    :param y: label
    :param n_iters: the number of iteration to train the model
    :param print_message: whether to print accuracy or not
    :return: weights for edges after training
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.2)
    neurons = [4, 10, 20, 40, 3]
    w = train(x_train, y_train, x_valid, y_valid, n_iters, neurons, print_message)
    return w


if __name__ == '__main__':
    x, y = preprocessing()
    ########### Tuning the model ############
    # for i in range(100, 300, 10):
    #     print(i)
    #     run_model(x, y, i)
    w = run_model(x, y, 200, False)
    print("Model training finished--------------")
    print("Input format: sepal length, sepal width, petal length, petal in width")
    while True:
        q1 = input("Please input the feature of the Iris: (q to quit)  ")
        if q1 == "q":
            break
        else:

            f = q1.split(",")
            f = [float(ele) for ele in f]
            print(f)
            res = predict(f, w)
            for i in range(len(res)):
                if res[i] == 1:
                    print(f"The Iris is {classes[i]}")
                    break
