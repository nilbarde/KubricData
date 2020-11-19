import requests
import pandas as pd
import scipy
import numpy as np
import sys

import matplotlib.pyplot as plt


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    # response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    data_train_raw = np.genfromtxt("linreg_train.csv", delimiter=",")
    x = np.array(data_train_raw[0, 1:])
    y = np.array(data_train_raw[1, 1:])
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    datasize = np.size(x)
    dev_cross = np.sum(y * x) - datasize * y_mean * x_mean
    dev_xx = np.sum(x * x) - datasize * x_mean * x_mean

    # ax + b = prediction
    a = dev_cross / dev_xx
    b = y_mean - a * x_mean
    print(a, b)
    for i in range(len(x)):
        print(x[i], a * x[i] + b, y[i])

    plt.scatter(x, y, color="m")
    plt.plot(x, a * x + b, color="g")
    plt.show()
    return a * area + b


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = np.array(list(validation_data.keys()))
    prices = np.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    # exit()
    rmse = np.sqrt(np.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
