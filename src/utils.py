import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils import resample
from typing import Tuple, Callable
import sys
import argparse


def sci_bootstrap(X_train, X_test, z_train, z_test, bootstraps, scikit_model):

    z_preds_test = np.empty((z_test.shape[0], bootstraps))

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        scikit_model.fit(X_, z_)
        z_pred_test = scikit_model.predict(X_test)
        z_preds_test[:, i] = z_pred_test

    return z_preds_test


def bias_variance(z_test: np.ndarray, z_preds_test: np.ndarray):
    MSEs, _ = scores(z_test, z_preds_test)
    error = np.mean(MSEs)
    bias = np.mean(
        (z_test - np.mean(z_preds_test, axis=1, keepdims=True).flatten()) ** 2
    )
    variance = np.mean(np.var(z_preds_test, axis=1, keepdims=True))

    return error, bias, variance


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# debug function
def SkrankeFunction(x, y):
    return 0 + 1 * x + 2 * y + 3 * x**2 + 4 * x * y + 5 * y**2


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)

    return X


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n


def preprocess(x: np.ndarray, y: np.ndarray, z: np.ndarray, N, test_size):
    X = create_X(x, y, N)

    zflat = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, zflat, test_size=test_size)

    return X, X_train, X_test, z_train, z_test


def minmax_dataset(X, X_train, X_test, z, z_train, z_test):
    x_scaler = MinMaxScaler()
    z_scaler = MinMaxScaler()

    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    X = x_scaler.transform(X)

    z_shape = z.shape

    # make all zeds into 1 dimensional arrays for standardscaler
    z_train = z_train.reshape((z_train.shape[0], 1))
    z_test = z_test.reshape((z_test.shape[0], 1))
    z = z.ravel().reshape((z.ravel().shape[0], 1))

    z_scaler.fit(z_train)
    z_train = np.ravel(z_scaler.transform(z_train))
    z_test = np.ravel(z_scaler.transform(z_test))
    z = np.ravel(z_scaler.transform(z))
    z = z.reshape(z_shape)

    return X, X_train, X_test, z, z_train, z_test


def scores(z, z_preds):
    N = z_preds.shape[1]
    MSEs = np.zeros((N))
    R2s = np.zeros((N))

    for n in range(N):
        MSEs[n] = MSE(z, z_preds[:, n])
        R2s[n] = R2(z, z_preds[:, n])

    return MSEs, R2s


def read_from_cmdline():
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Read in arguments for tasks")

    group = parser.add_mutually_exclusive_group()

    # with debug or file, we cannot have noise. We cannot have debug and file
    # either
    group.add_argument("-f", "--file", help="Terrain data file name")
    group.add_argument(
        "-d",
        "--debug",
        help="Use debug function for testing. Default false",
        action="store_true",
    )
    group.add_argument(
        "-no",
        "--noise",
        help="Amount of noise to have. Recommended range [0-0.1]. Default 0.05",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-st",
        "--step",
        help="Step size for linspace function. Range [0.01-0.4]. Default 0.05",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-b", "--betas", help="Betas to plot, when applicable. Default 10", type=int
    )
    parser.add_argument("-n", help="Polynomial degree. Default 9", type=int, default=9)
    parser.add_argument(
        "-nsc",
        "--noscale",
        help="Do not use scaling (centering for synthetic case or MinMaxScaling for organic case)",
        action="store_true",
    )

    # parse arguments and call run_filter
    args = parser.parse_args()

    # error checking
    if args.noise < 0 or args.noise > 1:
        raise ValueError(f"Noise value out of range [0,1]: {args.noise}")

    if args.step < 0.01 or args.step > 0.4:
        raise ValueError(f"Step value out of range [0,1]: {args.noise}")

    if args.n <= 0:
        raise ValueError(f"Polynomial degree must be positive: {args.N}")

    num_betas = int((args.n + 1) * (args.n + 2) / 2)  # Number of elements in beta
    if args.betas:
        if args.betas > num_betas:
            raise ValueError(
                f"More betas than exist in the design matrix: {args.betas}"
            )
        betas_to_plot = args.betas
    else:
        betas_to_plot = min(10, num_betas)

    if args.file:
        # Load the terrain
        z = np.asarray(imread(args.file), dtype="float64")
        x = np.arange(z.shape[0])
        y = np.arange(z.shape[1])
        x, y = np.meshgrid(x, y, indexing="ij")

        # split data into test and train
        X, X_train, X_test, z_train, z_test = preprocess(x, y, z, args.n, 0.2)

        # normalize data
        centering = False
        if not args.noscale:
            X, X_train, X_test, z, z_train, z_test = minmax_dataset(
                X, X_train, X_test, z, z_train, z_test
            )
    else:
        # create synthetic data
        x = np.arange(0, 1, args.step)
        y = np.arange(0, 1, args.step)
        x, y = np.meshgrid(x, y)
        if args.debug:
            z = SkrankeFunction(x, y)
        else:
            z = FrankeFunction(x, y)
            # add noise
            z += args.noise * np.random.standard_normal(z.shape)
        centering = not args.noscale

        X, X_train, X_test, z_train, z_test = preprocess(x, y, z, args.n, 0.2)

    return (
        betas_to_plot,
        args.n,
        X,
        X_train,
        X_test,
        z,
        z_train,
        z_test,
        centering,
        x,
        y,
        z,
    )
