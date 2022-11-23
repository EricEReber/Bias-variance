import seaborn as sns
from sklearn.neural_network import MLPRegressor

from utils import *

np.random.seed(2022)

"""
This script plots the bias variance trade-off for a feed forward neural network with different amounts of hidden layers for the Franke function
"""

# get data
(
    betas_to_plot,
    N,
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
) = read_from_cmdline()
z_test = z_test.ravel()
z_train = z_train.ravel()

# define run
stop = 500
step = 100
bootstraps = 100

"""
# results
errors = np.zeros(stop // step)
biases = np.zeros(stop // step)
variances = np.zeros(stop // step)

# for one hidden layers of different node sizes
for n in range(1, stop, step):
    sci_MLP = MLPRegressor(
        hidden_layer_sizes=(n),
        max_iter=2000,
        solver="adam",
    )
    z_preds = sci_bootstrap(
        X_train[:, 1:3],
        X_test[:, 1:3],
        z_train,
        z_test,
        bootstraps,
        sci_MLP,
    )

    # bias-variance trade-off
    error, bias, variance = bias_variance(z_test, z_preds)
    errors[n // step - 1] = error
    biases[n // step - 1] = bias
    variances[n // step - 1] = variance

# plot
sns.set(font_scale=3)
plt.title("Bias variance trade-off 1 hidden layers MLP", size=32)
plt.plot(range(1, stop, step), biases, label="bias", linewidth=5)
plt.plot(range(1, stop, step), errors, "r--", label="MSE test", linewidth=5)
plt.plot(range(1, stop, step), variances, label="variance", linewidth=5)
plt.xlabel("Total hidden nodes in hidden layes", size=22)
plt.legend()
plt.show()
"""

errors = np.zeros(stop // step)
biases = np.zeros(stop // step)
variances = np.zeros(stop // step)
# for two hidden layers of different node sizes
for n in range(1, stop, step):
    print(n)
    hidden_layer_size = n // 7 or 1
    sci_MLP = MLPRegressor(
        hidden_layer_sizes=(
            hidden_layer_size,
            hidden_layer_size,
            hidden_layer_size,
            hidden_layer_size,
            hidden_layer_size,
            hidden_layer_size,
            hidden_layer_size,
        ),
        max_iter=2000,
        solver="adam",
    )
    z_preds = sci_bootstrap(
        X_train[:, 1:3],
        X_test[:, 1:3],
        z_train,
        z_test,
        bootstraps,
        sci_MLP,
    )

    # bias-variance trade-off
    error, bias, variance = bias_variance(z_test, z_preds)
    errors[n // step - 1] = error
    biases[n // step - 1] = bias
    variances[n // step - 1] = variance

# plot
sns.set(font_scale=3)
plt.title("Bias variance trade-off 7 hidden layers MLP", size=32)
plt.plot(range(1, stop, step), biases, label="bias", linewidth=5)
plt.plot(range(1, stop, step), errors, "r--", label="MSE test", linewidth=5)
plt.plot(range(1, stop, step), variances, label="variance", linewidth=5)
plt.xlabel("Total hidden nodes in hidden layers", size=22)
plt.legend()
plt.show()
