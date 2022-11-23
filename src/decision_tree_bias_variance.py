import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

from utils import *

np.random.seed(2022)

"""
This script plots the bias variance trade-off for a decision tree regressor for different depths.
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
stop = 20
step = 1
bootstraps = 100

# results
errors = np.zeros(stop // step - 1)
biases = np.zeros(stop // step - 1)
variances = np.zeros(stop // step - 1)

# for different max depths
for n in range(1, stop, step):
    sci_tree = DecisionTreeRegressor(max_depth=n, min_samples_leaf=1)
    z_preds = sci_bootstrap(
        X_train[:, 1:3],
        X_test[:, 1:3],
        z_train,
        z_test,
        bootstraps,
        sci_tree,
    )

    # bias-variance trade-off
    error, bias, variance = bias_variance(z_test, z_preds)
    errors[n // step - 1] = error
    biases[n // step - 1] = bias
    variances[n // step - 1] = variance

# plot
sns.set(font_scale=3)
plt.title("Bias variance trade-off decision tree")
plt.plot(range(1, stop, step), biases, label="bias", linewidth=5)
plt.plot(range(1, stop, step), errors, "r--", label="MSE test", linewidth=5)
plt.plot(range(1, stop, step), variances, label="variance", linewidth=5)
plt.xlabel("Maximum depth of tree", size=22)
plt.legend()
plt.show()
