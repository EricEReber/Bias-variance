import seaborn as sns

from utils import *

np.random.seed(2022)

# get data
"""
This script plots the bias variance trade-off for OLS providing up to a 9th degree polynomial approximation of the Franke function 
"""
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

# define run
N = 20
bootstraps = 100
sci_OLS = LinearRegression(fit_intercept=False)
centering = False

errors = np.zeros(N + 1)
biases = np.zeros(N + 1)
variances = np.zeros(N + 1)

# for polynomial degree
for n in range(N + 1):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    z_preds = sci_bootstrap(
        X_train[:, :l],
        X_test[:, :l],
        z_train,
        z_test,
        bootstraps,
        sci_OLS,
    )

    # bias-variance trade-off
    error, bias, variance = bias_variance(z_test, z_preds)
    errors[n] = error
    biases[n] = bias
    variances[n] = variance

# plot
sns.set(font_scale=3)
plt.title("Bias variance trade-off OLS")
plt.plot(biases, label="bias", linewidth=5)
plt.plot(errors, "r--", label="MSE test", linewidth=5)
plt.plot(variances, label="variance", linewidth=5)
plt.xlabel("Polynomial degree (N)", size=22)
plt.legend()
plt.show()
