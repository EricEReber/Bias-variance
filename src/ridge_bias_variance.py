import seaborn as sns

from utils import *

np.random.seed(2022)

"""
This script plots the bias variance trade-off for Ridge providing up to a 9th degree polynomial approximation of the Franke function with different values for the regularization hyperparameter
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

# define run
N = 20
bootstraps = 100
lams = np.logspace(-14, 1, 6)
lams[0] = 0
centering = True

sns.set(font_scale=1.5)

# for lambdas
for i in range(len(lams)):
    sci_ridge = Ridge(alpha=lams[i], fit_intercept=False)
    plt.suptitle("Bias variance trade-off Ridge", size=32)
    plt.subplot(321 + i)
    errors = np.zeros(N + 1)
    biases = np.zeros(N + 1)
    variances = np.zeros(N + 1)

    # for polynomial degree
    for n in range(N + 1):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        z_preds, _ = bootstrap(
            X[:, :l],
            X_train[:, :l],
            X_test[:, :l],
            z_train,
            z_test,
            bootstraps,
            centering=centering,
            model=sci_ridge,
            lam=lams[i],
        )

        # bias-variance trade-off
        error, bias, variance = bias_variance(z_test, z_preds)
        errors[n] = error
        biases[n] = bias
        variances[n] = variance

    # plot
    plt.ylim(0, 0.03)
    plt.title(f"For lambda = {lams[i]}", size=20)
    plt.plot(biases, label="bias", linewidth=3)
    plt.plot(errors, "r--", label="MSE test", linewidth=3)
    plt.plot(variances, label="variance", linewidth=3)
    plt.legend(loc="upper right")
    plt.xlabel("Polynomial degree (N)", size=15)
plt.show()
