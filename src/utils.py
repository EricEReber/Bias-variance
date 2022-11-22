import numpy as np

def bias_variance(z_test: np.ndarray, z_preds_test: np.ndarray):
    MSEs, _ = scores(z_test, z_preds_test)
    error = np.mean(MSEs)
    bias = np.mean(
        (z_test - np.mean(z_preds_test, axis=1, keepdims=True).flatten()) ** 2
    )
    variance = np.mean(np.var(z_preds_test, axis=1, keepdims=True))

    return error, bias, variance

def bootstrap(
    X: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    bootstraps: int,
    *,
    centering: bool = False,
    model: Callable = OLS,
    lam: float = 0,
):
    z_preds_train = np.empty((z_train.shape[0], bootstraps))
    z_preds_test = np.empty((z_test.shape[0], bootstraps))

    # non resampled train
    _, z_pred_train, _, _ = evaluate_model(
        X, X_train, X_test, z_train, model, lam=lam, centering=centering
    )

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        _, _, z_pred_test, _ = evaluate_model(
            X, X_, X_test, z_, model, lam=lam, centering=centering
        )
        # z_preds_train[:, i] = z_pred_train
        z_preds_test[:, i] = z_pred_test

    return z_preds_test, z_pred_train
