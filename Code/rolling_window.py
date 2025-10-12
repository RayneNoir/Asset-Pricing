import numpy as np
import pandas as pd

def rolling_window(assetReturns, gamma=-1, factors=3, riskfreeRate = 0.0025):
    """
    Determines the out-of-sample rolling window.

    :param assetReturns: The return matrix
    :param gamma: Penalty term
    :param factors: The number of factors
    :param riskfreeRate: The risk-free rate
    :return: the loading matrices, the out-of-sample factors, optimal returns
    """
    excessReturns = assetReturns - riskfreeRate
    T = excessReturns.shape[0]
    n = excessReturns.shape[1]
    T_out_of_sample = T - 240

    prediction_error = np.zeros((T_out_of_sample, n))
    tangent_weights = np.zeros((T_out_of_sample, factors))
    optimal_returns = np.zeros((T_out_of_sample, 1))
    loading_factors = []
    out_of_sample_factors = []

    for i in range(240, T):
        return_in_sample = excessReturns.iloc[i - 240:i]
        Lambda, w_tangent, exposure = pca(return_in_sample, gamma=gamma, factors=factors)

        loading_factors.append(Lambda)
        tangent_weights[i - 240, :] = w_tangent.flatten()

        # Out-of-sample return
        out_of_sample_return = excessReturns.iloc[i].values
        factor_new = out_of_sample_return @ Lambda
        out_of_sample_factors.append(factor_new)

        # Predicted returns
        predicted_future = factor_new @ exposure
        prediction_error[i - 240, :] = out_of_sample_return - predicted_future

        # Optimal portfolio return
        optimal_returns[i - 240, 0] = (factor_new @ w_tangent).item()

    # Metrics
    oos_alpha = prediction_error.mean(axis=0)
    rms_alpha = np.sqrt(np.mean(oos_alpha ** 2))
    idiosyncratic_var = ((prediction_error - oos_alpha) ** 2).sum(axis=0) / (T_out_of_sample - 1)
    ariv = np.mean(idiosyncratic_var / excessReturns.var().values)
    sharpe_ratio = np.mean(optimal_returns) / np.sqrt(np.var(optimal_returns))

    return loading_factors, out_of_sample_factors, sharpe_ratio, oos_alpha, rms_alpha, idiosyncratic_var, ariv

def rolling_window_other_models(assetReturns, marketReturns, riskfreeRate = 0.0025):
    if isinstance(marketReturns, pd.Series):
        marketReturns = marketReturns.to_frame(name="market")

    # Excess returns
    excessReturns = assetReturns - riskfreeRate

    T, n = excessReturns.shape
    k = marketReturns.shape[1]
    T_out = T - 240

    prediction_error = np.zeros((T_out, n))
    optimal_returns = np.zeros((T_out, 1))

    for i in range(240, T):
        # In-sample data
        X_in = marketReturns.iloc[i - 240:i].copy()
        X_in.insert(0, "intercept", 1.0)
        Y_in = excessReturns.iloc[i - 240:i]

        # Out-of-sample
        X_out = marketReturns.iloc[i].values.reshape(1, k)
        Y_out = excessReturns.iloc[i].values

        # Beta estimation: OLS
        Q = np.linalg.inv(X_in.T.values @ X_in.values)
        B = Q @ X_in.T.values @ Y_in.values
        beta = B[1:, :]

        # Predicted returns out-of-sample
        if k == 1:
            predicted_future = beta.flatten() * X_out[0, 0]
            optimal_returns[i - 240] = X_out[0, 0]
        else:
            # Multi-factor
            predicted_future = X_out @ beta

            # Tangency portfolio weights
            Sigma_F = marketReturns.iloc[i - 240:i].cov().values
            mu_F = marketReturns.iloc[i - 240:i].mean().values.reshape(-1, 1)
            iota = np.ones((k, 1))
            w_tangent = np.linalg.inv(Sigma_F) @ mu_F / (iota.T @ np.linalg.inv(Sigma_F) @ mu_F)
            optimal_returns[i - 240] = (X_out @ w_tangent).item()

        # Prediction error
        prediction_error[i - 240, :] = Y_out - predicted_future.flatten()

    # Out-of-sample alpha metrics
    oos_alpha = prediction_error.mean(axis=0)
    rms_alpha = np.sqrt(np.mean(oos_alpha ** 2))

    # Idiosyncratic variance
    idiosyncratic_var = ((prediction_error - oos_alpha) ** 2).sum(axis=0) / (T_out - 1)
    ariv = np.mean(idiosyncratic_var / excessReturns.var().values)

    # Sharpe ratio of tangency portfolio
    sharpe_ratio = np.mean(optimal_returns) / np.sqrt(np.var(optimal_returns))

    return sharpe_ratio, oos_alpha, rms_alpha, idiosyncratic_var, ariv

def pca(excessReturns, gamma=-1, factors=3):
    """
    Computes the loadings, weights and beta for one iteration

    :param excessReturns: The excess returns matrix
    :param gamma: Penalty term
    :param factors: The number of factors
    :return: The weights for the tangent portfolio, the loading matrix and the betas
    """
    # Convert to numpy
    R = excessReturns.values
    T = R.shape[0]

    mean_excess_returns = R.mean(axis=0).reshape(-1, 1)

    # Equation (5) of LP
    Sigma_RP = (R.T @ R) / T + gamma * (mean_excess_returns @ mean_excess_returns.T)

    # PCA
    eigvals, eigvecs = np.linalg.eigh(Sigma_RP)
    idx = np.argsort(eigvals)[::-1][:factors]
    Lambda = eigvecs[:, idx]

    # Factor realizations
    F = R @ Lambda @ np.linalg.inv(Lambda.T @ Lambda)

    # Regression of returns on factors (exposures)
    X = np.hstack([np.ones((T, 1)), F])
    Q = np.linalg.inv(X.T @ X)
    B = Q @ X.T @ R
    beta = B[1:, :]

    # Tangency portfolio
    if factors == 1:
        w_tangent = np.array([[1.0]])
    else:
        Sigma_F = np.cov(F, rowvar=False)
        mu_F = F.mean(axis=0).reshape(-1, 1)
        iota = np.ones((factors, 1))
        w_tangent = np.linalg.inv(Sigma_F) @ mu_F / (iota.T @ np.linalg.inv(Sigma_F) @ mu_F)

    return Lambda, w_tangent, beta
