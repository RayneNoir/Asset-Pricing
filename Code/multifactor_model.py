import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling_window(assetReturns, multifactors, gamma=-1, threshold = 0.97, riskfreeRate=0.0025):
    k_threshold, cumulative_var_ratio, combined_factors, F = determine_factors(
        assetReturns,
        multifactors,
        gamma=gamma,
        threshold = threshold,
        riskfreeRate=riskfreeRate
    )

    T, n = combined_factors.shape
    T_out_of_sample = T - 240
    k = k_threshold

    prediction_error = np.zeros((T_out_of_sample, n))
    tangent_weights = np.zeros((T_out_of_sample, k))
    optimal_returns = np.zeros((T_out_of_sample, 1))
    loading_factors = []
    out_of_sample_factors = []

    for i in range(240, T):
        return_in_sample = combined_factors.iloc[i - 240:i]
        Lambda, w_tangent, exposure = pca(return_in_sample, gamma=gamma, factors=k)

        loading_factors.append(Lambda)
        tangent_weights[i - 240, :] = w_tangent.flatten()

        # Out-of-sample return
        out_of_sample_return = combined_factors.iloc[i].values
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
    ariv = np.mean(idiosyncratic_var / combined_factors.var().values)
    sharpe_ratio = np.mean(optimal_returns) / np.sqrt(np.var(optimal_returns))

    return sharpe_ratio, oos_alpha, rms_alpha, idiosyncratic_var, ariv, k_threshold, combined_factors

def determine_factors(assetReturns, multifactors, gamma=-1, threshold = 0.97, riskfreeRate=0.0025):
    excessReturns = assetReturns - riskfreeRate

    combined_factors = pd.concat([excessReturns, multifactors], axis=1, join='inner')

    # Convert to numpy
    R = combined_factors.values
    T = R.shape[0]

    mean_excess_returns = R.mean(axis=0).reshape(-1, 1)

    # Equation (5) of LP
    Sigma_RP = (R.T @ R) / T + gamma * (mean_excess_returns @ mean_excess_returns.T)

    # PCA
    eigvals, eigvecs = np.linalg.eigh(Sigma_RP)
    eigvals = np.sort(eigvals)[::-1]

    explained_var_ratio = eigvals / eigvals.sum()
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    k_threshold = int(np.argmax(cumulative_var_ratio >= threshold)) +1

    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(eigvals) + 1)

    plt.plot(x, explained_var_ratio, 'o-', label='Individual variance')

    # Horizontal line at the 97% threshold
    plt.axhline(y=float(explained_var_ratio[k_threshold-1]), color='red', linestyle='--', label=f'{int(threshold * 100)}% threshold')

    # Optional vertical line showing where it's reached
    plt.axvline(x=k_threshold, color='green', linestyle=':', label=f'k = {k_threshold}')

    plt.title(f"Scree Plot (γ = {gamma})")
    plt.xlabel("Factor number")
    plt.ylabel("Fraction of total variance explained")
    plt.xlim(1, 50)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"{k_threshold} factors needed to explain {100*cumulative_var_ratio[k_threshold]:.2f}% of the variance")

    idx = np.argsort(eigvals)[::-1][:k_threshold]
    Lambda = eigvecs[:, idx]

    return k_threshold, cumulative_var_ratio, combined_factors, Lambda

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

