import numpy as np
import pandas as pd

def pca(assetReturns, gamma=-1, factors=3, riskfreeRate = 0.0025):
    """
    This method performs the RP-PCA on the returns of the asset.

    :param assetReturns: The returns of the assets
    :param gamma: Penalty parameter
    :param factors: Factors in the model
    :param riskfreeRate: Risk-free rate
    :return: loading matrix, alpha, idiosyncratic variance and
    """
    excessReturns = assetReturns - riskfreeRate
    T = excessReturns.shape[0]
    n = excessReturns.shape[1]

    mean_excess_returns = excessReturns.mean(axis=0).T

    # Equation (5) of LP
    Sigma_RP = (1/T)*(excessReturns.T @ excessReturns) + gamma*(mean_excess_returns @ mean_excess_returns.T)

    # Perform PCA
    eigvals, eigvecs = np.linalg.eigh(Sigma_RP)
    idx = np.argsort(eigvals)[::-1][:factors]
    Lambda = eigvecs[:, idx]

    # Get factors
    F = excessReturns @ Lambda @  np.linalg.inv(Lambda.T @ Lambda)

    # Get coefficients of regression
    X = F.copy()
    X.insert(0, "a", 1.0)
    Q = np.linalg.inv(X.T @ X)
    B = excessReturns.T @ X @ Q

    alpha = B.T.iloc[0]

    # Errors
    yHat = (X.to_numpy() @ B.T.to_numpy())
    yHat = pd.DataFrame(
        yHat,
        index=assetReturns.index,
        columns=assetReturns.columns
    )
    errors = excessReturns - yHat
    variance_residual = errors.var(ddof=factors+1)
    idiosyncratic_var = np.mean(variance_residual / excessReturns.var())

    standard_deviation_alphas = np.sqrt(Q[0,0]*variance_residual)
    root_mean_squared_error_alpha = np.sqrt(alpha.T @ alpha /n)

    # Sharpe ratio
    Sigma_F = F.cov()
    mu_F = F.mean(axis=0).T
    b_MV = np.linalg.inv(Sigma_F) @ mu_F

    sharpe_ratio = np.sqrt(mu_F.T @ b_MV)

    return Lambda, idiosyncratic_var, sharpe_ratio, alpha, standard_deviation_alphas, root_mean_squared_error_alpha


