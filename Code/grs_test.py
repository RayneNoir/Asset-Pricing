import numpy as np
import pandas as pd
from scipy.stats import f, t

def estimate_capm(marketReturns, assetReturns, riskfreeRate = 0.0025):
    assetReturns = assetReturns - riskfreeRate
    T = assetReturns.shape[0]
    n = assetReturns.shape[1]
    assetVariance = assetReturns.var()

    iota = np.ones_like(marketReturns)

    X = np.column_stack([iota, marketReturns])
    Q = np.linalg.inv(X.T @ X)
    parameters = Q @ X.T @ assetReturns
    alphas = parameters.iloc[0]
    betas = parameters.iloc[1]

    yHat = (X @ parameters).to_numpy()
    yHat = pd.DataFrame(
        yHat,
        index=assetReturns.index,
        columns=assetReturns.columns
    )

    errors = assetReturns - yHat

    covarianceMatrix = (errors.T @ errors) / (T - 2)

    varianceResidual = np.diagonal(covarianceMatrix)
    standardDeviationAlpha = np.sqrt(Q[0,0]*varianceResidual)

    ariv = np.mean(varianceResidual / assetVariance)
    rms = np.sqrt(np.mean(alphas**2))

    # F-test
    testPortfolio = ((T - n - 1) / (n * (T - 2) * Q[0,0])) * (alphas.T @ np.linalg.inv(covarianceMatrix) @ alphas)
    pValuePortfolio = 1 - f.cdf(testPortfolio, n, T - n - 1)

    # t-tests
    testIndividual = alphas / standardDeviationAlpha
    pValueIndividual = 1 - t.cdf(np.abs(testIndividual), df=T-2)

    return alphas, betas, covarianceMatrix, standardDeviationAlpha, ariv, rms, testPortfolio, pValuePortfolio, testIndividual, pValueIndividual
