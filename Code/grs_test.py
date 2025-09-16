import numpy as np
import pandas as pd
from scipy.stats import f, t

def estimate_capm(marketReturns, assetReturns, riskfreeRate = 0.0025):
    """
    This function estimates parameters in the econometric CAPM model and their standard deviation.
    It also tests the alphas with the t-statistic, F-statistic and p-values.

    :param marketReturns: The excess market returns
    :param assetReturns: The returns of the assets
    :param riskfreeRate: the risk-free rate is set at 0.25%
    :return: the CAPM parameters, covariance matrix, standard deviation of intercept, t-statistic, F-statistic, corresponding p-values and the market Sharpe ratio.
    """

    assetReturns = assetReturns - riskfreeRate
    T = assetReturns.shape[0]
    n = assetReturns.shape[1]

    # Parameter estimation
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

    # F-test
    testPortfolio = ((T - n - 1) / (n * (T - 2) * Q[0,0])) * (alphas.T @ np.linalg.inv(covarianceMatrix) @ alphas)
    pValuePortfolio = 1 - f.cdf(testPortfolio, n, T - n - 1)

    # t-tests
    testIndividual = alphas / standardDeviationAlpha
    pValueIndividual = 1 - t.cdf(np.abs(testIndividual), df=T-2)

    # Market Sharpe ratio
    marketVariance = marketReturns.var()
    marketMean = marketReturns.mean()
    marketSharpeRatio = marketMean / np.sqrt(marketVariance)
    return alphas, betas, covarianceMatrix, standardDeviationAlpha, testPortfolio, pValuePortfolio, testIndividual, pValueIndividual, marketSharpeRatio

def estimate_factor_model(factorReturns, assetReturns, riskfreeRate = 0.0025):
    assetReturns = assetReturns - riskfreeRate
    T = assetReturns.shape[0]
    n = assetReturns.shape[1]
    k = factorReturns.shape[1]

    # Parameter estimation
    iota = np.ones_like(assetReturns, shape=(T,1))
    F = np.column_stack([iota, factorReturns])
    Q = np.linalg.inv(F.T @ F)
    parameters = Q @ F.T @ assetReturns
    alphas = parameters.iloc[0]
    betas = parameters.iloc[1:4].T

    yHat = (F @ parameters).to_numpy()
    yHat = pd.DataFrame(
        yHat,
        index=assetReturns.index,
        columns=assetReturns.columns
    )

    errors = assetReturns - yHat

    covarianceMatrix = (errors.T @ errors) / (T - k - 1)
    varianceResidual = np.diagonal(covarianceMatrix)
    standardDeviationAlpha = np.sqrt(Q[0, 0] * varianceResidual)

    # t-test
    testIndividual = alphas / standardDeviationAlpha
    pValueIndividual = 1 - t.cdf(np.abs(testIndividual), df=T - k - 1)

    # F-test
    populationVariance = ((T - k - 1)/T) * covarianceMatrix
    populationFactorVariance = factorReturns.cov(ddof=0)
    meanFactors = factorReturns.mean(axis=0).T

    testFactor = (((T - n - k)/ (n * (1 + meanFactors.T @ np.linalg.inv(populationFactorVariance) @ meanFactors)))
                  * (alphas.T @ np.linalg.inv(populationVariance) @ alphas))
    pValueFactor = 1 - f.cdf(testFactor, n, T - n - k)

    # Factor Sharpe ratio
    factorSharpeRatio = np.sqrt(meanFactors.T @ np.linalg.inv(populationFactorVariance) @ meanFactors)

    # Pricing error indicators
    assetVariance = assetReturns.var()
    rmsAlpha = np.sqrt(np.mean(alphas**2))
    arivFactor = np.mean(varianceResidual / assetVariance)

    return  alphas, betas, covarianceMatrix, standardDeviationAlpha, testFactor, pValueFactor, testIndividual, pValueIndividual, factorSharpeRatio, rmsAlpha, arivFactor