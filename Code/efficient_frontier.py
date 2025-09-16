import numpy as np

def compute_efficient_frontier(returns, riskless=False, excess=False, riskFreeRate=0.0025):
    """
    This function computes the efficient frontier

    :param returns: The returns for which to compute the efficient frontier
    :param riskless: True if the portfolio includes a riskless asset, False otherwise
    :param excess: True if the returns are already transformed into excess returns, False otherwise
    :param riskFreeRate: The risk-free rate is set at 0.25%
    :return a list of portfolio returns and volatility for different target mean returns
    """
    if not excess:
        returns = returns - riskFreeRate

    muReturns = returns.mean(axis=0).T.to_numpy()
    sigmaReturns = returns.cov(ddof=1).to_numpy()
    inverseSigma = np.linalg.inv(sigmaReturns)
    iota = np.ones_like(muReturns)

    portfolioReturns = []
    portfolioVolatilities = []

    A = muReturns.T @ inverseSigma @ muReturns
    B = muReturns.T @ inverseSigma @ iota
    C = iota.T @ inverseSigma @ iota
    if riskless:
        weightsTangent = (inverseSigma @ muReturns)/B
        muTangent = weightsTangent.T @ muReturns + riskFreeRate
        volatilityTangent = np.sqrt(weightsTangent.T @ sigmaReturns @ weightsTangent)
        muTargetGrid = np.linspace(0, np.max(returns), 1000)
        for r in muTargetGrid:
            portfolioReturn, portfolioVolatility = compute_portfolio(r, muReturns,
                                                                      riskless, A, B, C,
                                                                      sigmaReturns, inverseSigma, iota,
                                                                      riskFreeRate)
            portfolioReturns.append(portfolioReturn)
            portfolioVolatilities.append(portfolioVolatility)
        return np.array(portfolioReturns), np.array(portfolioVolatilities), muTangent, volatilityTangent
    else:
        muTargetGrid = np.linspace(0, np.max(returns), 1000)
        muGMV = B/C + riskFreeRate
        volatilityGMV = np.sqrt(1/C)
        for r in muTargetGrid:
            portfolioReturn, portfolioVolatility = compute_portfolio(r, muReturns,
                                                                      riskless, A, B, C,
                                                                      sigmaReturns, inverseSigma, iota,
                                                                      riskFreeRate)
            portfolioReturns.append(portfolioReturn + riskFreeRate)
            portfolioVolatilities.append(portfolioVolatility)

        return np.array(portfolioReturns), np.array(portfolioVolatilities), muGMV, volatilityGMV

def compute_portfolio(muTarget, mu, riskless, A, B, C, sigmaReturns, inverseSigma, iota, riskFreeRate):
    """
    Computes the portfolio return and volatility for a given target mean return

    :param muTarget: The target mean return
    :param mu: The mean returns
    :param riskless: True if the portfolio includes a riskless asset, False otherwise
    :param A: a constant needed for calculation
    :param B: a constant needed for calculation
    :param C: a constant needed for calculation
    :param sigmaReturns: the covariance matrix of the returns
    :param inverseSigma: the inverse of the covariance matrix
    :param iota: vector of ones
    :param riskFreeRate: the risk-free rate
    """
    if riskless:
        muExcessTangency = A/B
        tangencyPortfolioWeights = (inverseSigma @ mu) / B

        lambda_weight = muTarget / muExcessTangency
        portfolioWeights = lambda_weight * tangencyPortfolioWeights

        portfolioReturns = riskFreeRate + portfolioWeights.T @ mu
    else:
        gmvPortfolioWeights = (inverseSigma @ iota) / C

        if B == 0:
            adjustedPortfolioWeights = inverseSigma @ mu
            zeta = muTarget / A
            portfolioWeights = zeta *adjustedPortfolioWeights + gmvPortfolioWeights

        else:
            lambda_weight = (B*C*muTarget - B**2) / (A*C - B**2)
            muPortfolioWeights = (inverseSigma @ mu) / B
            portfolioWeights = lambda_weight*muPortfolioWeights + (1-lambda_weight)*gmvPortfolioWeights
        portfolioReturns = portfolioWeights.T @ mu

    portfolioVariance = portfolioWeights.T @ sigmaReturns @ portfolioWeights
    portfolioVolatility = np.sqrt(portfolioVariance)

    return portfolioReturns, portfolioVolatility

def compute_points(returns):
    meanReturns = returns.mean(axis=0).T.to_numpy()
    stdReturns = np.sqrt(returns.var())
    return meanReturns, stdReturns