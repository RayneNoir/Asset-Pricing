import efficient_frontier as ef
import utils as ut
import matplotlib.pyplot as plt

from efficient_frontier import compute_points


def plot_frontier(portfolioVolatilities, portfolioReturns, portfolioVolatilitiesRiskless, portfolioReturnsRiskless,
                  factorVolatilitiesRiskless, factorReturnsRiskless, meanPortfolios, stdPortfolios, meanFactors, stdFactors,
                  volatilityTangent, muTangent, volatilityTangentFactor, muTangentFactor, muGMV, volatilityGMV,
                  factorsReturns, factorsVolatilities, muFactorsGMV, volatilityFactorsGMV):

    plt.figure(figsize=(8, 6))
    plt.scatter(100*stdPortfolios, 100*meanPortfolios, color='b')
    plt.scatter(100*stdFactors, 100*meanFactors, color='r')
    plt.plot(100*portfolioVolatilities, 100*portfolioReturns,'b')
    plt.plot(100*portfolioVolatilitiesRiskless, 100*portfolioReturnsRiskless,color='orange',linestyle='--')
    plt.plot(100 * factorsVolatilities, 100 * factorsReturns, color='olive')
    plt.plot(100*factorVolatilitiesRiskless, 100*factorReturnsRiskless,'g--')
    plt.scatter(100*volatilityGMV, 100*muGMV, color='r', marker='x')
    plt.scatter(100 * volatilityFactorsGMV, 100 * muFactorsGMV, color='b', marker='x')
    plt.scatter(100*volatilityTangent, 100*muTangent, color='k', marker='o')
    plt.scatter(100*volatilityTangentFactor, 100*muTangentFactor, color='brown', marker='^')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility [%]')
    plt.ylabel('Total Returns [%]')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.legend(['25 Portfolios','3 factors','(1) 25 Portfolios', '(2) 25 Portfolios + riskless asset',
                '(3) Market returns + SMB + CMA', '(4) Market returns + SMB + CMA + riskless asset',
                'GMV portfolio of (1)', 'GMV portfolio of (4)', 'Tangency portfolio of (2)', 'Tangency portfolio of (3)'])
    plt.grid(True)
    plt.show()
    plt.savefig('Efficient frontier.png')

def main(plot = True):
    # loading in the relevant datasets
    returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)
    factors = ut.open_file('F-F_Research_Data_5_Factors_2x3.csv', percentage=True)
    factors.drop(columns=['HML','RMW','RF'], inplace=True)

    # Individual assets, factors
    meanPortfolios, stdPortfolios = compute_points(returns)
    meanFactors, stdFactors = compute_points(factors)

    # Frontier 1 (without riskless)
    portfolioReturns, portfolioVolatilities, muGMV, volatilityGMV = ef.compute_efficient_frontier(returns, riskless=False, excess=False)

    # Frontier 2 (with riskless)
    (portfolioReturnsRiskless, portfolioVolatilitiesRiskless,
     muTangent, volatilityTangent) = ef.compute_efficient_frontier(returns, riskless=True, excess=False)

    # Frontier 3 (Factors with riskless)
    (factorReturnsRiskless, factorVolatilitiesRiskless,
     muTangentFactor, volatilityTangentFactor) = ef.compute_efficient_frontier(factors, riskless=True, excess=True)

    # Frontier 4 (Factor without riskless)
    factorsReturns, factorsVolatilities, muFactorsGMV, volatilityFactorsGMV = ef.compute_efficient_frontier(factors,
                                                                                                  riskless=False,
                                                                                                  excess=True)

    # Plot frontier
    if plot:
        plot_frontier(portfolioVolatilities, portfolioReturns, portfolioVolatilitiesRiskless, portfolioReturnsRiskless,
                          factorVolatilitiesRiskless, factorReturnsRiskless, meanPortfolios, stdPortfolios, meanFactors, stdFactors,
                      volatilityTangent, muTangent, volatilityTangentFactor, muTangentFactor, muGMV, volatilityGMV,
                      factorsReturns, factorsVolatilities, muFactorsGMV, volatilityFactorsGMV)

    # Tangency portfolios
    print(21 * '-')
    print('Tangency Portfolios')
    print(21 * '-')
    print('| 25 portfolios + riskless asset |')
    print(21 * '-')
    print(f"Total Return = {100*muTangent:.2f}%")
    print(f"Volatility   = {100*volatilityTangent:.2f}%")
    print(f"Sharpe Ratio = {(muTangent / volatilityTangent):.2f}")
    print(21 * '-')
    print('| Excess market returns + FF factors + riskless asset |')
    print(21 * '-')
    print(f"Total Return = {100*muTangentFactor:.2f}%")
    print(f"Volatility   = {100*volatilityTangentFactor:.2f}%")
    print(f"Sharpe Ratio = {(muTangentFactor / volatilityTangentFactor):.2f}")
    print(21 * '-')

if __name__ == '__main__':
    plt.close('all')
    main(plot=True)