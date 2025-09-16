import efficient_frontier as ef
import utils as ut
import matplotlib.pyplot as plt

def plot_frontier(portfolioVolatilities, portfolioReturns, portfolioVolatilitiesRiskless, portfolioReturnsRiskless,
                  factorVolatilitiesRiskless, factorReturnsRiskless, volatilityTangent, muTangent, volatilityTangentFactor, muTangentFactor):

    plt.figure(figsize=(8, 6))
    plt.plot(100*portfolioVolatilities, 100*portfolioReturns)
    plt.plot(100*portfolioVolatilitiesRiskless, 100*portfolioReturnsRiskless)
    plt.plot(100*factorVolatilitiesRiskless, 100*factorReturnsRiskless)
    plt.plot(100*portfolioVolatilities[0], 100*portfolioReturns[0], marker='x')
    plt.plot(100*volatilityTangent, 100*muTangent, marker='o')
    plt.plot(100*volatilityTangentFactor, 100*muTangentFactor, marker='^')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility [%]')
    plt.ylabel('Total Returns [%]')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.legend(['(1) Portfolios without riskless asset', '(2) Portfolios + riskless asset',
                '(3) Excess market returns + FF factors + riskless asset',
                'GMV portfolio of (1)', 'Tangency portfolio of (2)', 'Tangency portfolio of (3)'])
    plt.grid(True)
    plt.show()
    plt.savefig('Efficient frontier.png')

def main(plot = True):
    # loading in the relevant datasets
    returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)
    factors = ut.open_file('F-F_Research_Data_5_Factors_2x3.csv', percentage=True)
    factors.drop(columns=['HML','RMW','RF'], inplace=True)

    # Frontier 1 (without riskless)
    portfolioReturns, portfolioVolatilities = ef.compute_efficient_frontier(returns, riskless=False, excess=False)

    # Frontier 2 (with riskless)
    (portfolioReturnsRiskless, portfolioVolatilitiesRiskless,
     muTangent, volatilityTangent) = ef.compute_efficient_frontier(returns, riskless=True, excess=False)

    # Frontier 3 (Factors)
    (factorReturnsRiskless, factorVolatilitiesRiskless,
     muTangentFactor, volatilityTangentFactor) = ef.compute_efficient_frontier(factors, riskless=True, excess=True)

    # Plot frontier
    if plot:
        plot_frontier(portfolioVolatilities, portfolioReturns, portfolioVolatilitiesRiskless, portfolioReturnsRiskless,
                          factorVolatilitiesRiskless, factorReturnsRiskless, volatilityTangent, muTangent,
                          volatilityTangentFactor, muTangentFactor)

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

    main(plot=False)
