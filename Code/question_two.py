import utils as ut
import grs_test as grs
import pandas as pd

# loading in the relevant datasets
returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)
factors = ut.open_file('F-F_Research_Data_5_Factors_2x3.csv', percentage=True)
factors.drop(columns=['HML','RMW','RF'], inplace=True)

# CAPM and test assets
alphas, betas, sigmaMatrix, stdAlphas, testPortfolio, pValuePortfolio, testIndividual, pValueIndividual, marketSharpeRatio = grs.estimate_capm(factors['Mkt-RF'], returns)
result = pd.DataFrame({
    'Alpha [%]': 100*alphas.values.flatten(),
    'Std [%]': stdAlphas*100,
    't-statistic': testIndividual,
    'p-value': pValueIndividual,
}, index=alphas.index).round({'Alpha [%]': 2, 'Std [%]': 2, 't-statistic': 3, 'p-value': 3})

# 3 factor model
(alphasFactor, betasFactor, sigmaFMatrix, stdAlphaFactor, testFactors, pValueFactors, testIndividualFactor, pValueIndividualFactor,
 factorSharpeRatio, rmsAlphaFactor, arivFactor) = grs.estimate_factor_model(factors, returns, riskfreeRate = 0.0025)
resultFactor = pd.DataFrame({
    'Alpha [%]': 100*alphasFactor.values.flatten(),
    'Std [%]': stdAlphaFactor*100,
    't-statistic': testIndividualFactor,
    'p-value': pValueIndividualFactor,
}, index=alphasFactor.index).round({'Alpha [%]': 2, 'Std [%]': 2, 't-statistic': 3, 'p-value': 3})

# The results
print(50*"-")
print('CAPM results')
print(50*"-")
print(result)
print(f"CAPM: F-statistic: {testPortfolio:.3f}, p-value: {pValuePortfolio:.3f} \nMarket Sharpe ratio: {marketSharpeRatio:.2f}")
print(50*"-")
print('3 Factor model')
print(50*"-")
print(resultFactor)
print(f"Factor model: F-statistic: {testFactors:.3f}, p-value: {pValueFactors:.3f} \nFactor model Sharpe ratio: {factorSharpeRatio:.2f}"
      f"\nAverage relative idiosyncratic variance (ARIV): {100*arivFactor:.2f}%, RMS pricing error: {rmsAlphaFactor*100:.2f}%")
print(50*"-")
