import utils as ut
import grs_test as grs
import pandas as pd

# loading in the relevant datasets
returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)
factors = ut.open_file('F-F_Research_Data_5_Factors_2x3.csv', percentage=True)
factors.drop(columns=['HML','RMW','RF'], inplace=True)

# CAPM and test assets
alphas, betas, sigmaMatrix, stdAlphas, ariv, rms, testPortfolio, pValuePortfolio, testIndividual, pValueIndividual = grs.estimate_capm(factors['Mkt-RF'], returns)
result = pd.DataFrame({
    'Alpha [%]': 100*alphas.values.flatten(),
    'Std [%]': stdAlphas*100,
    't-statistic': testIndividual,
    'p-value': pValueIndividual,
}, index=alphas.index).round({'Alpha [%]': 2, 'Std [%]': 2, 't-statistic': 3, 'p-value': 3})
print('CAPM results')
print(result)
print(f"CAPM: F-statistic: {testPortfolio:.3f}, p-value: {pValuePortfolio:.3f}")