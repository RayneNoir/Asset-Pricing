import utils as ut
import rolling_window as rw
import pandas as pd

# loading in the relevant datasets
returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)
factors = ut.open_file('F-F_Research_Data_5_Factors_2x3.csv', percentage=True)
factors.drop(columns=['HML','RMW','RF'], inplace=True)

# Parameters
gamma = 20      # Test with -1, 0, 20
K = 3           # Test with 1,3

loading_factors, out_of_sample_factors, sharpe_ratio, oos_alpha, rms_alpha, idiosyncratic_var, ariv = rw.rolling_window(
    returns,
    gamma=gamma,
    factors=K
)

sharpe_ratio_CAPM, oos_alpha_CAPM, rms_alpha_CAPM, idiosyncratic_var_CAPM, ariv_CAPM = rw.rolling_window_other_models(returns, factors['Mkt-RF'])
sharpe_ratio_three_factor, oos_alpha_three_factor, rms_alpha_three_factor, idiosyncratic_var_three_factor, ariv_three_factor = rw.rolling_window_other_models(returns, factors)

result = pd.DataFrame({
    'OOS Alpha [%]': 100*oos_alpha,
},index=returns.columns).round({'OOS Alpha [%]': 2})

result_CAPM = pd.DataFrame({
    'OOS Alpha [%]': 100*oos_alpha_CAPM,
},index=returns.columns).round({'OOS Alpha [%]': 2})

result_three_factor = pd.DataFrame({
    'OOS Alpha [%]': 100*oos_alpha_three_factor,
},index=returns.columns).round({'OOS Alpha [%]': 2})

print(50*"-")
print(f"OOS RP-PCA results (gamma={gamma:.0f}, factors={K:.0f})")
print(50*"-")
print(result)
print(f"Sharpe ratio: {sharpe_ratio:.2f}"
      f"\nAverage relative idiosyncratic variance (ARIV): {100*ariv:.2f}%"
      f"\nOOS RMS pricing error: {rms_alpha*100:.2f}%")
print(50*"-")
print(f"OOS CAPM results")
print(50*"-")
print(result_CAPM)
print(f"Sharpe ratio: {sharpe_ratio_CAPM:.2f}"
      f"\nAverage relative idiosyncratic variance (ARIV): {100*ariv_CAPM:.2f}%"
      f"\nOOS RMS pricing error: {rms_alpha_CAPM*100:.2f}%")
print(50*"-")
print(f"OOS three factor results")
print(50*"-")
print(result_three_factor)
print(f"Sharpe ratio: {sharpe_ratio_three_factor:.2f}"
      f"\nAverage relative idiosyncratic variance (ARIV): {100*ariv_three_factor:.2f}%"
      f"\nOOS RMS pricing error: {rms_alpha_three_factor*100:.2f}%")
print(50*"-")