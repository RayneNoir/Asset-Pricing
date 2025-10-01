import utils as ut
import rp_pca as rp
import pandas as pd

# loading in the relevant datasets
returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)

# Parameters
gamma = -1      # Test with -1, 0, 20
K = 1           # Test with 1,3

Lambda, ariv, sharpe_ratio, alphas, std_alphas, rms_alpha = rp.pca(returns, gamma=gamma, factors=K)
result = pd.DataFrame({
    'Alpha [%]': 100*alphas,
    'Std [%]': std_alphas*100,
},index=alphas.index).round({'Alpha [%]': 2, 'Std [%]': 2})

print(50*"-")
print(f"RP-PCA results (gamma={gamma:.0f}, factors={K:.0f})")
print(50*"-")
print(result)
print(f"Sharpe ratio: {sharpe_ratio:.2f}"
      f"\nAverage relative idiosyncratic variance (ARIV): {100*ariv:.2f}%, RMS pricing error: {rms_alpha*100:.2f}%")
print(50*"-")