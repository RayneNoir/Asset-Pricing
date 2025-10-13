import utils as ut
import pandas as pd
import numpy as np
import multifactor_model as mm

# loading in the relevant datasets
returns = ut.open_file('25_Portfolios_ME_INV_5x5.csv', percentage=True)
multi_factors = ut.open_file('jkpfactors_US_all.csv', percentage=False)

# Parameter
threshold = 0.97
gamma = 20

sharpe_ratio, oos_alpha, rms_alpha, idiosyncratic_var, ariv, k_threshold,combined_factors = mm.rolling_window(returns,
                                                                                multi_factors,
                                                                                gamma= gamma,
                                                                                threshold = threshold
                                                                                )

result = pd.DataFrame({
    'OOS Alpha [%]': 100*oos_alpha,
},index=combined_factors.columns).round({'OOS Alpha [%]': 2})

print(50*"-")
print(f"OOS RP-PCA results (γ={gamma:.0f}, factors={k_threshold:.0f})")
print(50*"-")
print(result)
print(f"Mean alpha: {np.mean(oos_alpha*100):.2f}%"
      f"\nStd alpha: {np.std(oos_alpha)*100:.2f}%"
      f"\nSharpe ratio: {sharpe_ratio:.2f}"
      f"\nAverage relative idiosyncratic variance (ARIV): {100*ariv:.2f}%"
      f"\nOOS RMS pricing error: {rms_alpha*100:.2f}%")
print(50*"-")