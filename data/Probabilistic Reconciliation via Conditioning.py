import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from properscoring import crps_ensemble
from bayesreconpy.hierarchy import get_reconc_matrices, temporal_aggregation
from bayesreconpy.reconc_BUIS import reconc_BUIS
"""
# Sample data (replace with your actual data)
carparts_example = pd.read_pickle('carparts_example.pkl')

flat_data = carparts_example.values.flatten()
flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaN values for plotting

# Generate a date range for the x-axis labels
# Start from "1998-01", with monthly frequency, matching the length of the data
date_index = pd.date_range(start="1998-01", periods=len(flat_data), freq="M")

# Create the layout: 1 row, 2 columns with custom widths
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

# Time series plot in the first subplot with timestamps on x-axis
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(date_index, flat_data, color='black')  # Single line plot with timestamps
ax1.set_xlabel("Time")
ax1.set_ylabel("Car part sales")
ax1.set_title("Time Series of Car Part Sales")
ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

# Histogram in the second subplot
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(flat_data, bins=15, color='gray', edgecolor='black')
ax2.set_xlabel("Car part sales")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Car Part Sales")

plt.tight_layout()
plt.show()


# Convert the DataFrame to a Series with a datetime index
carparts_example = carparts_example.stack()  # Convert to long format
carparts_example.index = pd.to_datetime(
    [f"{year}-{month}" for year, month in carparts_example.index], format="%Y-%b"
)

# Now split into train and test sets
train = carparts_example[:'2001-03']
test = carparts_example['2001-04':]

agg_levels = {
    "Annual": train.resample("A").sum(),      # Annual aggregation
    "Biannual": train.resample("6M").sum(),   # Biannual aggregation (every 6 months)
    "4-Monthly": train.resample("4M").sum(),  # Every 4 months
    "Quarterly": train.resample("Q").sum(),   # Quarterly aggregation
    "2-Monthly": train.resample("2M").sum(),  # Bi-monthly (every 2 months)
    "Monthly": train.resample("M").sum()      # Monthly aggregation
}


# Define the layout for the plots with adjusted spacing
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns
fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between plots

# Plot each aggregated time series in its own subplot
for ax, (level, data) in zip(axes.flatten(), agg_levels.items()):
    ax.plot(data, color='black')  # Change line color to black for consistency
    ax.set_xlabel("Time")
    ax.set_ylabel("Car part sales")
    ax.set_title(level, fontsize=12, fontweight='bold')  # Increase title font size and weight
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels if needed

    # Reduce the number of x-axis ticks for readability, if there are too many
    if level in ["Monthly", "2-Monthly"]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Adjust number of ticks for readability

plt.show()

fc_samples = pd.read_pickle('fc_samples.pkl')

agg_levels = [2, 3, 4, 6, 12]
h = 12
recon_matrices = get_reconc_matrices(agg_levels, h)

# Aggregation matrix A
A = recon_matrices['A']

np.random.seed(42)
recon_res = reconc_BUIS(A, fc_samples, in_type="samples", distr="discrete", seed=42)

print("Dimensions of reconciled_samples:", recon_res['reconciled_samples'].shape)


ae_fc = []
ae_reconc = []
crps_fc = []
crps_reconc = []

# Loop over each point in the test forecast horizon
for h in range(len(test)):
    # Retrieve the test observation for the current forecast step
    test_value = test.iloc[h]  # Use iloc to access the h-th element

    # Median of the base forecast and reconciled forecast at horizon h
    y_hat = np.median(fc_samples[len(A) + h])  # Base forecast median
    y_reconc = np.median(recon_res['bottom_reconciled_samples'][h, :])  # Reconciled forecast median

    # Compute Absolute Errors
    ae_fc.append(abs(test_value - y_hat))
    ae_reconc.append(abs(test_value - y_reconc))

    # Compute Continuous Ranked Probability Score (CRPS)
    crps_fc.append(crps_ensemble(test_value, fc_samples[len(A) + h]))
    crps_reconc.append(crps_ensemble(test_value, recon_res['bottom_reconciled_samples'][h, :]))

# Calculate Mean Absolute Error (MAE) and mean CRPS for base and reconciled forecasts
mae_fc = np.mean(ae_fc)
mae_reconc = np.mean(ae_reconc)
crps_fc_mean = np.mean(crps_fc)
crps_reconc_mean = np.mean(crps_reconc)

# Create DataFrame to store the metrics
metrics = pd.DataFrame({
    "base.forecasts": [mae_fc, crps_fc_mean],
    "reconciled.forecasts": [mae_reconc, crps_reconc_mean]
}, index=["MAE", "CRPS"])

print(metrics)
"""
## Temporal hierarchy over a smooth time series

M3_example = pd.read_pickle('M3_example.pkl')

agg_levels = [12, 6, 4, 3, 2, 1]  # Corresponding to Annual, Biannual, etc.
train_agg = temporal_aggregation(M3_example['train'], agg_levels)

# Rename the aggregated levels for easier reference
levels = ["Annual", "Biannual", "4-Monthly", "Quarterly", "2-Monthly", "Monthly"]
train_agg = dict(zip(levels, train_agg.values()))


import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import norm

# Define seasonal periods based on aggregation level names
seasonal_periods_map = {
    "Annual": None,  # No seasonality for annual aggregation
    "Biannual": 2,
    "4-Monthly": 3,
    "Quarterly": 4,
    "2-Monthly": 6,
    "Monthly": 12
}

H = len(M3_example['test'].to_numpy().flatten())  # Forecast horizon (18 in this case)

fc = []  # List to store forecasts
fc_idx = 0

for level_name, level_data in train_agg.items():
    # Get seasonal periods based on aggregation level
    seasonal_periods = seasonal_periods_map[level_name]
    data_length = len(level_data)

    # Decide model type based on data length and seasonal periods
    if seasonal_periods is None or data_length < 2 * seasonal_periods:
        # If data is insufficient for seasonality, use a simpler model without seasonality
        print(f"Insufficient data for seasonal model at {level_name}. Using trend-only model.")
        model = ExponentialSmoothing(level_data, trend="add", seasonal=None).fit()
    else:
        # Use additive seasonality when data has sufficient cycles
        model = ExponentialSmoothing(level_data, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()

    # Generate forecast horizon for each level within 18 months
    h = int(np.floor(H / (12 / (seasonal_periods if seasonal_periods else 1))))
    print(f"Forecasting at {level_name}, h={h}...")

    # Forecasting and calculating 95% confidence intervals manually
    level_fc = model.forecast(h)
    forecast_se = np.sqrt(model.sse / len(level_data))  # Standard error of the forecast

    # Save mean and standard deviation of Gaussian predictive distribution
    for i in range(h):
        mean_forecast = level_fc.iloc[i]

        # Calculate the 95% confidence interval manually
        upper_95 = mean_forecast + norm.ppf(0.975) * forecast_se
        sd_forecast = (upper_95 - mean_forecast) / norm.ppf(0.975)

        fc.append({"mean": mean_forecast, "sd": sd_forecast})
        fc_idx += 1




