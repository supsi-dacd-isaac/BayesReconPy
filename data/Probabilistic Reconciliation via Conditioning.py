import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from properscoring import crps_ensemble
from bayesreconpy.hierarchy import _get_reconc_matrices, _temporal_aggregation
from bayesreconpy.reconc_buis import reconc_buis
from bayesreconpy.reconc_gaussian import reconc_gaussian
from scipy.stats import norm


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
recon_matrices = _get_reconc_matrices(agg_levels, h)

# Aggregation matrix A
A = recon_matrices['A']

np.random.seed(42)
recon_res = reconc_buis(A, fc_samples, in_type="samples", distr="discrete", seed=42)

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

## Temporal hierarchy over a smooth time series

M3_example = pd.read_pickle('M3_example.pkl')
train_data = M3_example['train'].stack()

date_index = pd.date_range(start="1990-01", periods=len(train_data), freq="M")

# Convert the flattened data to a Series with a datetime index
train_series = pd.Series(train_data.values, index=date_index)

# Plotting the continuous time series
plt.figure(figsize=(10, 6))
plt.plot(train_series.index, train_series.values, color="black")
plt.xlabel("Time")
plt.ylabel("y")
plt.title("N1485")
plt.show()

agg_levels = [12, 6, 4, 3, 2, 1]  # Corresponding to Annual, Biannual, etc.
train_agg = _temporal_aggregation(M3_example['train'], agg_levels)

# Rename the aggregated levels for easier reference
levels = ["Annual", "Biannual", "4-Monthly", "Quarterly", "2-Monthly", "Monthly"]
train_agg = dict(zip(levels, train_agg.values()))

fc = pd.read_pickle('fc.pkl')

agg_levels = [2, 3, 4, 6, 12]
h = 18

# Generate the reconciliation matrices
rmat = _get_reconc_matrices(agg_levels=agg_levels, h=h)

# Prepare the matrix A for plotting
matrix_A = rmat['A']
# Reverse rows for display to align with R's `apply(t(rmat$A),1,rev)`
matrix_A_transformed = np.flip(matrix_A, axis=0)

# Plot the matrix
plt.figure(figsize=(6, 6))
plt.imshow(matrix_A_transformed, aspect='auto', cmap="Greys", origin="lower")
plt.xlabel(levels[5])
plt.xticks(ticks=np.arange(matrix_A.shape[1]), labels=np.arange(1, matrix_A.shape[1] + 1), rotation=90)
plt.yticks(ticks=[23, 22, 19, 15, 9], labels=levels[:5])
plt.colorbar(label="Matrix Values")

plt.show()


base_forecasts_mu = np.array([f["mean"] for f in fc])
base_forecasts_sigma = np.diag([f["sd"] ** 2 for f in fc])  # Variance matrix

# Reconcile Gaussian method
recon_gauss = reconc_gaussian(
    A=rmat['A'],
    base_forecasts_mu=base_forecasts_mu,
    base_forecasts_Sigma=base_forecasts_sigma
)

# Reconcile BUIS method
reconc_buis = reconc_buis(
    A=rmat['A'],
    base_forecasts=fc,
    in_type="params",
    distr="gaussian",
    num_samples=20000,
    seed=42
)

# Check consistency of results
bottom_reconciled_mean_gauss = np.dot(rmat['S'], recon_gauss['bottom_reconciled_mean'])
bottom_reconciled_mean_buis = np.mean(reconc_buis['reconciled_samples'], axis=1)

# Round and display results
comparison_results = np.round(np.vstack([
    bottom_reconciled_mean_gauss,
    bottom_reconciled_mean_buis
]))
print(comparison_results)




# Extract the last 18 forecasts for mean and standard deviation
yhat_mu = np.array([f["mean"] for f in fc[-18:]])
yhat_sigma = np.array([f["sd"] for f in fc[-18:]])

# Compute 95% confidence intervals
yhat_hi95 = norm.ppf(0.975, loc=yhat_mu, scale=yhat_sigma)
yhat_lo95 = norm.ppf(0.025, loc=yhat_mu, scale=yhat_sigma)

# Reconciled mean and 95% confidence intervals using quantiles
yreconc_mu = np.mean(reconc_buis['bottom_reconciled_samples'], axis=1)
yreconc_hi95 = np.quantile(reconc_buis['bottom_reconciled_samples'], 0.975, axis=1)
yreconc_lo95 = np.quantile(reconc_buis['bottom_reconciled_samples'], 0.025, axis=1)

# Define the monthly index for the training data (5 years * 12 months)
train_years = pd.date_range(start="1990-01-01", periods=60, freq="ME")

# Define the monthly index for the test data (filter out NaN values)
test_data_flattened = M3_example['test'].values.flatten()
test_data_non_nan = test_data_flattened[~np.isnan(test_data_flattened)]
test_years = pd.date_range(start="1994-04-01", periods=len(test_data_non_nan), freq="ME")

# Determine plot limits
ylim_min = min(M3_example['train'].min().min(), test_data_non_nan.min(), yhat_lo95.min(), yreconc_lo95.min()) - 1
ylim_max = max(M3_example['train'].max().max(), test_data_non_nan.max(), yhat_hi95.max(), yreconc_hi95.max()) + 1

# Plotting
plt.figure(figsize=(10, 6))

# Plot the training data
plt.plot(train_years, M3_example['train'].values.flatten(), color="black", label="Training Data")

# Plot the test data
plt.plot(test_years, test_data_non_nan, linestyle="--", color="gray", label="Test Data")

# Plot yhat forecasts and confidence interval
plt.plot(test_years, yhat_mu, color="coral", linewidth=2, label="Forecast (yhat)")
plt.fill_between(test_years, yhat_lo95, yhat_hi95, color="#FF7F5066", edgecolor="#FF7F5066", label="Forecast 95% CI")

# Plot reconciled forecasts and confidence interval
plt.plot(test_years, yreconc_mu, color="blue", linewidth=2, label="Reconciled Forecast")
plt.fill_between(test_years, yreconc_lo95, yreconc_hi95, color="#0000EE4D", edgecolor="#0000EE4D", label="Reconciled 95% CI")

# Configure plot
plt.ylim(ylim_min, ylim_max)
plt.xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("1995-08-01"))
plt.ylabel("y")
plt.title("N1485 Forecasts")
plt.legend()
plt.show()

infantMortality = pd.read_pickle('infantMortality.pkl')

fc = pd.read_pickle('fc_infantMortality.pkl')
residuals = pd.read_pickle('residuals_infantMortality.pkl')

# Define the matrix A
A = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
])

# Plotting the matrix A
plt.figure(figsize=(8, 6))
plt.imshow(A, cmap="Greys", aspect="auto")  # Removed np.flipud to avoid inversion

# Setting axis labels
bottom_labels = list(infantMortality.keys())[11:27]  # Adjust as necessary for actual labels
upper_labels = list(infantMortality.keys())[:11]     # Adjust as necessary for actual labels

# Customize ticks
plt.xticks(ticks=np.arange(A.shape[1]), labels=bottom_labels, rotation=90)
plt.yticks(ticks=np.arange(A.shape[0]), labels=upper_labels)  # No reversal for y-axis

# Display the plot
plt.xlabel("Bottom Time Series")
plt.ylabel("Upper Time Series")
plt.title("Hierarchical Structure Matrix A")
plt.tight_layout()
plt.show()


from bayesreconpy.shrink_cov import _schafer_strimmer_cov  # Replace with the actual import path if different

# Means
mu = np.array([fcast[0] for fcast in fc.values()])  # Extracting the means from each forecast entry

# Shrinkage covariance
shrink_res = _schafer_strimmer_cov(residuals)  # Apply shrinkage covariance estimation
lambda_star = shrink_res['lambda_star']
Sigma = shrink_res['shrink_cov']

print(f"The estimated shrinkage intensity is {round(lambda_star, 3)}")

# Perform Gaussian reconciliation
recon_gauss = reconc_gaussian(A=A, base_forecasts_mu=mu, base_forecasts_Sigma=Sigma)

# Extract reconciled means and covariances
bottom_mu_reconc = recon_gauss['bottom_reconciled_mean']
bottom_Sigma_reconc = recon_gauss['bottom_reconciled_covariance']

# Calculate the reconciled mu and Sigma for the upper variable
upper_mu_reconc = A @ bottom_mu_reconc  # Matrix multiplication
upper_Sigma_reconc = A @ bottom_Sigma_reconc @ A.T

print("Upper reconciled mean:", upper_mu_reconc)
