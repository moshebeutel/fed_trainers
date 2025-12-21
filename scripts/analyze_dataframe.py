import pandas as pd
from datetime import datetime, timedelta

# Read the CSV file
df = pd.read_csv('csv/cifar10_sgd_dp.csv')

# Convert the timestamp column to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate yesterday's date
yesterday = (datetime.now() - timedelta(days=1)).date()
today = datetime.now().date()

# Filter the DataFrame for runs that occurred yesterday
# yesterday_runs = df[df['timestamp'].dt.date == yesterday]
today_runs = df[df['timestamp'].dt.date == today]

# Group the filtered data by 'noise-multiplier'
grouped_runs = today_runs.groupby('num-epochs')

# Example: Display the number of runs for each noise multiplier group
print("best_val_acc:")
print(grouped_runs['best_val_acc'].max())
print("test_avg_acc:")
print(grouped_runs['test_avg_acc'].max())
# Filter for rows where noise-multiplier is 0.2
specific_rows = today_runs[today_runs['num-epochs'] == 150]
print(specific_rows)
# specific_noise_clip_rows = specific_noise_rows[specific_noise_rows['clip'] == 0.01]
#
# specific_noise_clip_agg_rows = specific_noise_clip_rows[specific_noise_clip_rows['num-client-agg'] == 10]
#
# # Print all columns for these rows
# pd.set_option('display.max_columns', None) # Ensure all columns are visible
# pd.set_option('display.width', 1000)       # Adjust width for better display
# print(specific_noise_clip_agg_rows)