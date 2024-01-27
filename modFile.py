import pandas as pd
import numpy as np
from scipy.stats import entropy

input_file_path = "9541.csv"
output_file_path = "9541_mod.csv"

columns_to_keep = ["animal_id", "timestamp", "acc_x", "acc_y", "acc_z", "temperature", "Class"]

df = pd.read_csv(input_file_path, delimiter=';')

df_mod = df[columns_to_keep].copy()  

df_mod['timestamp'] = pd.to_datetime(df_mod['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

df_mod['acc_y'] = -df_mod['acc_y']
df_mod['acc_z'] = -df_mod['acc_z']

df_mod['pitch'] = np.round(np.arctan2(-df_mod['acc_x'], np.sqrt(df_mod['acc_y']**2 + df_mod['acc_z']**2)) * (180/np.pi), 2)
df_mod['roll'] = np.round(np.arctan2(df_mod['acc_y'], df_mod['acc_z']) * (180/np.pi), 2)
df_mod['inclination'] = np.round(np.arctan2(np.sqrt(df_mod['acc_x']**2 + df_mod['acc_y']**2), df_mod['acc_z']) * (180/np.pi), 2)

# Calculate average, minimum, maximum, and standard deviation values for the last 30 seconds for each accelerometer axis and round to 2 decimals
df_mod['avg_acc_x'] = np.round(df_mod['acc_x'].rolling(window=30).mean(), 2)
df_mod['min_acc_x'] = df_mod['acc_x'].rolling(window=30).min()
df_mod['max_acc_x'] = df_mod['acc_x'].rolling(window=30).max()
df_mod['std_acc_x'] = np.round(df_mod['acc_x'].rolling(window=30).std(), 2)

df_mod['avg_acc_y'] = np.round(df_mod['acc_y'].rolling(window=30).mean(), 2)
df_mod['min_acc_y'] = df_mod['acc_y'].rolling(window=30).min()
df_mod['max_acc_y'] = df_mod['acc_y'].rolling(window=30).max()
df_mod['std_acc_y'] = np.round(df_mod['acc_y'].rolling(window=30).std(), 2)

df_mod['avg_acc_z'] = np.round(df_mod['acc_z'].rolling(window=30).mean(), 2)
df_mod['min_acc_z'] = df_mod['acc_z'].rolling(window=30).min()
df_mod['max_acc_z'] = df_mod['acc_z'].rolling(window=30).max()
df_mod['std_acc_z'] = np.round(df_mod['acc_z'].rolling(window=30).std(), 2)

# Calculate average standard deviation for all axes combined
df_mod['avg_std_all_axes'] = np.round(df_mod[['std_acc_x', 'std_acc_y', 'std_acc_z']].mean(axis=1), 2)

# Calculate movement intensity, SMA, Energy, Entropy, and Movement Variation for intervals of 30 seconds
df_mod['movement_intensity'] = np.round(np.sqrt(df_mod['acc_x']**2 + df_mod['acc_y']**2 + df_mod['acc_z']**2).rolling(window=30).mean(), 2)
df_mod['sma'] = np.round((df_mod[['acc_x', 'acc_y', 'acc_z']].abs().sum(axis=1)).rolling(window=30).mean(), 2)
df_mod['energy'] = np.round((df_mod['acc_x']**2 + df_mod['acc_y']**2 + df_mod['acc_z']**2).rolling(window=30).sum(), 2)
df_mod['entropy'] = np.round(-((1 + df_mod[['acc_x', 'acc_y', 'acc_z']].sum(axis=1)**2) * np.log(1 + df_mod[['acc_x', 'acc_y', 'acc_z']].sum(axis=1)**2)).rolling(window=30).sum(), 2)
df_mod['movement_variation'] = np.round(df_mod[['acc_x', 'acc_y', 'acc_z']].diff().abs().sum(axis=1).rolling(window=30).mean(), 2)

df_mod.to_csv(output_file_path, sep=';', index=False)

print(f"File '{output_file_path}' has been created with the selected columns and calculated values rounded to two decimal points.")
