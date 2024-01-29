import pandas as pd
from datetime import datetime, timedelta

# Assuming your CSV file is named '9541_mod.csv'b
file_path = '5727_mod.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, delimiter=';')

# Convert the 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

# Identify the first 'E' in each section of consecutive 'E' in the 'Class' column
# Also, find the timestamps of rows that are 90 seconds before and after the first 'E', and additional timestamps at 180-second intervals before the closest timestamp before

first_last_closest_timestamps = []

start_index = None
processed_first_in_section = False

for index, row in df.iterrows():
    if row['Class'] == 'E' and not processed_first_in_section:
        if start_index is None:
            start_index = index
        end_index = index

        # Find the timestamp 90 seconds before the first 'E'
        first_e_timestamp = df.at[start_index, 'timestamp']
        before_150s = first_e_timestamp - timedelta(seconds=150)
        before_150s_index = df['timestamp'].sub(before_150s).abs().idxmin()

        # Find the timestamp 150 seconds before the first 'E' (adjusted to intervals prior to before_150s)
        additional_timestamps = []
        for i in range(1, 49):
            additional_time = before_150s - timedelta(seconds=i * 3600)
            additional_index = df['timestamp'].sub(additional_time).abs().idxmin()
            additional_timestamps.append(df.at[additional_index, 'timestamp'])

        # Find the timestamp 150 seconds after the first 'E'
        after_150s = first_e_timestamp + timedelta(seconds=150)
        after_150s_index = df['timestamp'].sub(after_150s).abs().idxmin()

        additional_timestamps2 = []
        for i in range(1, 11):
            additional_time2 = after_150s + timedelta(seconds=i * 3600)
            additional_index2 = df['timestamp'].sub(additional_time2).abs().idxmin()
            additional_timestamps2.append(df.at[additional_index2, 'timestamp'])


        first_last_closest_timestamps.append((
            df.at[start_index, 'timestamp'],
            df.at[end_index, 'timestamp'],
            df.at[before_150s_index, 'timestamp'],
            additional_timestamps,  # Add additional timestamps before the closest timestamp before
            additional_timestamps2,
            df.at[after_150s_index, 'timestamp'],
        ))

        start_index = None
        processed_first_in_section = True

    elif row['Class'] != 'E':
        processed_first_in_section = False

# Create a new column "Y" with value "0" in between rows of 'before_150s' and 'after_150s', and empty otherwise
df['Y'] = ''

for first_timestamp, last_timestamp, before_150s, additional_timestamps, additional_timestamps2, after_150s in first_last_closest_timestamps:
    # Set Y=0 between rows of 'before_150s' and 'after_150s'
    df.loc[(df['timestamp'] >= before_150s) & (df['timestamp'] < after_150s) & (df['Y'] == ''), 'Y'] = '0'

    df.loc[(df['timestamp'] >= additional_timestamps[0]) & (df['timestamp'] < before_150s) & (df['Y'] == ''), 'Y'] = '1'

    for i in range(1,48):
        df.loc[(df['timestamp'] >= additional_timestamps[i]) & (df['timestamp'] < additional_timestamps[i-1]) & (df['Y'] == ''), 'Y'] = i+1

    df.loc[(df['timestamp'] >= after_150s) & (df['timestamp'] < additional_timestamps2[0]) & (df['Y'] == ''), 'Y'] = '-1'

    for i in range(1,10):
        df.loc[(df['timestamp'] >= additional_timestamps2[i-1]) & (df['timestamp'] < additional_timestamps2[i]) & (df['Y'] == ''), 'Y'] = (-i)-1
    

# Replace all remaining empty values in the "Y" column with "X"
df['Y'].replace('', 'X', inplace=True)

# Display the timestamps of the first and last 'E' in each section, as well as the closest values before and after the first 'E'
for first_timestamp, last_timestamp, before_150s, additional_timestamps, additional_timestamps2, after_150s in first_last_closest_timestamps:
    print("Primeiro E:", first_timestamp)
    print("Primeiro Y=0:", before_150s)
    print("Ultimo Y=0:", after_150s)
    print("---------------------------------------")
    print("Ys adicionais (intervalo 60 minutos antes do Y=0):", additional_timestamps)
    print("---------------------------------------")
    print("Ys a seguir (intervalo 60 minutos depois do Y=0):", additional_timestamps2)
    print("****************************************")

# Save the modified DataFrame to a new CSV file
new_file_path = '5727_mod_with_Y.csv'
df.to_csv(new_file_path, index=False, sep=';')
print(f"DataFrame saved to {new_file_path}")
