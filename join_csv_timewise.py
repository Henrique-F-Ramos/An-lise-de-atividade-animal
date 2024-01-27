import pandas as pd

def filter_timestamp_range(df, start_timestamp, end_timestamp):
    return df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]

file_names = ['3986_mod.csv', '5727_mod.csv', '9541_mod.csv']

combined_data = pd.DataFrame()

timestamp_ranges = {'3986_mod.csv': ('2023-10-02 04:54:54', '2023-10-02 06:10:42'),
                    '5727_mod.csv': ('2023-10-24 15:47:30', '2023-10-24 16:04:36'),
                    '9541_mod.csv': ('2023-10-21 04:16:31', '2023-10-21 04:49:07')}

for file_name in file_names:
    start_timestamp, end_timestamp = timestamp_ranges[file_name]
    
    df = pd.read_csv(file_name, sep=';')
    filtered_df = filter_timestamp_range(df, start_timestamp, end_timestamp)
    
    combined_data = combined_data.append(filtered_df, ignore_index=True)

combined_data['Class'].fillna('X', inplace=True)

combined_data.to_csv('combined_timewise.csv', sep=';', index=False)
