import pandas as pd
import matplotlib.pyplot as plt

file_path = "5727_mod.csv"

df = pd.read_csv(file_path, delimiter=';')

start_time = df.loc[df['Class'].notna(), 'timestamp'].min()
end_time = df.loc[df['Class'].notna(), 'timestamp'].max()

print(start_time)
print(end_time)

num_samples_before = 500
num_samples_after = 500

start_index = df[df['timestamp'] == start_time].index[0]
end_index = df[df['timestamp'] == end_time].index[0]

start_range = max(0, start_index - num_samples_before)
end_range = min(len(df) - 1, end_index + num_samples_after)

filtered_df = df.iloc[start_range:end_range + 1]

filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

plt.plot(filtered_df['timestamp'], filtered_df['pitch'], label='pitch', marker='o', linestyle='-')
plt.plot(filtered_df['timestamp'], filtered_df['roll'], label='roll', marker='o', linestyle='-')
#plt.plot(filtered_df['timestamp'], filtered_df['inclination'], label='inclination', marker='o', linestyle='-')

plt.axvspan(start_time, end_time, color='gray', alpha=0.3, label='Class Data Period')

plt.legend(loc='upper left')

plt.title('Sensor Data Over Time', fontsize=16, y=1.02) 
plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.grid(True)
plt.show()

