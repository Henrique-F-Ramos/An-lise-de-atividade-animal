import pandas as pd
import matplotlib.pyplot as plt

file_path = "6027_mod.csv"

df = pd.read_csv(file_path, delimiter=';')

start_date = '2023-11-12 00:00:00'  
end_date = '2023-11-13 23:59:58'    

start_time = pd.to_datetime(start_date)
end_time = pd.to_datetime(end_date)

df['timestamp'] = pd.to_datetime(df['timestamp'])

filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

class_start_time = filtered_df.loc[filtered_df['Class'].notna(), 'timestamp'].min()
class_end_time = filtered_df.loc[filtered_df['Class'].notna(), 'timestamp'].max()

class_start_time = max(start_time, class_start_time)
class_end_time = min(end_time, class_end_time)

plt.plot(filtered_df['timestamp'], filtered_df['pitch'], label='pitch', marker='o', linestyle='-')
plt.plot(filtered_df['timestamp'], filtered_df['roll'], label='roll', marker='o', linestyle='-')
plt.plot(filtered_df['timestamp'], filtered_df['inclination'], label='inclination', marker='o', linestyle='-')

plt.axvspan(class_start_time, class_end_time, color='gray', alpha=0.3, label='Class Data Period')

plt.legend(loc='upper left')

plt.title('Sensor Data Over Time', fontsize=16, y=1.02) 
plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.grid(True)
plt.show()
