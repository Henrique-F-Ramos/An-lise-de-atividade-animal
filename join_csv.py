import pandas as pd

file_names = ['3986_mod_with_Y.csv', '5727_mod_with_Y.csv', '9541_1_mod_with_Y.csv', '9541_2_mod_with_Y.csv']

combined_data = pd.DataFrame()

for file_name in file_names:
    df = pd.read_csv(file_name, sep=';')
    combined_data = combined_data.append(df, ignore_index=True)

combined_data['Class'].fillna('X', inplace=True)

combined_data.to_csv('combined_with_Y.csv', sep=';', index=False)
