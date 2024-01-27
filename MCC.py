import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

file_name = 'combined_with_Y.csv'
df = pd.read_csv(file_name, delimiter=';')

features = ['pitch', 'roll', 'inclination', 'temperature',
            'avg_acc_x', 'min_acc_x', 'max_acc_x', 'std_acc_x',
            'avg_acc_y', 'min_acc_y', 'max_acc_y', 'std_acc_y',
            'avg_acc_z', 'min_acc_z', 'max_acc_z', 'std_acc_z',
            'avg_std_all_axes', 'movement_intensity', 'sma',
            'energy', 'entropy', 'movement_variation']

n_values = []
mcc_scores = []


for n in range(10000, 200000, 5000):  
    # Randomly sample n rows where 'Y' is "X" and without "NaN" values
    x_rows_to_keep = df[df['Y'] == 'X'].dropna().sample(n=n, random_state=42)
    df_filtered = pd.concat([df[df['Y'] != 'X'].dropna(), x_rows_to_keep])


    X = df_filtered[features]
    y = df_filtered['Y']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)

    # Calculate MCC
    mcc = matthews_corrcoef(y_test, predictions)

    # Store values
    n_values.append(n)
    mcc_scores.append(mcc)
    print(n)

# Plot the results
plt.plot(n_values, mcc_scores, marker='o')
plt.xlabel('Datapoints fora do parto')
plt.ylabel('MCC')
plt.title('MCC vs. Datapoints fora do parto')
plt.grid(True)
plt.show()
