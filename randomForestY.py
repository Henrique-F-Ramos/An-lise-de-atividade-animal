import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
import emlearn

file_name = 'combined_with_Y.csv'
df = pd.read_csv(file_name, delimiter=';')

# Convert 'Y' column to strings
df['Y'] = df['Y'].astype(str)

# Order the unique values in 'Y'
ordered_y_values = ['X'] + sorted([val for val in df['Y'].unique() if val != 'X'], key=lambda x: int(x))

# Count values in 'Y' column and order them
y_counts_before = df['Y'].value_counts(dropna=False).loc[ordered_y_values]

print("\nValue counts in 'Y'")
print(y_counts_before)

# Randomly sample 1500 rows where 'Y' is "X" and without "NaN" values
x_rows_to_keep = df[df['Y'] == 'X'].dropna().sample(n=20000, random_state=42)
df_filtered = pd.concat([df[df['Y'] != 'X'].dropna(), x_rows_to_keep])

features = ['pitch', 'roll', 'inclination', 'temperature',
            'avg_acc_x', 'min_acc_x', 'max_acc_x', 'std_acc_x',
            'avg_acc_y', 'min_acc_y', 'max_acc_y', 'std_acc_y',
            'avg_acc_z', 'min_acc_z', 'max_acc_z', 'std_acc_z',
            'avg_std_all_axes', 'movement_intensity', 'sma',
            'energy', 'entropy', 'movement_variation']

df_filtered['Y'] = df_filtered['Y'].astype(str)

X = df_filtered[features]
y = df_filtered['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importances
feature_importances = rf_model.feature_importances_

# Print feature importances
print("\nFeature Importances:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Convert RandomForestClassifier to C code using emlearn
cmodel = emlearn.convert(rf_model, method='inline')

# Save the generated C code to a file
cmodel.save(file='rf_model.h', name='rf_model')

# Make predictions
predictions = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)
print(f"\nAccuracy: {accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")

unique_y_values = ['X'] + list(range(-10, 31))

# Classification report
class_report = classification_report(y_test, predictions, labels=unique_y_values, zero_division=1) 
print("\nClassification Report:")
print(class_report)

# Confusion matrix
sns.set(font_scale=1.2)
plt.figure(figsize=(14, 12))
sns.heatmap(confusion_matrix(y_test, predictions, labels=unique_y_values), annot=True, fmt='d', cmap='Blues', xticklabels=unique_y_values, yticklabels=unique_y_values, annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()