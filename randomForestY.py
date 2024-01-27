import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt

file_name = 'combined_with_Y.csv'
df = pd.read_csv(file_name, delimiter=';')

# Count values in 'Y' column 
y_counts_before = df['Y'].value_counts(dropna=False)

print("\nValue counts in 'Y'")
print(y_counts_before)

# Randomly sample 'n' rows where 'Y' is "X" and without "NaN" values
x_rows_to_keep = df[df['Y'] == 'X'].dropna().sample(n=7000, random_state=42)
df_filtered = pd.concat([df[df['Y'] != 'X'].dropna(), x_rows_to_keep])

features = ['pitch', 'roll', 'inclination', 'temperature',
            'avg_acc_x', 'min_acc_x', 'max_acc_x', 'std_acc_x',
            'avg_acc_y', 'min_acc_y', 'max_acc_y', 'std_acc_y',
            'avg_acc_z', 'min_acc_z', 'max_acc_z', 'std_acc_z',
            'avg_std_all_axes', 'movement_intensity', 'sma',
            'energy', 'entropy', 'movement_variation']

X = df_filtered[features]
y = df_filtered['Y']

# Use StratifiedKFold for classification tasks to ensure each fold has the same distribution of target classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k-fold cross-validation
predictions = cross_val_predict(rf_model, X, y, cv=cv)

# Evaluate the model
accuracy = accuracy_score(y, predictions)
mcc = matthews_corrcoef(y, predictions)
print(f"\nAccuracy: {accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")

unique_y_values = ['X'] + list(range(-2, 11))

# Classification report
class_report = classification_report(y, predictions, labels=unique_y_values, zero_division=1)
print("\nClassification Report:")
print(class_report)

# Confusion matrix
sns.set(font_scale=1.2)
plt.figure(figsize=(14, 12))
sns.heatmap(confusion_matrix(y, predictions, labels=unique_y_values), annot=True, fmt='d', cmap='Blues', xticklabels=unique_y_values, yticklabels=unique_y_values, annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
