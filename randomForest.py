import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix  
import seaborn as sns
import matplotlib.pyplot as plt

file_name = 'combined.csv'
df = pd.read_csv(file_name, delimiter=';')

class_counts_before = df['Class'].value_counts(dropna=False)

print("\nValue counts in 'Class'")
print(class_counts_before)

features = ['pitch', 'roll', 'inclination', 'temperature']
X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy}")


# confusion matrix
sns.set(font_scale=1.2)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions, labels=['B', 'C', 'L', 'E', 'X']), annot=True, fmt='d', cmap='Blues', xticklabels=['B', 'C', 'L', 'E', 'X'], yticklabels=['B', 'C', 'L', 'E', 'X'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()