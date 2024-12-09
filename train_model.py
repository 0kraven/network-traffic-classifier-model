import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all warnings (if you really need to)
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('training_data.csv')

# Clean the dataset: Keep only relevant columns for classification
df_cleaned = df[['Time', 'Protocol', 'Length', 'Source', 'Destination', 'bad_packet']]

# Feature Engineering: Label encoding for categorical variables
print("Starting label encoding for categorical columns...")

# Label encoding for 'Protocol' (fit on the entire training data)
le_protocol = LabelEncoder()
le_protocol.fit(df_cleaned['Protocol'])  # Fit on all possible labels in training data
df_cleaned['Protocol'] = le_protocol.transform(df_cleaned['Protocol'])
print("Label encoding for 'Protocol' completed.")

# Label encoding for 'Source' (fit on the entire training data)
le_source = LabelEncoder()
le_source.fit(df_cleaned['Source'])  # Fit on all possible labels in training data
df_cleaned['Source'] = le_source.transform(df_cleaned['Source'])
print("Label encoding for 'Source' completed.")

# Label encoding for 'Destination' (fit on the entire training data)
le_destination = LabelEncoder()
le_destination.fit(df_cleaned['Destination'])  # Fit on all possible labels in training data
df_cleaned['Destination'] = le_destination.transform(df_cleaned['Destination'])
print("Label encoding for 'Destination' completed.")

# Feature matrix (X) and target vector (y)
X = df_cleaned[['Time', 'Protocol', 'Length', 'Source', 'Destination']]  # Features
y = df_cleaned['bad_packet']  # Target (malicious = 1, normal = 0)

# Split the data into training and testing sets (70% train, 30% test)
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data splitting completed.")

# Train the Random Forest Classifier
print("Training the Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model training completed.")

# Predictions on test data
y_pred = clf.predict(X_test)

# Evaluate the model
print("Evaluating the model...")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Malicious'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Plot
importances = clf.feature_importances_
features = ['Time', 'Protocol', 'Length', 'Source', 'Destination']
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar', color='lightblue')
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the distribution of 'bad_packet' values
plt.figure(figsize=(6, 4))
sns.countplot(x='bad_packet', data=df_cleaned, palette='viridis', hue='bad_packet', legend=False)
plt.title('Distribution of Classes (bad_packet)')
plt.xlabel('bad_packet (Normal = 0, Malicious = 1)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Save the trained model
print("Saving the trained Random Forest model...")
joblib.dump(clf, 'traffic_classifier_model.pkl')
print("Model saved as 'traffic_classifier_model.pkl'.")

# Save the Label Encoders
print("Saving label encoders for future use...")
joblib.dump(le_protocol, 'le_protocol.pkl')
joblib.dump(le_source, 'le_source.pkl')
joblib.dump(le_destination, 'le_destination.pkl')
print("Label encoders saved.")

# Optionally, save the cleaned and labeled dataset for future use
print("Saving the cleaned and labeled dataset to 'labeled_traffic_data.csv'...")
df_cleaned.to_csv('labeled_traffic_data.csv', index=False)
print("Cleaned data saved as 'labeled_traffic_data.csv'.")


