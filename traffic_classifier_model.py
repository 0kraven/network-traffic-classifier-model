import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('training_data.csv')

# Clean the dataset: Keep only relevant columns for classification
df_cleaned = df[['Time', 'Protocol', 'Length', 'Source', 'Destination', 'bad_packet']]

# Feature Engineering: Label encoding for categorical variables
le_protocol = LabelEncoder()
df_cleaned['Protocol'] = le_protocol.fit_transform(df_cleaned['Protocol'])

le_source = LabelEncoder()
df_cleaned['Source'] = le_source.fit_transform(df_cleaned['Source'])

le_destination = LabelEncoder()
df_cleaned['Destination'] = le_destination.fit_transform(df_cleaned['Destination'])

# The target column is already available as 'bad_packet'
X = df_cleaned[['Time', 'Protocol', 'Length', 'Source', 'Destination']]  # Features
y = df_cleaned['bad_packet']  # Target (malicious = 1, normal = 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions on test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Output evaluation results
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)

# Confusion Matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Malicious'])
disp.plot(cmap='Blues')

# Save the trained model
joblib.dump(clf, 'traffic_classifier_model.pkl')

# Save the cleaned and labeled data (optional)
df_cleaned.to_csv('labeled_traffic_data.csv', index=False)

import joblib

# After fitting the LabelEncoders (le_protocol, le_source, le_destination)
joblib.dump(le_protocol, 'le_protocol.pkl')
joblib.dump(le_source, 'le_source.pkl')
joblib.dump(le_destination, 'le_destination.pkl')

# After training the Random Forest model (clf)
joblib.dump(clf, 'traffic_classifier_model.pkl')

# Optionally, save the cleaned dataset (df_cleaned) to a CSV for future use
df_cleaned.to_csv('labeled_traffic_data.csv', index=False)

