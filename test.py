import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the new dataset (for testing)
new_df = pd.read_csv('unlabeled_training_data.csv')

# Clean the new data: Keep only the relevant columns for prediction
new_df_cleaned = new_df[['Time', 'Protocol', 'Length', 'Source', 'Destination']]

# Load the saved label encoders
le_protocol = joblib.load('le_protocol.pkl')
le_source = joblib.load('le_source.pkl')
le_destination = joblib.load('le_destination.pkl')

# Function to safely encode labels (handles unseen labels)
def safe_transform(encoder, column):
    transformed_column = []
    for label in column:
        try:
            # Try to encode the label if it exists in the encoder's classes
            transformed_column.append(encoder.transform([label])[0])
        except ValueError:
            # If label is unseen, append a default value (e.g., -1 or a placeholder)
            transformed_column.append(-1)  # You could use a specific value if needed
    return transformed_column

# Apply label encoding to the relevant columns with safe transformation
new_df_cleaned['Protocol'] = safe_transform(le_protocol, new_df_cleaned['Protocol'])
new_df_cleaned['Source'] = safe_transform(le_source, new_df_cleaned['Source'])
new_df_cleaned['Destination'] = safe_transform(le_destination, new_df_cleaned['Destination'])

# Load the trained Random Forest model
clf = joblib.load('traffic_classifier_model.pkl')

# Prepare the features for prediction
X_new = new_df_cleaned[['Time', 'Protocol', 'Length', 'Source', 'Destination']]

# Make predictions on the new data
y_pred_new = clf.predict(X_new)

# If the true labels are available in the new data (i.e., 'bad_packet' column exists in the new dataset)
if 'bad_packet' in new_df.columns:
    y_true = new_df['bad_packet']
    # Evaluate the model's performance on the new data
    accuracy = accuracy_score(y_true, y_pred_new)
    report = classification_report(y_true, y_pred_new)
    cm = confusion_matrix(y_true, y_pred_new)
    
    # Output evaluation results
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)

    # Confusion Matrix visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Malicious'])
    disp.plot(cmap='Blues')

else:
    # If no 'bad_packet' column is available in the new data, just output the predictions
    new_df['Predicted_bad_packet'] = y_pred_new
    print(new_df[['Time', 'Protocol', 'Length', 'Source', 'Destination', 'Predicted_bad_packet']])

