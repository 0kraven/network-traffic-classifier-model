import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('labeled_training_data.csv')

# Step 1: Understand the Dataset
print("Dataset Shape:", df.shape)
print("Columns in Dataset:", df.columns)
print("Data Types in Dataset:", df.dtypes)
print("First few rows of the dataset:")
print(df.head())
print("\nSummary Statistics (Numerical Features):")
print(df.describe())
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

if 'bad_packet' in df.columns:
    print("\nClass Distribution of 'bad_packet':")
    print(df['bad_packet'].value_counts())

print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Step 2: Exploratory Data Analysis (EDA)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Length'], kde=True, color='blue', bins=30)
plt.title('Distribution of Packet Length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Time'], kde=True, color='green', bins=30)
plt.title('Distribution of Time')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='bad_packet', data=df, palette='Set2')
plt.title('Class Distribution of bad_packet')
plt.xlabel('bad_packet')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Length'], color='orange')
plt.title('Boxplot for Packet Length')
plt.xlabel('Length')
plt.show()

# Step 3: Feature Engineering
le_protocol = LabelEncoder()
df['Protocol'] = le_protocol.fit_transform(df['Protocol'])

if 'Destination' in df.columns:
    df = pd.get_dummies(df, columns=['Source', 'Destination'], drop_first=True)
else:
    print("'Destination' column does not exist. Please check the dataset.")

df['packet_interval'] = df['Time'].diff().fillna(0)
df['is_arp_request'] = df['Info'].apply(lambda x: 1 if 'Who has' in str(x) else 0)

if 'Destination_Broadcast' in df.columns:
    df['is_broadcast'] = df['Destination_Broadcast']
else:
    if 'Destination' in df.columns:
        df['is_broadcast'] = df['Destination'].apply(lambda x: 1 if x == 'Broadcast' else 0)
    else:
        print("'Destination' column not found, unable to create 'is_broadcast'.")

numerical_features = ['Time', 'Length', 'packet_interval']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 4: Prepare the Data for Modeling
X = df.drop(columns=['bad_packet', 'Info', 'No.'])
y = df['bad_packet']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Step 5: Initialize and Train Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(lr_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Print best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# Step 6: Model Evaluation
y_pred = grid_search.best_estimator_.predict(X_test)

# Perform cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean()} ± {cv_scores.std()}")

# Evaluate accuracy and print classification report
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Malicious'], yticklabels=['Normal', 'Malicious'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Save the model
joblib.dump(grid_search.best_estimator_, 'logistic_regression_model.pkl')

print("Model saved as 'logistic_regression_model.pkl'")
