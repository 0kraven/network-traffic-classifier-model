import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from collections import defaultdict  # Import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('training_data.csv')

# Check column names to ensure 'Info' exists
# print("Columns in dataset:", df.columns)

# Filter out only ARP packets
df_arp = df[df['Protocol'] == 'ARP']

# Initialize a new column 'bad_packet' with 0 (normal)
df_arp.loc[:, 'bad_packet'] = 0

# Extract features (Timestamp, Packet Length, Number of MAC addresses per IP)
ip_mac_map = defaultdict(set)
feature_list = []

for index, row in df_arp.iterrows():
    ip_address = row['Info'].split(' ')[0]  # '192.168.1.1'
    mac_address = row['Source']  # MAC address of the source device
    ip_mac_map[ip_address].add(mac_address)
    
    # Feature extraction
    num_mac_addresses = len(ip_mac_map[ip_address])
    feature_list.append([row['Time'], row['Length'], num_mac_addresses])

# Convert features to a DataFrame
features_df = pd.DataFrame(feature_list, columns=['Time', 'Length', 'Num_MAC_Addresses'])
df_arp = pd.concat([df_arp, features_df], axis=1)

# Drop rows with NaN values in 'bad_packet' (target variable)
df_arp = df_arp.dropna(subset=['bad_packet'])

# Prepare the feature matrix X and target variable y
X_train = df_arp[['Time', 'Length', 'Num_MAC_Addresses']]  # Features
y_train = df_arp['bad_packet']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'arp_spoofing_model.pkl')
print("Model trained and saved as 'arp_spoofing_model.pkl'.")

# Evaluate the model
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print('Classification Report:')
print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import joblib

# Load the model and dataset
clf = joblib.load('arp_spoofing_model.pkl')
df = pd.read_csv('dataset.csv')

# Filter out only ARP packets
df_arp = df[df['Protocol'] == 'ARP']

# Extract features and prepare the data as in your original code
ip_mac_map = defaultdict(set)
feature_list = []
for index, row in df_arp.iterrows():
    ip_address = row['Info'].split(' ')[0]  # '192.168.1.1'
    mac_address = row['Source']  # MAC address of the source device
    ip_mac_map[ip_address].add(mac_address)
    # Feature extraction
    num_mac_addresses = len(ip_mac_map[ip_address])
    feature_list.append([row['Time'], row['Length'], num_mac_addresses])

# Convert features to a DataFrame
features_df = pd.DataFrame(feature_list, columns=['Time', 'Length', 'Num_MAC_Addresses'])
df_arp = pd.concat([df_arp, features_df], axis=1)

# Prepare the feature matrix X and target variable y
X = df_arp[['Time', 'Length', 'Num_MAC_Addresses']]  # Features
y = df_arp['bad_packet']  # Target (assuming 'bad_packet' column exists as in the original code)

# Model prediction
y_pred = clf.predict(X)

# 1. **Distribution of 'Time' (ARP packet time distribution)**
plt.figure(figsize=(10, 6))
sns.histplot(df_arp['Time'], kde=True, bins=30, color='blue')
plt.title('Distribution of Time (ARP Packets)')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# 2. **Distribution of 'Length' (Packet length distribution)**
plt.figure(figsize=(10, 6))
sns.histplot(df_arp['Length'], kde=True, bins=30, color='green')
plt.title('Distribution of Packet Length (ARP)')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

# 3. **Distribution of 'Num_MAC_Addresses' (Number of MAC addresses per IP)**
plt.figure(figsize=(10, 6))
sns.histplot(df_arp['Num_MAC_Addresses'], kde=True, bins=30, color='red')
plt.title('Distribution of Number of MAC Addresses per IP')
plt.xlabel('Number of MAC Addresses')
plt.ylabel('Frequency')
plt.show()

# **Model Evaluation Graphs**

# 4. **Confusion Matrix**
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Spoofing'], yticklabels=['Normal', 'Spoofing'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

