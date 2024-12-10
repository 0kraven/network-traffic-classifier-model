import pandas as pd
import warnings

warnings.filterwarnings("ignore")
# Load the dataset
df = pd.read_csv('training_data.csv')

# Check column names to ensure 'Info' exists
print("Columns in dataset:", df.columns)

# Filter out only ARP packets
df_arp = df[df['Protocol'] == 'ARP']

# Check if 'Info' column exists
if 'Info' in df_arp.columns:
    # Create a dictionary to store IP-MAC mappings
    ip_mac_map = {}

    # Initialize a new column 'bad_packet' with 0 (normal)
    df_arp = df_arp.copy()  # Avoid SettingWithCopyWarning

    df_arp['bad_packet'] = 0  # Default label as normal

    # Iterate over the ARP packets to detect ARP spoofing
    for index, row in df_arp.iterrows():
        # Extract source and destination IPs and MAC addresses from the 'Info' column
        ip_address = row['Info'].split(' ')[0]  # Assuming the first word in 'Info' is the IP address
        mac_address = row['Source']  # MAC address of the source device
        
        if ip_address not in ip_mac_map:
            ip_mac_map[ip_address] = set()
        
        # Add the MAC address associated with the IP address
        ip_mac_map[ip_address].add(mac_address)
        
        # If the same IP is associated with more than one MAC address, flag the packet as malicious
        if len(ip_mac_map[ip_address]) > 1:
            df_arp.at[index, 'bad_packet'] = 1  # Mark this packet as malicious

    # Show a few samples of the labeled data
    print(df_arp[['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info', 'bad_packet']].head())

    # Save the labeled data for further analysis or training
    df_arp.to_csv('labeled_training_data.csv', index=False)
else:
    print("Error: 'Info' column not found in the dataset. Please check the dataset structure.")
