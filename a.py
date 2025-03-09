import os
import datetime

file_path = "/home/ubuntu/tornet-Higgins/tornet_baseline250308211618-None-None/tornadoDetector_001.keras"  # Replace with your file path
timestamp = os.path.getmtime(file_path)  # Get last modified time
last_modified = datetime.datetime.fromtimestamp(timestamp)

print(f"Last modified: {last_modified}")

