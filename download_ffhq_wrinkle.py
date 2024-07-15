import os
from gdrivedataset import loader

file_ids = [
    "1irv2YTwbGoBSOCIaJC25AL6dcRYfcJWZ",
    "1QcexZsNp3xc_3HhCUgzN732RFMjFmD30",
]

folder = "./data" # edit your base path

if not os.path.exists(folder):
    os.makedirs(folder)

for file_id in file_ids:
    try:
        loader.load_from_google_drive(file_id, folder)
        print(f"Downloaded file with ID: {file_id} to {folder}")
    except Exception as e:
        print(f"Failed to download file with ID: {file_id}. Error: {e}")
