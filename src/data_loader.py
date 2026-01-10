import yfinance as yf
import pandas as pd
import os
import kagglehub
import glob

# Project paths using os module (ROBUST)

# Directory of this script (src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root (parent of src)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Data folder inside project root called "data"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# 1. Download S&P 500 tickers from Kaggle and add them to the data folder

print("Downloading S&P 500 ticker list from Kaggle...")


# Code given in the website, simply downloads the dataset 
kaggle_path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
print("Kaggle dataset downloaded to:", kaggle_path)


# Robustly find the companies CSV, since the downloaded file may not be in the downloads folder of the OS (handles potential filename variations)
potential_files = glob.glob(os.path.join(kaggle_path, "*companies*.csv"))
if potential_files:
    kaggle_companies_csv = potential_files[0]
else:
    # Fallback: try default name, but print contents if missing to help debug
    print(f"Warning: *companies*.csv not found in {kaggle_path}")
    if os.path.exists(kaggle_path):
        print(f"Directory contents: {os.listdir(kaggle_path)}")
    
    # Attempt recursive search in case file is in a subdirectory
    potential_files_recursive = glob.glob(os.path.join(kaggle_path, "**", "*companies*.csv"), recursive=True)
    if potential_files_recursive:
        kaggle_companies_csv = potential_files_recursive[0]
    else:
        # If still not found, the cache is likely corrupted (empty folder)
        raise FileNotFoundError(f"No companies CSV found in {kaggle_path}. The folder appears empty. Please delete the folder '{kaggle_path}' manually to force a fresh download.")

# Destination path in our project data directory
companies_csv_filename = "sp500_companies.csv"
companies_csv_path = os.path.join(DATA_DIR, companies_csv_filename)

# Load from Kaggle cache
companies_df = pd.read_csv(kaggle_companies_csv)

# Save to our data directory
companies_df.to_csv(companies_csv_path, index=False)

print(f"Saved ticker list to {companies_csv_path}")
print(f"Loaded {len(companies_df)} tickers")


# 2. Load and clean tickers

sp500_tickers = companies_df["Symbol"].tolist()

# Yahoo Finance formatting: replace "." in the ticker for "-"
formatted_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
unique_tickers = sorted(set(formatted_tickers))

# Drop specific duplicate tickers (keeping the preferred class)
tickers_to_drop = ["GOOG", "FOX", "NWS"]
unique_tickers = [t for t in unique_tickers if t not in tickers_to_drop]

print(f"\nReady to download historical data for {len(unique_tickers)} tickers...")


# 3. Download historical price data using Yahoo Finance's API

# API requests are split into different batches and sleep function from time module is used to avoid the API from breaking
from time import sleep

all_data = []

batch_size = 50 # Loop for each batch that contains 50 tickers approximately
ticker_batches = [
    unique_tickers[i:i + batch_size]
    for i in range(0, len(unique_tickers), batch_size)
]

print(f"Downloading data in {len(ticker_batches)} batches...")

for i, batch in enumerate(ticker_batches, 1):
    print(f"Batch {i}/{len(ticker_batches)}: {len(batch)} tickers")
    
    batch_df = yf.download(
        batch,
        start="2015-01-01",
        end="2025-01-01",
        auto_adjust=True, # Only adjusted closed prices are relevant for the analysis
        threads=False,   
        progress=False
    )
    
    if batch_df.empty: # Handles most common error when no data is downloaded for the batch
        print(f"Batch {i} returned empty data")
        continue
    
    all_data.append(batch_df)

    sleep(2)  

if not all_data: 
    raise ValueError("No data was downloaded. Check your internet connection or ticker list.")

print("Concatenating batches...")
df = pd.concat(all_data, axis=1)

# Save raw wide-format data, which is standard for Yahoo Finance
raw_output_filename = "sp500_2015_2025_raw.csv"
raw_output_path = os.path.join(DATA_DIR, raw_output_filename)
df.to_csv(raw_output_path)
print(f"\nRaw price data saved to {raw_output_path}")


# 4. Convert to long format, which is the best way to input in ML models

print("\nReshaping data to long format...")

df_wide = pd.read_csv(
    raw_output_path,
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

# Stack tickers into index
# For each day there will be a row with every ticker, the same for the next day and so on
df_long = df_wide.stack(level=1, future_stack=True) 
df_long.index.names = ["date", "ticker"]

# Save long-format data
long_output_filename = "sp500_long_data.csv"
long_output_path = os.path.join(DATA_DIR, long_output_filename)
df_long.to_csv(long_output_path)

print(f"\nLong-format data successfully saved to {long_output_path}")

# Preview to make sure the format is right
print("Long format preview:")
print(df_long.head())