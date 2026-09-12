"""
Raw ingestion for the MLIT Real Estate Information Library 不動産情報ライブラリ
"""

from dotenv import load_dotenv
import os
import requests
import time
import gzip
import json
from datetime import datetime

# Constants
load_dotenv()
BASE_URL = os.environ["MLIT_URL"]
API_KEY = os.environ["MLIT_API_KEY"]
API_KEY_HEADER = "Ocp-Apim-Subscription-Key"

# Earliest published quarter is 2005 Q3
EARLIEST = (2005, 3)
now = datetime.now()
END = (now.year, (now.month - 1) // 3 + 1)

# Codes for Prefectures are strings from 01 to 47
PREFECTURES = tuple(f"{i:02d}" for i in range(1, 48))

OUTPUT_FOLDER = "data/raw"

def make_filename(pref, year, quarter):
    """Build the path where data for one group of records gets saved."""
    folder = OUTPUT_FOLDER + "/pref=" + pref + "/year=" + str(year)
    return folder + "/quarter=" + str(quarter) + ".json.gz"  

def request_one(pref, year, quarter):
    """Requests one group of records from the MLIT API."""
    headers = {API_KEY_HEADER: API_KEY}
    params = {"area": pref, "year": str(year), "quarter": str(quarter)}

    response = requests.get(BASE_URL + "/XIT001", headers=headers, params=params, timeout=60)
    if response.status_code == 404:
        return []
    if response.status_code != 200:
        raise Exception(f"API returned {response.status_code} for {params}")
    
    body = response.json()
    return body["data"]
    
def save_records(records, filename):
    """Saves the records to specified file name"""
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

    with gzip.open(filename, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log(pref, year, quarter, count):
    """Logs the downloads in download_log.csv"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    line = f"{pref},{year},{quarter},{count},{time.strftime('%Y-%m-%d %H:%M:%S')}"

    with open(f"{OUTPUT_FOLDER}/download_log.csv", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    start_year, start_quarter = EARLIEST
    end_year, end_quarter = END
    for pref in PREFECTURES:
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                
                if (year, quarter) < EARLIEST:
                    continue
                if (year, quarter) > END:
                    continue

                # Build filename for records
                filename = make_filename(pref, year, quarter)
                # Skips filenames that already exist to save API calls
                if os.path.exists(filename):
                    continue
                
                records = request_one(pref, year, quarter)

                if records:
                    save_records(records, filename)
                    print(f"pref={pref}, year={year}, quarter={quarter} successfully downloaded.")
                
                log(pref, year, quarter, len(records)) 

                time.sleep(1)   

if __name__ == "__main__":
    main()
    




