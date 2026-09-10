"""
Raw ingestion for the MLIT Real Estate Information Library 不動産情報ライブラリ

"""

from dotenv import load_dotenv
import os
import requests
# Constants

load_dotenv()
BASE_URL = os.environ["MLIT_URL"]
API_KEY = os.environ["MLIT_API_KEY"]
API_KEY_HEADER = "Ocp-Apim-Subscription-Key"

# Earliest published quarter is 2005 Q3
EARLIEST = (2005, 3)

# Codes for Prefectures are strings from 01 to 47
PREFECTURES = tuple(f"{i:02d}" for i in range(1, 48))

OUTPUT_FOLDER = "data/raw"

def make_filename(pref, year, quarter):
    """Build the path where one chunk of data gets saved."""
    folder = OUTPUT_FOLDER + "/pref=" + pref + "/year=" + str(year)
    return folder + "/quarter=" + str(quarter) + ".json.gz"  

def request_one(pref, year, quarter):
    headers = {API_KEY_HEADER: API_KEY}
    params = {"area": pref, "year": str(year), "quarter": str(quarter)}

    response = requests.get(BASE_URL + "/XIT001", headers=headers, params=params, timeout=60)
    if response.status_code == 404:
        return []
    if response.status_code != 200:
        raise Exception(f"API returned {response.status_code} for {params}")
    
    body = response.json()
    return body["data"]
    

data = request_one("01", 2010, 3)
print(data)

# response = httpx.get(
#     "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001",
#     params={"year": 2024, "quarter": 1, "area": "13"},
#     headers={"Ocp-Apim-Subscription-Key": os.environ["MLIT_API_KEY"]},
#     timeout=60.0,
# )

# print("status:", response.status_code)
# print("rows:", len(response.json()["data"]))
# print("first row:", response.json()["data"][0])