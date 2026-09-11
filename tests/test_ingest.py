from jp_housing_pred import ingest
import gzip
import json

def test_make_filename():
    pref = "13"
    year = 2005
    quarter = 3

    result = ingest.make_filename(pref, year, quarter)
    assert result == "data/raw/pref=13/year=2005/quarter=3.json.gz" 

def test_save_records(tmp_path):
    records = [
        {
        "PriceCategory": "不動産取引価格情報",
        "Type": "宅地(土地と建物)",
        "Region": "商業地",
        "MunicipalityCode": "13101",
        "Prefecture": "東京都",
        "Municipality": "千代田区",
        "DistrictName": "岩本町",
        "TradePrice": "64000000",
        "PricePerUnit": "",
        "FloorPlan": "",
        "Area": "50",
        "UnitPrice": "",
        "LandShape": "長方形",
        "Frontage": "4.7",
        "TotalFloorArea": "95",
        "BuildingYear": "1969年",
        "Structure": "木造",
        "Use": "住宅、店舗",
        "Purpose": "住宅",
        "Direction": "北西",
        "Classification": "区道",
        "Breadth": "3",
        "CityPlanning": "商業地域",
        "CoverageRatio": "80",
        "FloorAreaRatio": "600",
        "Period": "2024年第1四半期",
        "Renovation": "",
        "Remarks": "",
        "DistrictCode": "131010030"
        },
        {
        "PriceCategory": "不動産取引価格情報",
        "Type": "中古マンション等",
        "Region": "",
        "MunicipalityCode": "13101",
        "Prefecture": "東京都",
        "Municipality": "千代田区",
        "DistrictName": "岩本町",
        "TradePrice": "20000000",
        "PricePerUnit": "",
        "FloorPlan": "１Ｋ",
        "Area": "20",
        "UnitPrice": "",
        "LandShape": "",
        "Frontage": "",
        "TotalFloorArea": "",
        "BuildingYear": "2004年",
        "Structure": "ＳＲＣ",
        "Use": "住宅",
        "Purpose": "住宅",
        "Direction": "",
        "Classification": "",
        "Breadth": "",
        "CityPlanning": "商業地域",
        "CoverageRatio": "80",
        "FloorAreaRatio": "700",
        "Period": "2016年第1四半期",
        "Renovation": "未改装",
        "Remarks": "",
        "DistrictCode": "131010030"
        },
    ]

    loaded = []
    filename = tmp_path / "pref=13" / "year=2005" / "quarter=3.json.gz"

    ingest.save_records(records, filename)

    with gzip.open(filename, "rt", encoding="utf-8") as f:
        for record in f:
            loaded.append(json.loads(record))
    
    assert loaded == records