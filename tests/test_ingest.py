from jp_housing_pred import ingest
import gzip
import json
from types import SimpleNamespace
import pytest

# ------------ make_filename() ------------
def test_make_filename():
    # Arrange
    pref = "13"
    year = 2005
    quarter = 3

    # Act
    result = ingest.make_filename(pref, year, quarter)

    # Assert
    expected = "data/raw/pref=13/year=2005/quarter=3.json.gz"
    assert result == expected, f"Filename path mismatch. Expected '{expected}', got '{result}'"

# ------------ request_one() ------------
def fake_response(status_code, body=None):
    return SimpleNamespace(status_code=status_code, json=lambda: body)

@pytest.mark.parametrize(
    "status_code, body, expected",
    [
        (200, {"status": "OK", "data": [{"TradePrice": "1"}]}, [{"TradePrice": "1"}]),
        (404, None, []),
    ]
)
def test_request_one_returns_data(monkeypatch, status_code, body, expected):
    # Arrange
    monkeypatch.setattr(ingest.requests, "get", lambda url, **kwargs: fake_response(status_code, body))

    # Act
    result = ingest.request_one("13", 2024, 1)

    # Assert
    assert result == expected

# ------------ save_records() ------------
def test_save_records(tmp_path):
    # Arrange
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

    # Act
    ingest.save_records(records, filename)

    # Assert
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        for record in f:
            loaded.append(json.loads(record))
    
    assert loaded == records, f"Records are different after saving. Expected '{records}', got '{loaded}'"

    # ------------ log() ------------

    def test_log_appends(tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr(ingest, "OUTPUT_FOLDER", str(tmp_path))

        # Act
        ingest.log("13", 2024, 1, 100)
        ingest.log("13", 2024, 2, 0)

        # Assert
        with open(tmp_path / "download_log.csv", encoding="utf-8") as f:
            lines = f.read().splitlines()
        
        assert len(lines) == 2
        assert lines[0].startswith("13,2024,1,100,")
        assert lines[1].startswith("13,2024,2,0,")
 