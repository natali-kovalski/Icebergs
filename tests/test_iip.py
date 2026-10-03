from pathlib import Path

import pandas as pd
from shapely.geometry import box

from iceberg_sar.iip import load_sightings, select_sightings

# The 2018 season file has line breaks inside quoted header names; others have one-line headers.
CSV = '''ICEBERG YEAR,ICEBERG NUMBER,"SIGHTING
DATE","SIGHTING
TIME",LAT,LON,METHOD,SIZE,SHAPE,SOURCE
2019,1,4/15/2019,1531,50.5,-53.2,R/V,MED,GEN,GMRS
2019,2,04/29/2019,940,50.6,-53.4,SAT-HIGH,GEN,GEN,SN1A
2019,3,4/29/2019,939.0,52.9,-53.0,SAT-LOW,GEN,GEN,SNL1
'''


def test_load_and_select(tmp_path: Path) -> None:
    f = tmp_path / "IIP_2019IcebergSeason.csv"
    f.write_text(CSV, encoding="utf-8")
    s = load_sightings(f)
    assert list(s.time) == [pd.Timestamp("2019-04-15 15:31"), pd.Timestamp("2019-04-29 09:40"),
                            pd.Timestamp("2019-04-29 09:39")]
    area = box(-54, 50, -53, 51)
    day = (pd.Timestamp("2019-04-29 09:00"), pd.Timestamp("2019-04-29 10:00"))
    assert select_sightings(s, area, *day).number.tolist() == [2]   # 3 is outside the box
    assert select_sightings(s, area, *day, satellite=False).empty
    allday = select_sightings(s, area, pd.Timestamp("2019-04-15"), pd.Timestamp("2019-04-30"),
                              satellite=False)
    assert allday.number.tolist() == [1]
