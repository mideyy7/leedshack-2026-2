import pandas as pd


df = pd.read_csv("data/delivery.csv")

"""
import re

regexp = re.compile(r'\((.*)\)$')
locs_set = set()

for index, row in df.iterrows():
    for s in (row["source_name"], row["destination_name"]):
        if pd.isna(s):
            #print(f"{row = }: {s} is NaN")
            continue

        match = regexp.search(s)

        if match:
            extracted = match.group(1)
            locs_set.add(extracted)
        else:
            raise Exception(f"{index = } {row = } {s = }")


print(locs_set)
"""

locs_set = {
    'Jammu & Kashmir', 'Andhra Pradesh', 'Gujarat', 'Telangana', 'Kerala',
    'West Bengal', 'Jharkhand', 'Chandigarh', 'Uttarakhand', 'Assam', 'Haryana',
    'Orissa', 'Dadra and Nagar Haveli', 'Arunachal Pradesh', 'Delhi', 'Bihar', 'Tamil Nadu',
    'Meghalaya', 'Nagaland', 'Mizoram', 'Uttar Pradesh', 'Maharashtra', 'Rajasthan', 'Punjab',
    'Chhattisgarh', 'Daman & Diu', 'Himachal Pradesh', 'Karnataka', 'Goa', 'Pondicherry', 'Tripura',
    'Madhya Pradesh',
}

"""
df["od_start_time"] = pd.to_datetime(df["od_start_time"])
df["od_end_time"] = pd.to_datetime(df["od_end_time"])

print(min(
    df["od_start_time"].min(), df["od_end_time"].min(),
), max(
    df["od_start_time"].max(), df["od_end_time"].max(),
))

from datetime import datetime

earliest = datetime(2018, 9, 12)
latest = datetime(2018, 10, 8)
"""


import requests

from datetime import datetime, date, time
import csv

fp = open("weather2.csv", "w")
csv_wr = csv.writer(fp)

csv_wr.writerow(
    ("loc", "ts", "tempC", "humidity", "pressure", "windGustKmph", "precipMM", "weatherDescValue"),
)

API_URL = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"
API_KEY = "a54670f460404febb6a44013260802"

params = {
    "format": "json",
    "key": API_KEY,
}

def get_rows(loc, resp):
    data = resp.json()["data"]["weather"]

    for day in data:
        date_part = date.strptime(day["date"], "%Y-%m-%d")

        for hr in day["hourly"]:
            hrs = int(hr["time"]) // 100
            dt = datetime.combine(date_part, time(hour=hrs))

            yield (
                loc, dt,
                hr["tempC"], hr["humidity"], hr["pressure"], hr["WindGustKmph"],
                hr["precipMM"], hr["weatherDesc"][0]["value"],
            )


for loc in locs_set:
    params.update(
        q='{},India'.format(loc),
    )

    params.update(
        date="2018-09-12",
        enddate="2018-09-30",
    )

    resp = requests.get(API_URL, params=params)
    
    for row in get_rows(loc, resp):
        csv_wr.writerow(row)

    params.update(
        date="2018-10-01",
        enddate="2018-10-08",
    )

    resp = requests.get(API_URL, params=params)

    for row in get_rows(loc, resp):
        csv_wr.writerow(row)
