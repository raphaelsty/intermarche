from glob import glob

import pandas as pd
from tqdm import tqdm

train_test = pd.read_pickle("data/prepared/sales.pkl")

items = pd.read_pickle("./data/prepared/items.pkl").rename(columns={"id": "item"})

train_test = pd.merge(left=train_test.reset_index(), right=items, on="item", how="left").set_index(
    "index"
)

stores = pd.read_pickle("./data/prepared/stores.pkl").rename(columns={"id": "store"})

train_test = pd.merge(
    left=train_test.reset_index(), right=stores, on="store", how="left"
).set_index("index")

train_test["year"] = train_test.date.dt.year
train_test["quarter"] = train_test.date.dt.quarter
train_test["day_of_week"] = train_test.date.dt.day_of_week
train_test["month"] = train_test.date.dt.month

for path in tqdm(list(glob("data/features/*.pkl"))):

    agg = pd.read_pickle(path)

    if agg.index.names == [None]:
        train_test = pd.concat((train_test, agg), axis=1)
    else:
        train_test = train_test.join(agg, on=agg.index.names)

train_test = train_test.drop(
    [
        "shelf",
        "family",
        "sub_family",
        "vocation",
        "region",
        "size",
        "surface",
        "day_of_week",
        "month",
        "year",
        "quarter",
    ],
    axis="columns",
)

(
    train_test[train_test.date.dt.year.eq(2018)]
    .set_index(["date", "store", "item"])
    .to_pickle("data/train.pkl")
)

(
    train_test[train_test.date.dt.year.eq(2019)]
    .drop(columns=["quantity"])
    .set_index(["date", "store", "item"])
    .to_pickle("data/test.pkl")
)
