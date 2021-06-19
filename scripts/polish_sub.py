import datetime

import pandas as pd

sub = pd.read_pickle("data/sub.pkl")
sales = pd.read_pickle("data/prepared/sales.pkl")

sales["day_of_the_week"] = sales.date.dt.weekday

agg = sales.groupby(["store", "item", "day_of_the_week"])["quantity"].sum().rename("drop") <= 2

date = sub.index.get_level_values("date").to_series()

ref = date.between(pd.Timestamp("2019-01-07"), pd.Timestamp("2019-01-20")).values

for n in range(2, 12):

    if n % 2 == 0:

        update = date.between(
            pd.Timestamp("2019-01-07") + datetime.timedelta(days=7 * n),
            pd.Timestamp("2019-01-20") + datetime.timedelta(days=7 * n),
        ).values

        sub.loc[update] = sub.loc[ref].values

ref = date.between(pd.Timestamp("2019-01-07"), pd.Timestamp("2019-01-13")).values

update = date.between(
    pd.Timestamp("2019-03-25"),
    pd.Timestamp("2019-03-31"),
).values

sub.loc[update] = sub.loc[ref].values

sub = sub[sub.gt(0.5)]

sub = sub.rename("qte").reset_index()
sub["day_of_the_week"] = sub["date"].dt.weekday
sub = sub.merge(agg, how="left", on=["store", "item", "day_of_the_week"])
sub = sub[sub["drop"] == False].copy()
sub = sub.set_index(["date", "store", "item"]).drop(["day_of_the_week", "drop"], axis="columns")

(
    pd.DataFrame(
        {
            "id": [f"{i[1]}_{i[2]}_{i[0].strftime('%Y%m%d')}" for i in sub.index],
            "qte": sub["qte"].values.round(),
        }
    ).to_csv("submission.csv.zip", index=False, compression="zip")
)
