import pandas as pd
from tqdm import tqdm

sales = pd.read_pickle("data/prepared/sales.pkl")

sales["quantity"] = sales.quantity.astype(float)
sales["day_of_week"] = sales.date.dt.weekday

aggs = {
    ("store", "item"): [
        (90, "median"),
    ],
    ("store", "item", "day_of_week"): [
        (9, "median"),
    ],
}

for key, lag_funcs in tqdm(aggs.items()):

    for lag, func in lag_funcs:

        name = f"{func}_prev_{lag}_" + "_x_".join(key)
        path = f"data/features/{name}.pkl"
        agg = sales.groupby(list(key))["quantity"].apply(
            lambda x: (x.shift(1).rolling(window=lag, min_periods=1).agg(func).ffill())
        )
        agg = agg.rename(name)
        agg.to_pickle(path)
