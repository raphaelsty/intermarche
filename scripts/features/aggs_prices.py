import pandas as pd
from tqdm import tqdm

sales = pd.read_pickle("data/prepared/sales.pkl")
prices = pd.read_pickle("data/prepared/prices.pkl")

sales["year"] = sales.date.dt.year
sales["quarter"] = sales.date.dt.quarter

sales = pd.merge(left=sales, right=prices, on=["store", "item", "year", "quarter"], how="left")

aggs = {
    ("item",): ["mean"],
    ("store",): ["mean"],
    ("store", "item"): ["mean"],
}

for by, hows in tqdm(aggs.items()):

    for how in hows:

        how_name = how.__name__ if callable(how) else how
        name = f"prices_{how_name}_" + "_x_".join(by)
        path = f"data/features/{name}.pkl"
        agg = sales.groupby(list(by))["price_mid"].agg(how)
        agg = agg.rename(name)
        agg.to_pickle(path)
