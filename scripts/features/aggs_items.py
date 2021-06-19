import pandas as pd
from tqdm import tqdm

sales = pd.read_pickle("data/prepared/sales.pkl")

sales["day_of_week"] = sales.date.dt.day_of_week

items = pd.read_pickle("data/prepared/items.pkl").rename(columns={"id": "item"})
sales = pd.merge(left=sales, right=items, on="item", how="left")

aggs = {
    ("store",): ["mean"],
    ("store", "family"): ["mean"],
    ("store", "sub_family"): ["mean"],
    ("store", "family", "day_of_week"): ["mean"],
    ("store", "sub_family", "day_of_week"): ["mean"],
}

for by, hows in tqdm(aggs.items()):

    for how in hows:

        how_name = how.__name__ if callable(how) else how
        name = f"{how_name}_" + "_x_".join(by)
        path = f"data/features/{name}.pkl"
        agg = sales.groupby(list(by))["quantity"].agg(how)
        agg = agg.rename(name)
        agg.to_pickle(path)
