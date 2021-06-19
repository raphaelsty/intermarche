import pandas as pd
from tqdm import tqdm

sales = pd.read_pickle("data/prepared/sales.pkl")

sales["day_of_week"] = sales.date.dt.weekday

aggs = {
    ("store",): ["mean"],
    ("item",): ["mean"],
    ("store", "item"): ["mean", "std"],
    ("store", "day_of_week"): ["mean", "std"],
    ("item", "day_of_week"): ["mean", "std"],
    ("store", "item", "day_of_week"): ["mean", "std"],
}

for by, hows in tqdm(aggs.items()):

    for how in hows:

        how_name = how.__name__ if callable(how) else how
        name = f"{how_name}_" + "_x_".join(by)
        path = f"data/features/{name}.pkl"
        agg = sales.groupby(list(by))["quantity"].agg(how)
        agg = agg.rename(name)
        agg.to_pickle(path)
