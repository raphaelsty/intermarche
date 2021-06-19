import pandas as pd

items = pd.read_csv("data/raw/nomenclature_produits.csv", dtype={"ID_ARTC": "uint16"})
items = items.rename(
    columns={
        "ID_ARTC": "id",
        "LB_VENT_RAYN": "shelf",
        "LB_VENT_FAML": "family",
        "LB_VENT_SOUS_FAML": "sub_family",
    }
)
items.to_pickle("data/prepared/items.pkl")

stores = pd.read_csv("data/raw/points_de_vente.csv", dtype={"ID_PDV": "uint16"})
stores = stores.rename(
    columns={
        "ID_PDV": "id",
        "ID_VOCT": "vocation",
        "ID_REGN": "region",
        "NB_CAIS_GRP": "size",
        "SURF_GRP": "surface",
    }
)
stores.to_pickle("data/prepared/stores.pkl")


prices = pd.read_csv(
    "data/raw/prix_vente.csv",
    dtype={"ID_PDV": "uint16", "ID_ARTC": "uint16", "ANNEE": "uint16", "TRIMESTRE": "uint16"},
)
prices = prices.rename(
    columns={
        "ID_PDV": "store",
        "ID_ARTC": "item",
        "ANNEE": "year",
        "TRIMESTRE": "quarter",
        "PRIX_UNITAIRE": "price",
    }
)
prices["price"] = prices["price"].str.replace("Moins de 0.99€", "Entre 0 et 0.99€", regex=False)
price_interval = (
    prices["price"].str.extract("(?P<low>\d\d?) et (?P<high>\d\d?\.\d\d)").astype(float)
)
prices["price_low"] = price_interval["low"]
prices["price_high"] = price_interval["high"]
prices["price_mid"] = prices.eval("(price_low + price_high) / 2")
prices = prices.drop(columns="price")
prices.to_pickle("data/prepared/prices.pkl")


sales_2018 = pd.read_csv("data/raw/ventes_2018.csv", parse_dates=["DATE"])
sales_2018 = sales_2018.rename(
    columns={"ID_PDV": "store", "ID_ARTC": "item", "DATE": "date", "QTE": "quantity"}
)

# Remove samples not in test
prices = pd.read_pickle("data/prepared/prices.pkl")
prices = prices[prices.year == 2019].copy()
prices = prices[["store", "item"]].assign(drop=False).copy()
sales_2018 = sales_2018.merge(prices, on=["store", "item"], how="left")
sales_2018["drop"] = sales_2018["drop"].fillna(True)
sales_2018 = sales_2018[sales_2018["drop"] == False].drop("drop", axis="columns").copy()

# dates x stores x items
calendar = pd.date_range(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2019-03-31"))
stores_x_items = sales_2018.groupby(["store", "item"]).size().index
sales = pd.merge(pd.Series(calendar, name="date"), stores_x_items.to_frame(), how="cross")

# Add past sales
sales = pd.merge(sales, sales_2018, on=["store", "item", "date"], how="left")

# Add 5 zeros after each pair (store, item) if there are missing values.
n = 6

agg = (
    sales.groupby(["item", "store"])
    .apply(
        lambda x: x.quantity.isnull()
        .astype(int)
        .groupby(x.quantity.notnull().astype(int).cumsum())
        .cumsum()
    )
    .reset_index()
)
agg = agg.set_index(agg["level_2"].values).sort_index()
agg = agg.drop(["item", "store", "level_2"], axis="columns")
agg = agg.rename(columns={"quantity": "nans"})
sales = pd.concat([sales, agg], axis="columns")

sales = sales.loc[sales.quantity.notnull() | sales.nans.lt(n) | sales.date.dt.year.eq(2019)]
sales.loc[sales.date.dt.year.eq(2018) & sales.quantity.isnull(), "quantity"] = sales.loc[
    sales.date.dt.year.eq(2018) & sales.quantity.isnull(), "quantity"
].fillna(0)

sales = sales.drop("nans", axis="columns")

# Save
sales = sales.astype({"store": "UInt16", "item": "UInt16", "quantity": "UInt16"})
sales.to_pickle("data/prepared/sales.pkl")
