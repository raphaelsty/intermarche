import pandas as pd

sales = pd.read_pickle("data/prepared/sales.pkl")

ts = pd.Timestamp
french_holidays = {
    ts("2018-01-01"),
    ts("2018-04-02"),
    ts("2018-04-30"),
    ts("2018-05-01"),
    ts("2018-05-07"),
    ts("2018-05-08"),
    ts("2018-05-10"),
    ts("2018-05-11"),
    ts("2018-05-10"),
    ts("2018-05-21"),
    ts("2018-07-14"),
    ts("2018-08-15"),
    ts("2018-11-01"),
    ts("2018-11-02"),
    ts("2018-11-11"),
    ts("2018-12-24"),
    ts("2018-12-25"),
    ts("2018-12-26"),
    ts("2018-12-31"),
    ts("2019-01-01"),
}

(
    pd.DataFrame(
        {
            "day": sales.date.dt.day,
            "is_holiday": sales.date.isin(french_holidays),
            "is_weekday": sales.date.dt.dayofweek < 5,
        },
        index=sales.index,
    ).to_pickle("data/features/cal.pkl")
)
