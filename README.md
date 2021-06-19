4th place solution to [datafactory challenge by Intermarché](https://challenge.datafactory-intermarche.fr/fr/challenge/1/details). The objective of the challenge is to predict the sales made by intermarche in the first quarter of 2019. We have the data of the past year (2018) to train our model to fit the sales.

The data are structured such as:

|    date    | store | item | quantity |
|:----------:|:-----:|:----:|:--------:|
| 2018-01-01 |   1   |  12  |     1    |
| 2018-01-01 |   1   |  17  |     2    |
| 2018-01-01 |   1   |  22  |     3    |


We have additional tables available such as:

- Product characteristics.
- Store characteristics.
- Product prices by store and by quarter.

The main difficulty of the challenge is to find the days for which a store has recorded no sales for a given product.
Indeed, Intermarché does not provide records for which the target variable (quantity) is equal to 0. I found that adding up to 5 zeros after a sale for a given pair (store / item) maximized the performance of my model and limited the overfitting of my aggregates.

The list of my features:

- Aggregates by item / store (mean + std)
- Aggregates on prices. (mean)
- Aggregates on the characteristics of the stores. (mean)
- Aggregates on product characteristics. (mean)
- Rolling medians over the last 9 weeks.
- Features on dates. (weekend / holidays / day of the week)

I used LightGBM and performed a 3-fold cross-validation with bagging to make my prediction. I transformed the target variable to train my model using `quantity = log(1 + quantity)`. Poisson loss helps a bit. I didn't look for the hyperparameters of the model.

Finally I set all predictions of February and March as the predictions of the second and third week of January.

Also I set to 0 the set of predictions associated to triplets (store / item / day of the week) for which we have not enough records in the training set. 

To reproduce my results, you must download the data in the folder `data/raw`.

```sh
python scripts/prepare_raw_data.py
python scripts/features/aggs_items.py
python scripts/features/aggs_prices.py
python scripts/features/aggs_stores.py
python scripts/features/aggs.py 
python scripts/features/lags.py
python scripts/features/cal.py 
python scripts/make_train_test.py
python scripts/learn.py
python scripts/polish_sub.py
```