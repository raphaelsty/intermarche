import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

X_train = pd.read_pickle("data/train.pkl")
X_test = pd.read_pickle("data/test.pkl")

y_train = np.log1p(X_train.pop("quantity"))

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train = X_train.fillna(-1.0)
X_test = X_test.fillna(-1.0)

cv = KFold(3, shuffle=True, random_state=42)
sub = pd.Series(0.0, index=X_test.index)

model = lgbm.LGBMRegressor(
    objective="poisson",
    num_leaves=2 ** 5 - 1,
    n_estimators=500,
    random_state=42,
)

for fit_idx, val_idx in tqdm(cv.split(X_train, y_train), total=cv.n_splits):

    X_fit = X_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_fit = y_train.iloc[fit_idx]
    y_val = y_train.iloc[val_idx]

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=("fit", "val"),
        eval_metric="l2",
        early_stopping_rounds=10,
        feature_name=X_fit.columns.tolist(),
        verbose=50,
    )

    sub += model.predict(X_test)

sub = np.expm1(sub / cv.n_splits)
sub.to_pickle(f"data/sub.pkl")
