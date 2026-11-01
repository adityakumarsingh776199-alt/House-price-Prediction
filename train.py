"""Compare regression models with cross-validation, evaluate the best on a held-out test set, save it."""
import json
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUM, CAT = ["area_sqft", "bedrooms", "bathrooms", "age_years"], ["locality"]
df = pd.read_csv("data/houses.csv")
X, y = df[NUM + CAT], df["price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)


def make(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(300, random_state=42, n_jobs=-1),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
cv = {}
for name, m in models.items():
    cv[name] = float(cross_val_score(make(m), Xtr, ytr, cv=5, scoring="r2").mean()); print(f"{name}: CV R2 = {cv[name]:.3f}")
best = max(cv, key=cv.get)
pipe = make(models[best]).fit(Xtr, ytr)
pred = pipe.predict(Xte)
metrics = {"best_model": best, "cv_r2": cv, "test_r2": round(r2_score(yte, pred), 4),
           "test_rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 0), "test_mae": round(mean_absolute_error(yte, pred), 0)}
json.dump(metrics, open("outputs/metrics.json", "w"), indent=2); print(metrics)
joblib.dump(pipe, "outputs/model.joblib")

imp = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=42, scoring="r2")
s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6, 3.5)); s.plot.barh(ax=ax, color="#1b2cf0"); ax.set(title="Feature importance (permutation)")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=150)
fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(yte / 1e5, pred / 1e5, s=6, alpha=.4)
lim = [0, max(yte.max(), pred.max()) / 1e5]; ax.plot(lim, lim, "r--"); ax.set(xlabel="Actual (₹ lakh)", ylabel="Predicted (₹ lakh)", title="Actual vs predicted")
fig.tight_layout(); fig.savefig("outputs/actual_vs_predicted.png", dpi=150)
