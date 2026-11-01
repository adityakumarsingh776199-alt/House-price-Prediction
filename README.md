# House Price Prediction (PriceLens)
Regression project that estimates a house price from area, bedrooms, bathrooms, age and locality. Three models are compared with 5-fold cross-validation, the best is evaluated on a held-out test set, and a Streamlit app serves live estimates.

**Stack:** Python, Pandas, scikit-learn, Matplotlib, Streamlit

## Run
```
pip install -r requirements.txt
python generate_data.py     # creates data/houses.csv (sample data; swap in a real dataset with the same columns)
python train.py             # CV comparison, metrics.json, model.joblib, charts in outputs/
streamlit run app.py
```
> The dataset is synthetic sample data. Metrics will differ on real data.
