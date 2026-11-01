"""Create a sample housing dataset (data/houses.csv). Replace it with a real dataset if you have one."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 5000
mult = {"Sector 5": 1.5, "Model Town": 1.3, "Civil Lines": 1.2, "Green Park": 1.1,
        "City Center": 1.0, "Old Town": .8, "Industrial Area": .7, "Outskirts": .6}
loc = rng.choice(list(mult), n)
area = rng.normal(1200, 500, n).clip(400, 4000).round()
bed = np.clip(np.round(area / 450 + rng.normal(0, .6, n)), 1, 6).astype(int)
bath = np.clip(bed - rng.integers(0, 2, n), 1, 5)
age = rng.integers(0, 41, n)
price = (area * 4500 * np.array([mult[l] for l in loc]) + bed * 120000 + bath * 80000 - age * 18000)
price = (price * rng.lognormal(0, .08, n)).clip(500000).round(-3)
pd.DataFrame({"area_sqft": area, "bedrooms": bed, "bathrooms": bath, "age_years": age,
              "locality": loc, "price": price}).to_csv("data/houses.csv", index=False)
print("saved data/houses.csv", n)
