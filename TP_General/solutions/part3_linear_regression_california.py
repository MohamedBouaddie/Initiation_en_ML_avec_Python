"""
TP Général — Partie 3 : Régression linéaire simple (California Housing)
Prédire Target (MedHouseVal) uniquement avec MedInc.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Target"] = housing.target

X_reg = df[["MedInc"]]
y_reg = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

print("Coef =", reg.coef_[0], "| Intercept =", reg.intercept_)

# Visualisation
plt.figure()
plt.scatter(X_test, y_test)
# Droite : prédire sur X trié pour tracer proprement
X_sorted = X_test.sort_values(by="MedInc")
plt.plot(X_sorted, reg.predict(X_sorted))
plt.title("Régression linéaire : Target ~ MedInc")
plt.xlabel("MedInc")
plt.ylabel("Target (100k$)")
plt.show()

# Prédiction pour MedInc = 5.0
pred_5 = reg.predict([[5.0]])[0]
print("Prix prédit pour MedInc=5.0 :", pred_5, "(en 100k$)")
