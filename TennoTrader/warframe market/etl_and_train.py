# etl_and_train.py

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# 1) Conecta e lê todos os dados
DB_PATH = "warframe_market.db"
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM market_data", conn, parse_dates=["ts"])

# 2) Cria coluna datetime e ordena por item
df["datetime"] = pd.to_datetime(df["ts"], unit="s")
df = df.sort_values(["item", "datetime"]).set_index("datetime")

# 3) Features de série temporal por item
df["pct_sell"] = (
    df.groupby("item")["avg_sell"]
      .pct_change()
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
)
df["vol_3"] = (
    df.groupby("item")["pct_sell"]
      .transform(lambda s: s.rolling(3, min_periods=1).std())
      .fillna(0)
)
df["ma_sell_3"] = (
    df.groupby("item")["avg_sell"]
      .transform(lambda s: s.rolling(3, min_periods=1).mean())
      .fillna(0)
)

# 4) Seleção de features e alvo
FEATURE_COLS = [
    "pct_sell","vol_3","ma_sell_3",
    "spread","demand","supply","liquidity","score"
]
TARGET = "avg_sell"

# 5) Limpa infinitos e preenche NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

# 6) Split treino/teste por data ou random 80/20 se test vazio
CUTOFF = "2025-09-30 13:00"
train = df.loc[df.index <= CUTOFF]
test  = df.loc[df.index >  CUTOFF]

if test.empty:
    print("⚠️ Test set vazio com o cutoff. Usando random split 80/20.")
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
else:
    X_train, y_train = train[FEATURE_COLS].loc[train.index], train[TARGET]
    X_test,  y_test  = df[FEATURE_COLS].loc[test.index],  test[TARGET]

print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

# 7) Treina modelos
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42)\
         .fit(X_train, y_train)

# 8) Predições
pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)

# 9) Avaliação sem parâmetro 'squared'
def report(name, true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    print(f"{name:<20} MAE: {mae:.2f} | RMSE: {rmse:.2f}")

report("Linear Regression", y_test, pred_lr)
report("Random Forest"     , y_test, pred_rf)
