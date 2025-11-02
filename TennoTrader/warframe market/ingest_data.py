# ingest_data.py

import pandas as pd
import sqlite3

# 1) Defina o esquema / colunas na mesma ordem do seu TSV
cols = [
    "ts", "item", "avg_sell", "avg_buy",
    "median_sell", "median_buy",
    "weighted_avg_sell", "weighted_avg_buy",
    "spread", "demand", "supply", "liquidity", "score"
]

# 2) Caminho para o seu TSV e para o DB
TSV_PATH = "market_data.tsv"
DB_PATH  = "warframe_market.db"

# 3) Leia o TSV em um DataFrame
df = pd.read_csv(
    TSV_PATH,
    sep="\t",
    header=None,
    names=cols,
    dtype={
        "ts": int,
        "item": str,
        "avg_sell": float,
        "avg_buy": float,
        "median_sell": float,
        "median_buy": float,
        "weighted_avg_sell": float,
        "weighted_avg_buy": float,
        "spread": float,
        "demand": int,
        "supply": int,
        "liquidity": int,
        "score": float
    }
)

# 4) Converta timestamp e reorganize índice (opcional)
df["datetime"] = pd.to_datetime(df["ts"], unit="s")
df.set_index("datetime", inplace=True)

# 5) Insere ou anexa ao banco SQLite
conn = sqlite3.connect(DB_PATH)

# Se quiser recriar a tabela do zero, descomente a próxima linha:
# conn.execute("DROP TABLE IF EXISTS market_data;")

# Grava no SQLite — se a tabela não existir, ela será criada automaticamente
df.reset_index(drop=True).to_sql(
    "market_data",
    conn,
    if_exists="append",  # "replace" para apagar antes e criar nova
    index=False
)

conn.close()

print("Importação finalizada:", len(df), "registros inseridos.")
