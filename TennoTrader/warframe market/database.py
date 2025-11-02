import sqlite3
import time

DB_PATH = "warframe_market.db"

def init_db(path=DB_PATH):
    """Inicializa o banco e cria tabela se não existir."""
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            ts           INTEGER,
            item         TEXT,
            avg_sell     REAL,
            avg_buy      REAL,
            median_sell  REAL,
            median_buy   REAL,
            weighted_avg_sell REAL,
            weighted_avg_buy  REAL,
            spread       REAL,
            demand       INTEGER,
            supply       INTEGER,
            liquidity    INTEGER,
            score        REAL
        )
    """)
    conn.commit()
    return conn

def insert_record(conn, record: dict):
    """
    Insere um registro no formato retornado por calcular_metricas,
    adicionando timestamp.
    """
    ts = int(time.time())
    fields = (
        ts, record["item"],
        record["avg_sell"], record["avg_buy"],
        record["median_sell"], record["median_buy"],
        record["weighted_avg_sell"], record["weighted_avg_buy"],
        record["spread"],
        record["demand"], record["supply"],
        record["liquidity"], record["score"]
    )
    c = conn.cursor()
    c.execute("""
        INSERT INTO market_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, fields)
    conn.commit()
