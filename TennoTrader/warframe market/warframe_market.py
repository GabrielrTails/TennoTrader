# warframe_market.py

import time
import json
import threading
import tkinter as tk
from collections import deque

import requests
import urllib3
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ttkbootstrap import Window, Style, ttk
from tkinter import messagebox

from database import init_db, insert_record
from analise_dados import calcular_metricas

# ─── BANCO DE DADOS E MODELO ──────────────────────────────────────────────────
DB_PATH   = "warframe_market.db"
db_conn   = init_db()
MODEL     = None
running   = True

def train_model():
    global MODEL
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM market_data", conn, parse_dates=["ts"])
    conn.close()

    # usamos apenas as colunas já presentes
    feature_cols = ["spread", "demand", "supply", "liquidity", "score"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["avg_sell"]

    MODEL = RandomForestRegressor(n_estimators=100, random_state=42)
    MODEL.fit(X, y)
    print(f"[ML] Modelo treinado com {len(df)} registros")

threading.Thread(target=train_model, daemon=True).start()

# ─── PARÂMETROS GLOBAIS ───────────────────────────────────────────────────────
BASE_URL        = "https://api.warframe.market/v1"
HEADERS         = {
    "Accept":"application/json",
    "Content-Type":"application/json",
    "platform":"pc",
    "language":"en"
}
CACHE_FILE      = "cache.json"
BATCH_SIZE      = 20
INTERVAL_WORKER = 30    # s entre coleta
INTERVAL_UI     = 2     # s entre redraw UI

fila        = deque()
resultados  = {}  # {'item':…, 'spread':…, 'score':…, 'pred_sell':…}
ordem_atual = {"coluna":"pred_sell","reversa":False}
tema_escuro  = True

# ─── SESSÃO HTTP COM RETRIES ─────────────────────────────────────────────────
session = requests.Session()
retries = urllib3.util.retry.Retry(
    total=3, backoff_factor=1,
    status_forcelist=[429,500,502,503,504]
)
adapter = requests.adapters.HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ─── CACHE ────────────────────────────────────────────────────────────────────
def salvar_cache(path=CACHE_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(resultados.values()), f)
    except: pass

def carregar_cache(path=CACHE_FILE):
    try:
        data = json.load(open(path, encoding="utf-8"))
        return {d["item"]:d for d in data}
    except:
        return {}

# ─── API / CÁLCULO ────────────────────────────────────────────────────────────
def safe_request(endpoint):
    resp = session.get(f"{BASE_URL}{endpoint}", headers=HEADERS, timeout=10)
    if resp.status_code == 504:
        raise RuntimeError("API em manutenção")
    resp.raise_for_status()
    return resp.json()

def get_all_items():
    try:
        return [i["url_name"] 
                for i in safe_request("/items")["payload"]["items"]]
    except RuntimeError:
        return None
    except:
        return []

def get_orders(item):
    try:
        orders = safe_request(f"/items/{item}/orders")["payload"]["orders"]
        sells = [o for o in orders 
                 if o["order_type"]=="sell" and o["user"]["status"] in ("online","ingame")]
        buys  = [o for o in orders 
                 if o["order_type"]=="buy"  and o["user"]["status"] in ("online","ingame")]
        return sells, buys
    except RuntimeError:
        return None, None
    except:
        return [], []

def calcular_dados(item):
    sells, buys = get_orders(item)
    if sells is None:
        raise RuntimeError("API manutenção")
    if not sells and not buys:
        return None
    return {"item":item, **calcular_metricas(sells, buys)}

# ─── WORKERS ─────────────────────────────────────────────────────────────────
def data_worker():
    while running:
        for _ in range(min(BATCH_SIZE, len(fila))):
            item = fila.popleft()
            try:
                rec = calcular_dados(item)
                if rec:
                    insert_record(db_conn, rec)
                    resultados[rec["item"]] = rec
            except RuntimeError:
                pass
        time.sleep(INTERVAL_WORKER)

def predictions_worker():
    while MODEL is None:
        time.sleep(0.5)
    while running:
        for rec in resultados.values():
            feat = [rec["spread"], rec["demand"], rec["supply"],
                    rec["liquidity"], rec["score"]]
            try:
                rec["pred_sell"] = float(MODEL.predict([feat])[0])
            except:
                rec["pred_sell"] = 0.0
        time.sleep(INTERVAL_WORKER)

threading.Thread(target=data_worker, daemon=True).start()
threading.Thread(target=predictions_worker, daemon=True).start()

# ─── INICIALIZAÇÃO DE ITENS OU CACHE ────────────────────────────────────────
items = get_all_items()
if items is None:
    cache = carregar_cache()
    if cache:
        messagebox.showwarning("Manutenção","API indisponível, usando cache.")
        resultados.update(cache)
    else:
        messagebox.showerror("Manutenção",
            "API indisponível e sem cache.\nEncerre o app e tente mais tarde.")
        exit(1)
else:
    fila.extend(items)

# ─── EXPLICAÇÕES E RECOMENDAÇÕES ─────────────────────────────────────────────
def explain_terms():
    return (
        "Termos úteis:\n"
        "- Flip      : comprar barato e vender caro para lucro rápido.\n"
        "- Spread    : diferença entre preço médio de venda e de compra.\n"
        "- Score     : nossa métrica de oportunidade (mais alto → melhor).\n"
        "- Liquidez  : quantas ordens ativas (mais alto → mais rápido de vender).\n"
    )

def explain_columns():
    return (
        "Colunas da tabela:\n"
        "- Item            : nome do item.\n"
        "- Venda           : preço médio de venda.\n"
        "- Compra          : preço médio de compra.\n"
        "- Demanda         : ordens de compra ativas.\n"
        "- Oferta          : ordens de venda ativas.\n"
        "- Score           : nosso índice de oportunidade.\n"
        "- Mediana         : preço mediano de venda.\n"
        "- Pond. V.        : média ponderada de venda.\n"
        "- Spread          : (Venda – Compra).\n"
        "- Predição        : preço futuro previsto.\n"
    )

def generate_recommendations(data):
    if not data:
        return "Sem dados para recomendar."
    # Rápido Flip: alta score e spread moderado
    flip = sorted(
        [d for d in data if d["score"] >= 12 and d["spread"] >= 5],
        key=lambda x: x["score"], reverse=True
    )[:3]
    # Oportunidade de Arbitragem: spread alto e liquidez ≥5
    arb = sorted(
        [d for d in data if d["spread"] >= 10 and d["liquidity"] >= 5],
        key=lambda x: x["spread"], reverse=True
    )[:2]
    # Investimento Long-Term: liquidez alta e score médio
    hold = sorted(
        [d for d in data if d["liquidity"] >= 20 and 5 <= d["score"] < 12],
        key=lambda x: x["liquidity"], reverse=True
    )[:2]
    # Itens de Risco: spread baixo e liquidez baixa
    avoid = sorted(
        [d for d in data if d["spread"] < 3 and d["liquidity"] < 3],
        key=lambda x: x["spread"]
    )[:2]

    lines = ["🔍 Recomendações Profissionais:"]
    if flip:
        lines.append("• Flip Rápido:")
        for d in flip:
            lines.append(f"  - {d['item']} (Score {d['score']:.1f}, Spread {d['spread']:.1f})")
    if arb:
        lines.append("• Arbitragem:")
        for d in arb:
            lines.append(f"  - {d['item']} (Spread {d['spread']:.1f}, Liquidez {d['liquidity']})")
    if hold:
        lines.append("• Investimento (Long-Term):")
        for d in hold:
            lines.append(f"  - {d['item']} (Liquidez {d['liquidity']}, Score {d['score']:.1f})")
    if avoid:
        lines.append("• Itens de Risco (Evitar):")
        for d in avoid:
            lines.append(f"  - {d['item']} (Spread {d['spread']:.1f}, Liquidez {d['liquidity']})")

    lines.append("👉 Confira missões/grind que dropam estes itens e venda em regiões populosas.")
    return "\n".join(lines)

# ─── MONTAGEM DA UI ──────────────────────────────────────────────────────────
root = Window(themename="flatly")
root.title("TennoTrader — Painel de Previsões")
root.geometry("1200x700")

def alternar_tema():
    global tema_escuro
    tema_escuro = not tema_escuro
    style.theme_use("darkly" if tema_escuro else "flatly")

style = Style("flatly")
ttk.Button(root, text="🌓 Tema", command=alternar_tema,
           style="secondary.TButton").pack(pady=5, anchor="ne", padx=10)

nb = ttk.Notebook(root)
fr_resumo = ttk.Frame(nb)
fr_top    = ttk.Frame(nb)
fr_all    = ttk.Frame(nb)

nb.add(fr_resumo, text="Resumo")
nb.add(fr_top,    text="Top 10")
nb.add(fr_all,    text="Todos")
nb.pack(expand=True, fill="both")

lbl_summary = ttk.Label(fr_resumo, text="Gerando recomendações…",
                        anchor="nw", justify="left", font=("Segoe UI", 11))
lbl_summary.pack(padx=20, pady=(20,5), anchor="nw")

lbl_help = ttk.Label(fr_resumo, text=explain_terms(),
                     anchor="nw", justify="left",
                     font=("Segoe UI", 9), foreground="#555")
lbl_help.pack(padx=20, pady=(0,5), anchor="nw")

lbl_columns = ttk.Label(fr_resumo, text=explain_columns(),
                        anchor="nw", justify="left",
                        font=("Segoe UI", 9), foreground="#555")
lbl_columns.pack(padx=20, pady=(0,20), anchor="nw")

cols = (
    "item","avg_sell","avg_buy","demand","supply","score",
    "median_sell","weighted_avg_sell","spread","pred_sell"
)
hdrs = (
    "Item","Venda","Compra","Demanda","Oferta","Score",
    "Mediana","Pond. V.","Spread","Predição"
)

tree_top = ttk.Treeview(fr_top, columns=cols, show="headings")
tree_all = ttk.Treeview(fr_all, columns=cols, show="headings")

def ordenar_por_coluna(col):
    if ordem_atual["coluna"] == col:
        ordem_atual["reversa"] = not ordem_atual["reversa"]
    else:
        ordem_atual["coluna"] = col
        ordem_atual["reversa"] = False
    ui_refresh()

for tree in (tree_top, tree_all):
    for c,h in zip(cols, hdrs):
        tree.heading(c, text=h, anchor="center",
                     command=lambda c=c: ordenar_por_coluna(c))
        tree.column(c, width=100, anchor="center")
    tree.tag_configure("alto",     background="#2e7d32", foreground="white")
    tree.tag_configure("apertado", background="#f9a825", foreground="black")
    tree.tag_configure("baixo",    background="#c62828", foreground="white")
    tree.pack(expand=True, fill="both")

# ─── ATUALIZAÇÃO DA UI ───────────────────────────────────────────────────────
def ui_refresh():
    dados = list(resultados.values())
    lbl_summary.config(text=generate_recommendations(dados))

    chave = ordem_atual["coluna"]
    rev   = ordem_atual["reversa"]
    ordenados = sorted(dados, key=lambda x: x.get(chave,0), reverse=rev)

    tree_top.delete(*tree_top.get_children())
    for d in ordenados[:10]:
        iid = tree_top.insert("", "end", values=(
            d["item"],
            f"{d['avg_sell']}p",
            f"{d['avg_buy']}p",
            d["demand"], d["supply"], d["score"],
            d["median_sell"], d["weighted_avg_sell"], d["spread"],
            f"{d.get('pred_sell',0):.1f}p"
        ))
        if d["score"] >= 15:      tree_top.item(iid, tags=("alto",))
        elif d["spread"] < 5:      tree_top.item(iid, tags=("apertado",))
        elif d["liquidity"] < 3:   tree_top.item(iid, tags=("baixo",))

    tree_all.delete(*tree_all.get_children())
    for d in ordenados:
        iid = tree_all.insert("", "end", values=(
            d["item"],
            f"{d['avg_sell']}p",
            f"{d['avg_buy']}p",
            d["demand"], d["supply"], d["score"],
            d["median_sell"], d["weighted_avg_sell"], d["spread"],
            f"{d.get('pred_sell',0):.1f}p"
        ))
        if d["score"] >= 15:      tree_all.item(iid, tags=("alto",))
        elif d["spread"] < 5:      tree_all.item(iid, tags=("apertado",))
        elif d["liquidity"] < 3:   tree_all.item(iid, tags=("baixo",))

    root.after(int(INTERVAL_UI*1000), ui_refresh)

root.after(100, ui_refresh)

def on_close():
    global running
    running = False
    salvar_cache()
    db_conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
