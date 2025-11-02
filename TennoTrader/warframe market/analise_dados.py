# analise_dados.py

import statistics

def remover_outliers(valores, limite=1.5):
    # só tenta se tiver dados suficientes
    if len(valores) < 4:
        return valores

    try:
        # quantis em 4 partes: retorna [Q1, Q2, Q3]
        q1, _, q3 = statistics.quantiles(valores, n=4)
        iqr = q3 - q1
        lower, upper = q1 - limite * iqr, q3 + limite * iqr
        return [v for v in valores if lower <= v <= upper]
    except Exception:
        # se der qualquer problema, volta com os dados brutos
        return valores

# resto do código permanece igual
def media_simples(valores):
    return sum(valores) / len(valores) if valores else 0

def media_ponderada(valores, pesos):
    total_peso = sum(pesos)
    return sum(v * p for v, p in zip(valores, pesos)) / total_peso if total_peso else 0

def mediana(valores):
    return statistics.median(valores) if valores else 0

def calcular_metricas(sells, buys):
    sells_val = sorted(o["platinum"] for o in sells)
    buys_val  = sorted(o["platinum"] for o in buys)

    # remove outliers só quando for seguro
    sells_filtrados = remover_outliers(sells_val)
    buys_filtrados  = remover_outliers(buys_val)

    avg_sell = media_simples(sells_filtrados)
    avg_buy  = media_simples(buys_filtrados)
    median_sell = mediana(sells_filtrados)
    median_buy  = mediana(buys_filtrados)

    volume_sell = [o.get("volume", 1) for o in sells]
    volume_buy  = [o.get("volume", 1) for o in buys]

    weighted_avg_sell = media_ponderada(sells_val, volume_sell)
    weighted_avg_buy  = media_ponderada(buys_val,  volume_buy)

    spread    = weighted_avg_sell - weighted_avg_buy
    demand    = len(buys)
    supply    = len(sells)
    liquidity = min(demand, supply)

    score = liquidity * 3 + (demand - supply) + (10 - spread)

    return {
        "avg_sell": round(avg_sell, 1),
        "avg_buy":  round(avg_buy,  1),
        "median_sell": round(median_sell, 1),
        "median_buy":  round(median_buy,  1),
        "weighted_avg_sell": round(weighted_avg_sell, 1),
        "weighted_avg_buy":  round(weighted_avg_buy,  1),
        "spread": round(spread, 1),
        "demand":    demand,
        "supply":    supply,
        "liquidity": liquidity,
        "score":     round(score, 2)
    }
