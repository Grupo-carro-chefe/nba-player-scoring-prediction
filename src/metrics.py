# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/metrics.py
# SINTESE: Métricas de regressão, classificação, calibração e bootstrap por jogador.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: métricas, ECE e bootstrap pareado por jogador.
# === FIM DO CABECALHO ===

"""Metricas de regressao, classificacao e calibracao, e bootstrap por cluster.

Duas escolhas merecem nota:

- As metricas por faixa de minutos usam a media de minutos ANTERIOR ao jogo
  (`min_ma5_prev`). Os minutos do proprio jogo so existem depois que ele acaba.
- A comparacao entre modelos usa bootstrap pareado POR JOGADOR. As linhas de um
  mesmo atleta nao sao independentes, e um bootstrap linha a linha produziria
  intervalos de confianca otimistas demais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.modeling import SEED, minute_band


# ----------------------------------------------------------------------------
# Regressao
# ----------------------------------------------------------------------------

def regression_metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "vies": float(np.mean(pred - y)),  # > 0: superestima
    }


def regression_by_group(df: pd.DataFrame, pred, groups: pd.Series, y_col: str = "y_pts") -> pd.DataFrame:
    """MAE/RMSE por grupo, com o erro relativo a media do grupo."""
    tmp = pd.DataFrame({"y": df[y_col].to_numpy(dtype=float), "p": np.asarray(pred, dtype=float), "g": groups.to_numpy()})
    rows = []
    for g, part in tmp.groupby("g", observed=True, sort=True):
        m = regression_metrics(part["y"], part["p"])
        m["grupo"] = str(g)
        m["media_pontos"] = float(part["y"].mean())
        m["mae_relativo"] = m["mae"] / m["media_pontos"] if m["media_pontos"] else float("nan")
        rows.append(m)
    return pd.DataFrame(rows).set_index("grupo")


def regression_by_minutes(df: pd.DataFrame, pred) -> pd.DataFrame:
    return regression_by_group(df, pred, minute_band(df))


# ----------------------------------------------------------------------------
# Classificacao e calibracao
# ----------------------------------------------------------------------------

def expected_calibration_error(y, proba, n_bins: int = 10) -> float:
    """ECE com faixas de mesmo tamanho (quantis da probabilidade prevista).

    Media, ponderada pelo tamanho da faixa, de |taxa observada - prob. media|.
    Faixas por quantil evitam faixas vazias quando as probabilidades se
    concentram perto de 0,5, que e o caso deste problema.
    """
    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba, kind="mergesort")
    ece = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        ece += len(idx) / len(y) * abs(y[idx].mean() - proba[idx].mean())
    return float(ece)


def reliability_table(y, proba, n_bins: int = 10) -> pd.DataFrame:
    """Probabilidade media prevista vs. frequencia observada, por faixa."""
    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba, kind="mergesort")
    rows = []
    for k, idx in enumerate(np.array_split(order, n_bins)):
        rows.append({
            "faixa": k + 1,
            "n": int(len(idx)),
            "prob_media_prevista": float(proba[idx].mean()),
            "frequencia_observada": float(y[idx].mean()),
        })
    return pd.DataFrame(rows)


def classification_metrics(y, proba, base_rate: float, threshold: float = 0.5) -> dict[str, float]:
    """Metricas de acerto, de discriminacao e de calibracao.

    `base_rate` e a taxa da classe positiva NO TREINO. Ela define o estimador
    ingenuo (prever sempre essa probabilidade), contra o qual se mede o Brier
    skill score: BSS = 1 - Brier_modelo / Brier_ingenuo. BSS > 0 quer dizer que
    o modelo carrega informacao alem da frequencia media.
    """
    y = np.asarray(y, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= threshold).astype(int)
    brier = float(brier_score_loss(y, proba))
    brier_naive = float(brier_score_loss(y, np.full(len(y), base_rate)))
    return {
        "n": int(len(y)),
        "taxa_positiva": float(y.mean()),
        "acuracia": float(accuracy_score(y, pred)),
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "revocacao": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": brier,
        "brier_ingenuo": brier_naive,
        "brier_skill": 1.0 - brier / brier_naive,
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "ece": expected_calibration_error(y, proba),
    }


# ----------------------------------------------------------------------------
# Bootstrap pareado por cluster
# ----------------------------------------------------------------------------

def paired_cluster_bootstrap(
    loss_a,
    loss_b,
    clusters,
    n_boot: int = 2000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """IC para a diferenca media de perda (A - B), reamostrando clusters.

    Os clusters (jogadores) sao sorteados com reposicao; todas as linhas de um
    jogador entram juntas. Como a perda media e uma razao de somas, basta
    pre-agregar soma das diferencas e contagem por cluster: cada replica custa
    uma multiplicacao de matriz, nao uma reavaliacao do modelo.

    Retorna delta (negativo: A erra menos que B), IC e p-valor bicaudal
    aproximado (fracao de replicas do lado oposto de zero, vezes 2).
    """
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    codes, uniq = pd.factorize(np.asarray(clusters))
    k = len(uniq)
    soma = np.bincount(codes, weights=diff, minlength=k)
    cont = np.bincount(codes, minlength=k).astype(float)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    # pesos[b, c] = quantas vezes o cluster c foi sorteado na replica b
    pesos = np.zeros((n_boot, k))
    np.add.at(pesos, (np.repeat(np.arange(n_boot), k), draws.ravel()), 1.0)
    deltas = (pesos @ soma) / (pesos @ cont)

    delta = float(diff.mean())
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return {
        "delta": delta,
        "ic_inf": float(lo),
        "ic_sup": float(hi),
        "p_valor": float(min(1.0, p)),
        "n_clusters": int(k),
        "n_boot": int(n_boot),
    }


def cluster_bootstrap_statistic(
    y,
    score,
    clusters,
    statistic,
    n_boot: int = 1000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """IC por cluster para uma estatistica nao aditiva (ex.: AUC)."""
    y = np.asarray(y)
    score = np.asarray(score)
    codes, uniq = pd.factorize(np.asarray(clusters))
    k = len(uniq)
    idx_by_cluster = [np.flatnonzero(codes == c) for c in range(k)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(statistic(y[idx], score[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"valor": float(statistic(y, score)), "ic_inf": float(lo), "ic_sup": float(hi), "n_boot": len(vals)}
