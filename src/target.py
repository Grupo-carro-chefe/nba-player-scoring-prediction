# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - Andre Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/target.py
# SINTESE: Alvos de regressão e classificação com linha sintética.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Alvos: pontos (regressao) e superacao da linha (classificacao).

Nao ha fonte gratuita e reproduzivel de linha historica de props, entao usamos
uma linha sintetica: mediana dos 10 jogos anteriores, arredondada para .5.
O trade-off esta em docs/AUDITORIA_TECNICA.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LINE_WINDOW = 10
GROUP = ["PLAYER_ID", "SEASON"]


def round_to_half(s: pd.Series) -> pd.Series:
    return (s * 2).round() / 2


def make_synthetic_line(df: pd.DataFrame, window: int = LINE_WINDOW) -> pd.Series:
    """Mediana dos `window` jogos anteriores, arredondada para .5.

    Mediana em vez de media porque um jogo de 45 pontos ou um de 2 minutos por
    lesao puxariam a media para longe do patamar real do jogador. O .5 elimina
    empate: os pontos sao inteiros, entao nunca ha push.
    """
    d = df.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"])
    med = d.groupby(GROUP, sort=False)["PTS"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=window).median()
    )
    return round_to_half(med).reindex(df.index)


def add_targets(df: pd.DataFrame, window: int = LINE_WINDOW) -> pd.DataFrame:
    out = df.copy()
    out["y_pts"] = pd.to_numeric(out["PTS"], errors="coerce")
    out["line_synthetic"] = make_synthetic_line(out, window=window)

    # NaN enquanto o jogador nao tem `window` jogos na temporada; essas linhas
    # saem do dataset em build_dataset, nao viram zero.
    over = out["y_pts"] > out["line_synthetic"]
    out["y_over"] = np.where(out["line_synthetic"].isna(), np.nan, over.astype(float))
    return out


def target_report(df: pd.DataFrame) -> dict:
    total = len(df)
    usable = int(df["y_over"].notna().sum())
    pos = float(df["y_over"].mean()) if usable else float("nan")
    return {
        "linhas_totais": total,
        "linhas_com_alvo": usable,
        "linhas_sem_historico": total - usable,
        "proporcao_classe_positiva": round(pos, 4),
        "janela_da_linha": LINE_WINDOW,
    }
