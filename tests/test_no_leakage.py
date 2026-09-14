# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_no_leakage.py
# SINTESE: Testes de vazamento temporal nas features.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Testes de vazamento temporal nas features.

O principal e test_future_cannot_change_past: alteramos um jogo futuro e
exigimos que nenhuma feature das linhas anteriores mude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import POST_GAME_COLUMNS, PRE_GAME_FEATURES, build_features
from src.target import add_targets, make_synthetic_line


def _fake_players(n: int = 12, season: str = "2023-24", player_id: int = 1, start_pts: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="2D")
    return pd.DataFrame(
        {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": f"Jogador {player_id}",
            "SEASON": season,
            "SEASON_TYPE": "Regular Season",
            "IS_PLAYOFF": False,
            "GAME_ID": [f"00{player_id}{i:04d}" for i in range(n)],
            "GAME_DATE": dates,
            "TEAM_ABBREVIATION": "AAA",
            "OPPONENT": "BBB",
            "IS_HOME": [i % 2 == 0 for i in range(n)],
            "PTS": [start_pts + i for i in range(n)],
            "MIN": [30.0] * n,
            "FGA": [15.0] * n,
            "FG3A": [5.0] * n,
            "FTA": [4.0] * n,
            "REB": [5.0] * n,
            "AST": [4.0] * n,
        }
    )


def test_rolling_mean_uses_only_past():
    df = build_features(_fake_players(n=5, start_pts=10))  # PTS = 10,11,12,13,14

    assert pd.isna(df.loc[0, "pts_ma3_prev"])
    assert df.loc[1, "pts_ma3_prev"] == pytest.approx(10.0)
    assert df.loc[2, "pts_ma3_prev"] == pytest.approx(10.5)
    assert df.loc[4, "pts_ma3_prev"] == pytest.approx((11 + 12 + 13) / 3)


def test_future_cannot_change_past():
    base = _fake_players(n=10)
    poisoned = base.copy()
    poisoned.loc[poisoned.index[-1], "PTS"] = 999

    f_base = build_features(base)
    f_poison = build_features(poisoned)

    past = slice(0, len(base) - 1)
    cols = [c for c in PRE_GAME_FEATURES if c in f_base.columns]

    pd.testing.assert_frame_equal(
        f_base.loc[past, cols].reset_index(drop=True),
        f_poison.loc[past, cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_synthetic_line_does_not_see_current_game():
    base = _fake_players(n=15)
    poisoned = base.copy()
    poisoned.loc[poisoned.index[-1], "PTS"] = 999

    pd.testing.assert_series_equal(
        make_synthetic_line(base), make_synthetic_line(poisoned), check_names=False
    )


def test_window_resets_between_seasons():
    s1 = _fake_players(n=6, season="2022-23", start_pts=40)
    s2 = _fake_players(n=6, season="2023-24", start_pts=5)
    df = build_features(pd.concat([s1, s2], ignore_index=True))

    first_of_s2 = df[df["SEASON"] == "2023-24"].iloc[0]
    assert pd.isna(first_of_s2["pts_ma3_prev"])
    assert first_of_s2["games_played_prev"] == 0

    # a media da 2a temporada nao pode refletir os 40+ pontos da 1a
    assert df[df["SEASON"] == "2023-24"]["pts_ma10_prev"].max(skipna=True) < 20


def test_no_post_game_column_in_feature_list():
    overlap = set(PRE_GAME_FEATURES) & set(POST_GAME_COLUMNS)
    assert not overlap, f"colunas pos-jogo em X: {overlap}"
    assert "y_pts" not in PRE_GAME_FEATURES
    assert "y_over" not in PRE_GAME_FEATURES


def test_no_feature_perfectly_predicts_target():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "PLAYER_ID": 1, "PLAYER_NAME": "X", "SEASON": "2023-24",
            "SEASON_TYPE": "Regular Season", "IS_PLAYOFF": False,
            "GAME_ID": [f"{i:08d}" for i in range(n)],
            "GAME_DATE": pd.date_range("2024-01-01", periods=n, freq="D"),
            "TEAM_ABBREVIATION": "AAA", "OPPONENT": "BBB",
            "IS_HOME": rng.integers(0, 2, n).astype(bool),
            "PTS": rng.normal(22, 8, n).round().clip(0),
            "MIN": rng.normal(31, 5, n).clip(5),
            "FGA": rng.normal(16, 4, n).clip(0),
            "FG3A": rng.normal(5, 2, n).clip(0),
            "FTA": rng.normal(4, 2, n).clip(0),
            "REB": rng.normal(5, 2, n).clip(0),
            "AST": rng.normal(4, 2, n).clip(0),
        }
    )
    feats = add_targets(build_features(df))
    cols = [c for c in PRE_GAME_FEATURES if c in feats.columns]
    corr = feats[cols].corrwith(feats["y_pts"]).abs()

    suspeitas = corr[corr > 0.95].dropna()
    assert suspeitas.empty, f"correlacao quase perfeita com o alvo: {suspeitas.to_dict()}"
