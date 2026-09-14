# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - Andre Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/features.py
# SINTESE: Features pré-jogo (médias móveis shiftadas, calendário, adversário).
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Features pre-jogo: estatisticas dos jogos anteriores, nunca do jogo atual.

Toda janela e agrupada por (jogador, temporada) e deslocada com shift(1). O
agrupamento por temporada evita arrastar jogos da temporada passada - inclusive
playoffs - para dentro da media da temporada corrente.

As janelas usam groupby().rolling()/.ewm()/.expanding() em vez de
transform(lambda): o primeiro roda em Cython, o segundo executa a funcao grupo a
grupo em Python. Com ~2.500 jogadores x 5 temporadas a diferenca e de minutos.
"""

from __future__ import annotations

import pandas as pd

DECAY = 0.9  # peso 0.9**k por jogo de distancia; meia-vida ~6.6 jogos
GROUP = ["PLAYER_ID", "SEASON"]
WINDOWS = (3, 5, 10)
STARTER_MIN_THRESHOLD = 28.0
ROLLING_STATS = ("PTS", "MIN", "FGA", "FG3A", "FTA", "REB", "AST")


def _align(result: pd.Series, index: pd.Index) -> pd.Series:
    """Descarta as chaves de grupo do MultiIndex e realinha ao df original."""
    return result.droplevel(list(range(result.index.nlevels - 1))).reindex(index)


def build_opponent_strength(teams: pd.DataFrame) -> pd.DataFrame:
    """Pontos cedidos e ritmo do adversario ate o jogo anterior dele.

    Cada GAME_ID tem duas linhas no log por time, entao os pontos cedidos por um
    time sao os pontos marcados pelo outro. Preferimos isso a um defensive rating
    de temporada fechada, que resume jogos que ainda nao aconteceram.
    """
    t = teams[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].copy()

    total_by_game = t.groupby("GAME_ID")["PTS"].transform("sum")
    t["PTS_ALLOWED"] = total_by_game - t["PTS"]
    t["GAME_TOTAL"] = total_by_game

    t = t.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    keys = [t["TEAM_ID"], t["SEASON"]]
    grp = t.groupby(["TEAM_ID", "SEASON"], sort=False)

    for src, dest in (("PTS_ALLOWED", "opp_def_pts_allowed_prev"), ("GAME_TOTAL", "opp_pace_proxy_prev")):
        shifted = grp[src].shift(1)
        t[dest] = _align(shifted.groupby(keys, sort=False).expanding(min_periods=1).mean(), t.index)

    t["opp_games_played_prev"] = grp.cumcount()

    cols = ["GAME_ID", "TEAM_ABBREVIATION", "opp_def_pts_allowed_prev", "opp_pace_proxy_prev", "opp_games_played_prev"]
    return t[cols].rename(columns={"TEAM_ABBREVIATION": "OPPONENT"})


def build_features(players: pd.DataFrame, teams: pd.DataFrame | None = None) -> pd.DataFrame:
    df = players.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)

    keys = [df["PLAYER_ID"], df["SEASON"]]
    grp = df.groupby(GROUP, sort=False)

    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        shifted = grp[stat].shift(1)
        sg = shifted.groupby(keys, sort=False)
        for w in WINDOWS:
            df[f"{stat.lower()}_ma{w}_prev"] = _align(sg.rolling(w, min_periods=1).mean(), df.index)

    pts_shifted = grp["PTS"].shift(1).groupby(keys, sort=False)
    min_shifted = grp["MIN"].shift(1).groupby(keys, sort=False)

    df["pts_std5_prev"] = _align(pts_shifted.rolling(5, min_periods=1).std(), df.index)
    df["pts_ewm_prev"] = _align(pts_shifted.ewm(alpha=1 - DECAY, adjust=True).mean(), df.index)
    df["min_ewm_prev"] = _align(min_shifted.ewm(alpha=1 - DECAY, adjust=True).mean(), df.index)
    df["pts_season_mean_prev"] = _align(pts_shifted.expanding(min_periods=1).mean(), df.index)
    df["min_season_mean_prev"] = _align(min_shifted.expanding(min_periods=1).mean(), df.index)
    df["games_played_prev"] = grp.cumcount()

    prev_date = grp["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - prev_date).dt.days
    df["is_b2b"] = (df["rest_days"] == 1).astype("Int64")
    df.loc[df["rest_days"].isna(), "is_b2b"] = pd.NA

    df["is_home"] = df["IS_HOME"].astype("Int64")

    # Proxy: o game log nao traz START_POSITION e o box score custaria uma
    # requisicao por jogo. Ver ressalva no dicionario de dados.
    df["starter_proxy_prev"] = (df["min_ma5_prev"] >= STARTER_MIN_THRESHOLD).astype("Int64")
    df.loc[df["min_ma5_prev"].isna(), "starter_proxy_prev"] = pd.NA

    if teams is not None and not teams.empty:
        df = df.merge(build_opponent_strength(teams), on=["GAME_ID", "OPPONENT"], how="left")

    return df


PRE_GAME_FEATURES: list[str] = [
    "pts_ma3_prev", "pts_ma5_prev", "pts_ma10_prev", "pts_ewm_prev",
    "pts_std5_prev", "pts_season_mean_prev",
    "min_ma3_prev", "min_ma5_prev", "min_ma10_prev", "min_ewm_prev", "min_season_mean_prev",
    "fga_ma5_prev", "fg3a_ma5_prev", "fta_ma5_prev",
    "reb_ma5_prev", "ast_ma5_prev", "games_played_prev", "starter_proxy_prev",
    "rest_days", "is_b2b", "is_home",
    "opp_def_pts_allowed_prev", "opp_pace_proxy_prev", "opp_games_played_prev",
]

# So existem depois do jogo; nunca entram em X.
POST_GAME_COLUMNS: list[str] = [
    "PTS", "MIN", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "PLUS_MINUS", "WL",
]
