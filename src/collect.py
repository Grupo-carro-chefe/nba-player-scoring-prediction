# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/collect.py
# SINTESE: Coleta de game logs da NBA com cache, retry e manifest.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Coleta game logs da NBA no grao jogador-jogo.

    python -m src.collect
    python -m src.collect --seasons 2023-24 2024-25
    python -m src.collect --include-playoffs

Usamos LeagueGameLog em vez de PlayerGameLog: o mesmo grao sai em 1 request por
temporada, contra ~550 (um por jogador). O log por time e coletado junto e serve
so para calcular a defesa do adversario em src/features.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.io_utils import read_table, table_exists, write_table

# 2019-20 e 2020-21 ficam de fora: bolha, calendario curto e ginasio vazio
# distorcem media movel e efeito casa/fora (docs/LIMITACOES_E_VIESES.md).
DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

RAW_DIR = Path("data/raw")
CACHE_DIR = Path(".cache_nba")

SLEEP_BETWEEN = 1.2  # a API derruba rajadas
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
TIMEOUT = 60

NUMERIC_COLS = [
    "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS",
]


@dataclass
class CollectionReport:
    seasons: list[str]
    season_types: list[str]
    endpoints: list[str] = field(default_factory=list)
    requests_made: int = 0
    cache_hits: int = 0
    filters: dict = field(default_factory=dict)
    row_counts: dict = field(default_factory=dict)


def _cache_path(kind: str, season: str, season_type: str) -> Path:
    slug = season_type.lower().replace(" ", "-")
    return CACHE_DIR / f"{kind}_{season}_{slug}"


def _fetch_league_game_log(
    season: str,
    season_type: str,
    player_or_team: str,
    report: CollectionReport,
) -> pd.DataFrame:
    kind = "player" if player_or_team == "P" else "team"
    cache = _cache_path(kind, season, season_type)
    if table_exists(cache):
        report.cache_hits += 1
        print(f"  [cache] {kind} {season} {season_type}")
        return read_table(cache)

    # import aqui para o modulo seguir importavel sem a nba_api instalada
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as err:
        raise SystemExit(
            "nba_api nao esta instalada. Rode: pip install -r requirements.txt"
        ) from err

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [api]   {kind} {season} {season_type} (tentativa {attempt})")
            resp = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation=player_or_team,
                timeout=TIMEOUT,
            )
            df = resp.get_data_frames()[0]
            report.requests_made += 1
            ep = f"leaguegamelog.LeagueGameLog(player_or_team_abbreviation={player_or_team!r})"
            if ep not in report.endpoints:
                report.endpoints.append(ep)
            write_table(df, cache)
            time.sleep(SLEEP_BETWEEN)
            return df
        except Exception as err:  # noqa: BLE001
            last_err = err
            wait = BACKOFF_BASE**attempt
            print(f"          falhou ({err.__class__.__name__}); aguardando {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError(f"falha ao coletar {kind} {season} {season_type}: {last_err}")


def _normalize(df: pd.DataFrame, season: str, season_type: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.upper() for c in out.columns]
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    out["SEASON"] = season
    out["SEASON_TYPE"] = season_type
    out["IS_PLAYOFF"] = season_type == "Playoffs"

    # MATCHUP vem como "LAL vs. PHX" (casa) ou "LAL @ PHX" (fora)
    matchup = out["MATCHUP"].astype(str)
    out["IS_HOME"] = ~matchup.str.contains("@", regex=False)
    out["OPPONENT"] = (
        matchup.str.replace(".", "", regex=False)
        .str.split(r"\s+(?:vs|@)\s+", regex=True)
        .str[-1]
        .str.strip()
    )

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def collect(
    seasons: list[str] | None = None,
    include_playoffs: bool = False,
    out_dir: Path = RAW_DIR,
) -> dict[str, pd.DataFrame]:
    seasons = seasons or DEFAULT_SEASONS
    season_types = ["Regular Season"] + (["Playoffs"] if include_playoffs else [])

    report = CollectionReport(
        seasons=seasons,
        season_types=season_types,
        filters={
            "grao": "uma linha por jogador-jogo",
            "temporadas_excluidas": ["2019-20", "2020-21"],
            "motivo_exclusao": "temporadas atipicas de COVID",
            "playoffs_incluidos": include_playoffs,
        },
    )

    players: list[pd.DataFrame] = []
    teams: list[pd.DataFrame] = []
    for season in seasons:
        print(f"Temporada {season}")
        for st in season_types:
            players.append(_normalize(_fetch_league_game_log(season, st, "P", report), season, st))
            teams.append(_normalize(_fetch_league_game_log(season, st, "T", report), season, st))

    df_players = pd.concat(players, ignore_index=True).sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df_teams = pd.concat(teams, ignore_index=True).sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    for name, df in (("player_game_logs", df_players), ("team_game_logs", df_teams)):
        write_table(df, out_dir / name)
        report.row_counts[name] = int(len(df))

    try:
        import nba_api

        version = getattr(nba_api, "__version__", "desconhecida")
    except ImportError:
        version = "nao instalada"

    manifest = {
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonte": "stats.nba.com via nba_api",
        "nba_api_version": version,
        "endpoints": report.endpoints,
        "temporadas": report.seasons,
        "season_types": report.season_types,
        "requests_realizados": report.requests_made,
        "cache_hits": report.cache_hits,
        "linhas": report.row_counts,
        "filtros": report.filters,
        "arquivos": sorted(p.name for p in out_dir.glob("*_game_logs.*")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\njogador-jogo: {report.row_counts['player_game_logs']:,}")
    print(f"time-jogo   : {report.row_counts['team_game_logs']:,}")
    print(f"manifest    : {out_dir / 'manifest.json'}")
    return {"players": df_players, "teams": df_teams}


def main() -> None:
    ap = argparse.ArgumentParser(description="Coleta game logs da NBA.")
    ap.add_argument("--seasons", nargs="+", default=None, help=f"padrao: {' '.join(DEFAULT_SEASONS)}")
    ap.add_argument("--include-playoffs", action="store_true")
    ap.add_argument("--out", default=str(RAW_DIR))
    args = ap.parse_args()
    collect(seasons=args.seasons, include_playoffs=args.include_playoffs, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
