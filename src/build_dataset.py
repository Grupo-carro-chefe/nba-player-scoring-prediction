# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/build_dataset.py
# SINTESE: Pipeline de coleta bruta até dataset analítico com split temporal.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Monta o dataset analitico a partir dos dados brutos.

    python -m src.build_dataset

Le data/raw e escreve data/processed/dataset.* + dataset_manifest.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features import POST_GAME_COLUMNS, PRE_GAME_FEATURES, build_features
from src.io_utils import read_table, table_exists, write_table
from src.target import add_targets, target_report

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

# Recorta para jogadores de rotacao usando a media de minutos ANTERIOR, nao os
# minutos do jogo atual (que seriam vazamento).
MIN_MINUTES_PREV = 15.0

# Split temporal. Aleatorio colocaria o jogo de 12/01 no teste e o de 13/01 no
# treino, deixando o modelo ver o estado futuro do proprio jogador.
SPLIT_BY_SEASON = {
    "2021-22": "train",
    "2022-23": "train",
    "2023-24": "train",
    "2024-25": "val",
    "2025-26": "test",
}

ID_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT",
    "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "IS_PLAYOFF",
]


def build(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROC_DIR,
    min_minutes_prev: float = MIN_MINUTES_PREV,
) -> pd.DataFrame:
    players_base = raw_dir / "player_game_logs"
    if not table_exists(players_base):
        raise SystemExit(
            f"nao encontrei dados brutos em {raw_dir}. Rode primeiro: python -m src.collect"
        )

    players = read_table(players_base)
    teams_base = raw_dir / "team_game_logs"
    teams = read_table(teams_base) if table_exists(teams_base) else None

    df = add_targets(build_features(players, teams))

    before = len(df)
    df = df[df["min_ma5_prev"].fillna(0) >= min_minutes_prev].copy()
    dropped_minutes = before - len(df)

    before = len(df)
    df = df[df["y_over"].notna()].copy()
    dropped_no_target = before - len(df)

    df["split"] = df["SEASON"].map(SPLIT_BY_SEASON).fillna("unassigned")

    keep = [c for c in ID_COLUMNS + PRE_GAME_FEATURES + ["line_synthetic", "y_pts", "y_over", "split"] if c in df.columns]
    out = df[keep].sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)

    write_table(out, out_dir / "dataset")

    manifest = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "linhas": int(len(out)),
        "colunas": int(out.shape[1]),
        "features_pre_jogo": [c for c in PRE_GAME_FEATURES if c in out.columns],
        "colunas_pos_jogo_removidas": POST_GAME_COLUMNS,
        "alvos": {"regressao": "y_pts", "classificacao": "y_over", "linha": "line_synthetic"},
        "filtros": {
            "min_minutes_prev": min_minutes_prev,
            "descartadas_por_minutos": int(dropped_minutes),
            "descartadas_sem_alvo": int(dropped_no_target),
        },
        "split_temporal": SPLIT_BY_SEASON,
        "distribuicao_split": out["split"].value_counts().to_dict(),
        "alvo": target_report(out),
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"dataset: {len(out):,} linhas x {out.shape[1]} colunas -> {out_dir}/dataset.*")
    print(f"split  : {manifest['distribuicao_split']}")
    print(f"classe positiva: {manifest['alvo']['proporcao_classe_positiva']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Monta o dataset analitico.")
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(PROC_DIR))
    ap.add_argument("--min-minutes-prev", type=float, default=MIN_MINUTES_PREV)
    args = ap.parse_args()
    build(Path(args.raw), Path(args.out), args.min_minutes_prev)


if __name__ == "__main__":
    main()
