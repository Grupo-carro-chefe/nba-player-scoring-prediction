# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - Andre Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/io_utils.py
# SINTESE: Leitura e escrita de tabelas em parquet com fallback para CSV.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Leitura e escrita de tabelas em parquet, com fallback para CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATE_COLS = ("GAME_DATE",)


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def write_table(df: pd.DataFrame, base: Path) -> list[Path]:
    """Grava base.csv sempre e base.parquet se houver engine."""
    base.parent.mkdir(parents=True, exist_ok=True)
    written = [base.with_suffix(".csv")]
    df.to_csv(written[0], index=False)
    if parquet_available():
        p = base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        written.append(p)
    else:
        print(f"  [aviso] sem pyarrow: gravei so {written[0].name}")
    return written


def read_table(base: Path) -> pd.DataFrame:
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")

    if parquet.exists() and parquet_available():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"nao encontrei {parquet.name} nem {csv.name} em {base.parent}")

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def table_exists(base: Path) -> bool:
    return base.with_suffix(".parquet").exists() or base.with_suffix(".csv").exists()
