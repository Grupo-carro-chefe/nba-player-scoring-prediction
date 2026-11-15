# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_splits.py
# SINTESE: Testes da divisão temporal treino/validação/teste e das dobras do CV.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: não sobreposição e ordem temporal dos splits.
# === FIM DO CABECALHO ===

"""Testes da divisao temporal: sem sobreposicao, em ordem, dobras so para o passado.

O relatorio N1 afirma que ha verificacao automatizada de que treino, validacao e
teste nao se sobrepoem. Estes testes sao essa verificacao.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.modeling import DATASET, assert_disjoint_temporal, load_dataset, season_folds


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not DATASET.exists():
        pytest.skip("dataset ainda nao gerado (rode python -m src.build_dataset)")
    return load_dataset()


def test_splits_sao_disjuntos_e_ordenados_no_tempo(df):
    assert_disjoint_temporal(df)  # levanta AssertionError se houver problema


def test_nenhuma_linha_fora_dos_splits(df):
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_cada_temporada_pertence_a_um_unico_split(df):
    por_temporada = df.groupby("SEASON")["split"].nunique()
    assert (por_temporada == 1).all(), por_temporada[por_temporada > 1].to_dict()


def test_fim_do_treino_antes_do_inicio_do_teste(df):
    assert df.loc[df["split"] == "train", "GAME_DATE"].max() < df.loc[df["split"] == "val", "GAME_DATE"].min()
    assert df.loc[df["split"] == "val", "GAME_DATE"].max() < df.loc[df["split"] == "test", "GAME_DATE"].min()


def test_dobras_do_cv_treinam_so_no_passado(df):
    train = df[df["split"] == "train"].reset_index(drop=True)
    folds = season_folds(train)
    assert len(folds) == train["SEASON"].nunique() - 1
    anterior_va = None
    for tr, va in folds:
        assert train.loc[tr, "GAME_DATE"].max() < train.loc[va, "GAME_DATE"].min()
        assert train.loc[va, "SEASON"].nunique() == 1  # valida em uma temporada inteira
        if anterior_va is not None:
            assert set(anterior_va) <= set(tr)  # janela expansiva
        anterior_va = va


def test_assert_disjoint_detecta_sobreposicao():
    base = pd.DataFrame({
        "GAME_ID": ["a", "b", "c"],
        "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "split": ["train", "val", "test"],
    })
    assert_disjoint_temporal(base)
    ruim = base.copy()
    ruim.loc[2, "GAME_DATE"] = pd.Timestamp("2024-01-15")  # teste antes da validacao
    with pytest.raises(AssertionError):
        assert_disjoint_temporal(ruim)
    repetido = base.copy()
    repetido.loc[2, "GAME_ID"] = "a"
    with pytest.raises(AssertionError):
        assert_disjoint_temporal(repetido)
