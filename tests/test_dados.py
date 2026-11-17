# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_dados.py
# SINTESE: Invariantes do dataset versionado e coerência dos manifestos e do dicionário.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: invariantes do dataset versionado.
# === FIM DO CABECALHO ===

"""Invariantes do dataset versionado e coerencia dos seus metadados.

Os testes de vazamento (test_no_leakage.py) provam que a CONSTRUCAO e correta
em dados sinteticos. Estes olham o arquivo que de fato e entregue: se alguem
regenerar o dataset com um filtro diferente, ou editar o CSV a mao, um destes
testes precisa falhar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_dataset import MIN_MINUTES_PREV, SPLIT_BY_SEASON
from src.collect import DEFAULT_SEASONS
from src.features import PRE_GAME_FEATURES, STARTER_MIN_THRESHOLD
from src.modeling import load_dataset

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW_MANIFEST = ROOT / "data" / "raw" / "manifest.json"
DICIONARIO = ROOT / "docs" / "DICIONARIO_DE_DADOS.md"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not (PROC / "dataset.parquet").exists():
        pytest.skip("dataset ainda nao gerado (rode python -m src.build_dataset)")
    return load_dataset()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((PROC / "dataset_manifest.json").read_text(encoding="utf-8"))


def test_chave_jogador_jogo_e_unica(df):
    assert not df.duplicated(["PLAYER_ID", "GAME_ID"]).any()


def test_colunas_de_identificacao_nao_tem_nulos(df):
    ids = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "GAME_ID", "GAME_DATE", "SEASON", "split"]
    assert int(df[ids].isna().sum().sum()) == 0


def test_pontos_sao_inteiros_nao_negativos(df):
    assert (df["y_pts"] >= 0).all()
    assert (df["y_pts"] == df["y_pts"].round()).all()
    assert df["y_pts"].max() < 100  # recorde da NBA e 100; acima disso e erro de dado


def test_alvo_de_classificacao_e_consistente_com_a_linha(df):
    assert set(df["y_over"].unique()) <= {0.0, 1.0}
    assert (df["y_over"] == (df["y_pts"] > df["line_synthetic"]).astype(float)).all()
    empates = df["y_pts"] == df["line_synthetic"]
    assert empates.any()
    assert (df.loc[empates, "y_over"] == 0).all(), "empate com a linha tem de contar como 'nao superou'"


def test_linha_sintetica_e_multiplo_de_meio_ponto(df):
    assert ((df["line_synthetic"] * 2) % 1 == 0).all()
    assert (df["line_synthetic"] >= 0).all()


def test_toda_linha_tem_ao_menos_dez_jogos_de_historico(df):
    """A linha e a mediana dos 10 jogos anteriores: sem 10 jogos nao ha linha."""
    assert (df["games_played_prev"] >= 10).all()


def test_so_jogadores_de_rotacao(df):
    assert (df["min_ma5_prev"] >= MIN_MINUTES_PREV).all()


def test_split_corresponde_a_temporada(df):
    esperado = df["SEASON"].map(SPLIT_BY_SEASON)
    assert (df["split"] == esperado).all()
    assert set(df["SEASON"].unique()) == set(DEFAULT_SEASONS)


def test_so_temporada_regular(df):
    assert not df["IS_PLAYOFF"].any()
    assert set(df["SEASON_TYPE"].unique()) == {"Regular Season"}


def test_back_to_back_coerente_com_dias_de_descanso(df):
    assert (df["rest_days"] >= 1).all()
    assert (df["is_b2b"] == (df["rest_days"] == 1).astype(float)).all()


def test_colunas_binarias_so_tem_zero_e_um(df):
    for col in ("is_home", "is_b2b", "starter_proxy_prev"):
        assert set(df[col].unique()) <= {0.0, 1.0}, col


def test_titular_estimado_segue_o_limiar_de_minutos(df):
    assert (df["starter_proxy_prev"] == (df["min_ma5_prev"] >= STARTER_MIN_THRESHOLD).astype(float)).all()


def test_faixas_fisicas_plausiveis(df):
    for col in [c for c in PRE_GAME_FEATURES if c.startswith("pts_") and "std" not in c]:
        assert df[col].between(0, 80).all(), col
    for col in [c for c in PRE_GAME_FEATURES if c.startswith("min_")]:
        assert df[col].between(0, 60).all(), col
    assert (df["pts_std5_prev"] >= 0).all()
    assert (df["opp_games_played_prev"] >= 0).all()
    assert df["opp_def_pts_allowed_prev"].between(60, 160).all()


def test_nenhum_time_joga_contra_si_mesmo(df):
    assert (df["TEAM_ABBREVIATION"] != df["OPPONENT"]).all()


def test_cada_jogo_tem_uma_data_e_dois_times(df):
    assert (df.groupby("GAME_ID")["GAME_DATE"].nunique() == 1).all()
    times = df.groupby("GAME_ID").apply(
        lambda g: len(set(g["TEAM_ABBREVIATION"]) | set(g["OPPONENT"])), include_groups=False
    )
    assert (times == 2).all()


def test_jogos_de_um_mesmo_jogador_sao_unicos_por_data(df):
    assert not df.duplicated(["PLAYER_ID", "GAME_DATE"]).any()


def test_csv_e_parquet_sao_o_mesmo_dataset():
    csv_path, pq_path = PROC / "dataset.csv", PROC / "dataset.parquet"
    if not (csv_path.exists() and pq_path.exists()):
        pytest.skip("dataset ainda nao gerado")
    csv = pd.read_csv(csv_path, dtype={"GAME_ID": str})
    pq = pd.read_parquet(pq_path)
    assert list(csv.columns) == list(pq.columns)
    assert len(csv) == len(pq)
    numericas = [c for c in pq.columns if pd.api.types.is_numeric_dtype(pq[c]) and not pd.api.types.is_bool_dtype(pq[c])]
    for col in numericas:
        a = pd.to_numeric(csv[col]).to_numpy(dtype=float)
        b = pq[col].astype("float64").to_numpy()
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), col
    for col in ("GAME_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "SEASON", "split"):
        assert (csv[col].astype(str).to_numpy() == pq[col].astype(str).to_numpy()).all(), col
    assert (pd.to_datetime(csv["GAME_DATE"]) == pd.to_datetime(pq["GAME_DATE"])).all()


def test_manifesto_descreve_o_dataset_real(df, manifest):
    assert manifest["linhas"] == len(df)
    pq = pd.read_parquet(PROC / "dataset.parquet")
    assert manifest["colunas"] == pq.shape[1]
    assert manifest["distribuicao_split"] == df["split"].value_counts().to_dict()
    assert manifest["features_pre_jogo"] == PRE_GAME_FEATURES
    assert manifest["alvo"]["linhas_com_alvo"] == len(df)
    assert manifest["alvo"]["proporcao_classe_positiva"] == pytest.approx(df["y_over"].mean(), abs=5e-5)
    assert manifest["filtros"]["min_minutes_prev"] == MIN_MINUTES_PREV
    assert manifest["split_temporal"] == SPLIT_BY_SEASON


def test_manifesto_da_coleta_bate_com_o_do_dataset(manifest):
    if not RAW_MANIFEST.exists():
        pytest.skip("manifesto da coleta ausente")
    raw = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    assert raw["temporadas"] == DEFAULT_SEASONS
    assert raw["season_types"] == ["Regular Season"]
    assert raw["linhas"]["player_game_logs"] == manifest["filtros"]["linhas_brutas"]
    assert raw["fonte"].startswith("stats.nba.com")


def test_dicionario_de_dados_cobre_todas_as_colunas():
    cols = list(pd.read_parquet(PROC / "dataset.parquet").columns)
    texto = DICIONARIO.read_text(encoding="utf-8")
    faltando = [c for c in cols if f"`{c}`" not in texto]
    assert not faltando, f"colunas do dataset sem entrada em docs/DICIONARIO_DE_DADOS.md: {faltando}"


def test_empates_com_a_linha_estao_documentados(df):
    """README e limitacoes citam o numero de empates; ele precisa ser o real."""
    empates = int((df["y_pts"] == df["line_synthetic"]).sum())
    citado = f"{empates:,}".replace(",", ".")
    for doc in ("README.md", "docs/LIMITACOES_E_VIESES.md"):
        assert citado in (ROOT / doc).read_text(encoding="utf-8"), f"{doc} nao cita {citado} empates"
