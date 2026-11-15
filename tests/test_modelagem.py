# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_modelagem.py
# SINTESE: Testes da modelagem: features, pipeline, métricas, intervalos, sementes.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: testes de modelagem.
# === FIM DO CABECALHO ===

"""Testes da modelagem: features permitidas, pipeline sem vazamento, metricas,
intervalos e reprodutibilidade.

Todos usam dados sinteticos pequenos, exceto quando o nome diz o contrario,
entao rodam em segundos e nao dependem do dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.features import POST_GAME_COLUMNS, PRE_GAME_FEATURES  # noqa: E402
from src.metrics import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    paired_cluster_bootstrap,
    regression_metrics,
)
from src.modeling import (  # noqa: E402
    CLF_EXTRA_FEATURES,
    FEATURE_FAMILIES,
    add_clf_features,
    assert_no_post_game,
    clf_feature_set,
    regression_zoo,
)
from src.uncertainty import ConformalBand, coverage_report, oracle_local, oracle_season_loo  # noqa: E402


# ----------------------------------------------------------------- features

def test_familias_particionam_as_features_pre_jogo():
    todas = [c for cols in FEATURE_FAMILIES.values() for c in cols]
    assert len(todas) == len(set(todas)), "feature repetida entre familias"
    assert set(todas) == set(PRE_GAME_FEATURES)


def test_features_dos_modelos_nao_tem_coluna_pos_jogo():
    for variant in ("completo", "sem_linha_inteira", "so_pre_jogo"):
        assert_no_post_game(clf_feature_set(variant))
    assert_no_post_game(PRE_GAME_FEATURES)
    with pytest.raises(AssertionError):
        assert_no_post_game(PRE_GAME_FEATURES + ["fga"])  # FGA do proprio jogo, em minusculas
    with pytest.raises(AssertionError):
        assert_no_post_game(["y_pts"])


def test_features_da_classificacao_so_usam_a_propria_linha():
    """As features extras dependem apenas de colunas pre-jogo da mesma linha."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(10, 3, 50) for c in PRE_GAME_FEATURES})
    df["line_synthetic"] = rng.integers(5, 25, 50) + rng.choice([0.0, 0.5], 50)
    a = add_clf_features(df)
    b = add_clf_features(df.assign(y_pts=999, PTS=999, MIN=99))  # pos-jogo nao muda nada
    pd.testing.assert_frame_equal(a[CLF_EXTRA_FEATURES], b[CLF_EXTRA_FEATURES])
    assert set(a["linha_inteira"].unique()) <= {0.0, 1.0}
    assert (a.loc[df["line_synthetic"] % 1 == 0, "linha_inteira"] == 1).all()


def test_scaler_e_ajustado_so_com_o_treino():
    """O StandardScaler do pipeline guarda a media do treino, nao a do teste."""
    pipe, _ = regression_zoo()["ridge"]
    rng = np.random.default_rng(1)
    X_tr = pd.DataFrame({c: rng.normal(10, 1, 300) for c in PRE_GAME_FEATURES})
    X_te = X_tr + 1000  # teste com outra escala
    y = rng.normal(12, 3, 300)
    pipe.fit(X_tr, y)
    esperado = X_tr.copy()
    esperado["rest_days"] = esperado["rest_days"].clip(upper=10)  # mesmo limite do pipeline
    scaler = pipe.named_steps["scaler"]
    assert np.allclose(scaler.mean_, esperado.mean().to_numpy(), atol=1e-9)
    pipe.predict(X_te)  # prever nao reajusta nada
    assert np.allclose(scaler.mean_, esperado.mean().to_numpy(), atol=1e-9)


def test_post_game_columns_continuam_maiusculas_como_no_bruto():
    # documenta a convencao que torna necessaria a comparacao sem caixa
    assert all(c == c.upper() for c in POST_GAME_COLUMNS)


# ----------------------------------------------------------------- metricas

def test_metricas_de_regressao_em_caso_de_brinquedo():
    m = regression_metrics([1, 2, 3], [2, 2, 5])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["vies"] == pytest.approx(1.0)


def test_brier_e_ece():
    y = np.array([0, 1] * 500)
    m = classification_metrics(y, np.full(len(y), 0.5), base_rate=0.5)
    assert m["brier"] == pytest.approx(0.25)
    assert m["brier_skill"] == pytest.approx(0.0, abs=1e-12)
    # calibracao perfeita sintetica: y ~ Bernoulli(p)
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 50_000)
    yy = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(yy, p) < 0.01
    # confianca excessiva: sempre 0,9 em dados 50/50
    assert expected_calibration_error(y, np.full(len(y), 0.9)) == pytest.approx(0.4, abs=1e-9)


def test_bootstrap_pareado():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(100), 20)
    a = rng.exponential(5, clusters.size)
    igual = paired_cluster_bootstrap(a, a.copy(), clusters, n_boot=500)
    assert igual["delta"] == 0 and igual["ic_inf"] <= 0 <= igual["ic_sup"]
    pior = paired_cluster_bootstrap(a + 1.0, a, clusters, n_boot=500)
    assert pior["ic_inf"] > 0  # erro sistematicamente maior: IC nao cobre zero


def test_bootstrap_por_cluster_e_mais_largo_que_por_linha():
    """Com efeito de jogador, reamostrar linhas subestima a incerteza."""
    rng = np.random.default_rng(4)
    clusters = np.repeat(np.arange(60), 50)
    efeito = rng.normal(0, 1, 60)[clusters]  # diferenca correlacionada dentro do jogador
    diff = efeito + rng.normal(0, 0.2, clusters.size)
    por_cluster = paired_cluster_bootstrap(diff, np.zeros_like(diff), clusters, n_boot=800)
    por_linha = paired_cluster_bootstrap(diff, np.zeros_like(diff), np.arange(clusters.size), n_boot=800)
    largura = lambda r: r["ic_sup"] - r["ic_inf"]  # noqa: E731
    assert largura(por_cluster) > 2 * largura(por_linha)


# ----------------------------------------------------------------- incerteza

def test_conformal_cobre_o_nominal_em_dados_heterocedasticos():
    rng = np.random.default_rng(5)

    def amostra(n):
        mu = rng.uniform(3, 30, n)
        return mu, rng.poisson(mu).astype(float)

    mu_cal, y_cal = amostra(20_000)
    mu_te, y_te = amostra(20_000)
    for alpha in (0.2, 0.1):
        band = ConformalBand.fit(y_cal, mu_cal, alpha)
        lo, hi = band.predict(mu_te)
        cob = coverage_report(y_te, lo, hi)["cobertura"]
        assert abs(cob - (1 - alpha)) < 0.015, (alpha, cob)
    # intervalo cresce com a previsao (normalizacao por raiz de mu)
    lo, hi = band.predict(np.array([4.0, 25.0]))
    assert (hi - lo)[1] > (hi - lo)[0]
    assert (lo >= 0).all()


def test_oraculos_excluem_o_proprio_jogo():
    df = pd.DataFrame({
        "PLAYER_ID": [1] * 5,
        "SEASON": ["2025-26"] * 5,
        "GAME_DATE": pd.date_range("2026-01-01", periods=5),
        "y_pts": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    loo = oracle_season_loo(df)
    assert loo[0] == pytest.approx((20 + 30 + 40 + 50) / 4)
    local = oracle_local(df, half_window=1)
    assert local[2] == pytest.approx((20 + 40) / 2)  # vizinhos, sem o proprio
    assert local[0] == pytest.approx(20.0)


# --------------------------------------------------------- reprodutibilidade

def test_mesma_semente_mesma_previsao():
    rng = np.random.default_rng(6)
    X = pd.DataFrame({c: rng.normal(10, 3, 400) for c in PRE_GAME_FEATURES})
    y = X["pts_ewm_prev"] + rng.normal(0, 2, 400)
    pipe_a, _ = regression_zoo()["random_forest"]
    pipe_b, _ = regression_zoo()["random_forest"]
    params = {"model__n_estimators": 20, "model__n_jobs": 1}
    pa = pipe_a.set_params(**params).fit(X, y).predict(X)
    pb = pipe_b.set_params(**params).fit(X, y).predict(X)
    np.testing.assert_array_equal(pa, pb)
