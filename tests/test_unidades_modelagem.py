# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_unidades_modelagem.py
# SINTESE: Métricas, splits, catálogo de modelos, conformal e oráculos em casos pequenos.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: métricas, splits, modelos, conformal e oráculos.
# === FIM DO CABECALHO ===

"""Testes de unidade de metrics, modeling e uncertainty com valores calculados a mao.

test_modelagem.py cobre as propriedades gerais (sem vazamento, cobertura nominal,
semente). Aqui ficam os casos pequenos em que o resultado certo cabe na cabeca,
mais os caminhos de erro das travas, que ate agora nunca eram exercitados.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.features import PRE_GAME_FEATURES
from src.metrics import (
    classification_metrics,
    cluster_bootstrap_statistic,
    expected_calibration_error,
    paired_cluster_bootstrap,
    regression_by_group,
    regression_by_minutes,
    regression_metrics,
    reliability_table,
)
from src.modeling import (
    CLF_EXTRA_FEATURES,
    REG_BASELINES,
    REST_DAYS_CAP,
    SEED,
    add_clf_features,
    assert_disjoint_temporal,
    assert_no_post_game,
    cap_rest_days,
    classification_zoo,
    clf_feature_set,
    get_split,
    load_dataset,
    minute_band,
    predict_baselines,
    regression_zoo,
    season_folds,
)
from src.uncertainty import (
    MIN_SCALE,
    ConformalBand,
    coverage_by_group,
    coverage_report,
    oracle_local,
    oracle_median,
    oracle_season_loo,
)


# ----------------------------------------------------------------------------
# metrics: regressao
# ----------------------------------------------------------------------------

def test_metricas_de_regressao_incluem_vies_com_sinal():
    m = regression_metrics([10, 20, 30], [12, 18, 36])
    assert m["n"] == 3
    assert m["mae"] == pytest.approx((2 + 2 + 6) / 3)
    assert m["rmse"] == pytest.approx(math.sqrt((4 + 4 + 36) / 3))
    assert m["vies"] == pytest.approx((2 - 2 + 6) / 3)  # positivo: superestima


def test_erro_por_grupo():
    df = pd.DataFrame({"y_pts": [10, 20, 30, 40]})
    r = regression_by_group(df, [12, 18, 30, 50], pd.Series(["a", "a", "b", "b"]))
    assert list(r.index) == ["a", "b"]
    assert r.loc["a", "mae"] == pytest.approx(2.0) and r.loc["b", "mae"] == pytest.approx(5.0)
    assert r.loc["a", "media_pontos"] == pytest.approx(15.0)
    assert r.loc["a", "mae_relativo"] == pytest.approx(2 / 15)
    assert r.loc["b", "vies"] == pytest.approx(5.0)


def test_faixas_de_minutos_usam_o_limite_inferior_inclusivo():
    df = pd.DataFrame({"min_ma5_prev": [15, 19.99, 20, 34.99, 35, 100, 14.99]})
    faixas = minute_band(df)
    assert faixas.iloc[:6].astype(str).tolist() == ["15-20", "15-20", "20-25", "30-35", "35+", "35+"]
    assert pd.isna(faixas.iloc[6])  # abaixo de 15 min nao e jogador de rotacao


def test_erro_por_faixa_de_minutos_na_ordem_natural():
    df = pd.DataFrame({"min_ma5_prev": [16, 22, 27, 32, 40], "y_pts": [10, 11, 12, 13, 14]})
    r = regression_by_minutes(df, [11, 11, 11, 11, 11])
    assert list(r.index) == ["15-20", "20-25", "25-30", "30-35", "35+"]
    assert (r["n"] == 1).all()
    assert r.loc["35+", "mae"] == pytest.approx(3.0)


# ----------------------------------------------------------------------------
# metrics: classificacao e calibracao
# ----------------------------------------------------------------------------

def test_ece_com_uma_faixa_e_a_diferenca_das_medias():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert expected_calibration_error(y, np.full(10, 0.6), n_bins=1) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(10, 0.9), n_bins=1) == pytest.approx(0.3)


def test_ece_de_previsao_perfeita_e_zero_e_fica_entre_zero_e_um():
    y = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    ece = expected_calibration_error(rng.integers(0, 2, 500), rng.random(500))
    assert 0.0 <= ece <= 1.0


def test_tabela_de_confiabilidade():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)  # bem calibrada por construcao
    t = reliability_table(y, p, n_bins=10)
    assert t["n"].sum() == 1000 and len(t) == 10
    assert t["prob_media_prevista"].is_monotonic_increasing
    assert (t["prob_media_prevista"] - t["frequencia_observada"]).abs().max() < 0.12
    assert list(t.columns) == ["faixa", "n", "prob_media_prevista", "frequencia_observada"]


def test_metricas_de_classificacao_em_caso_de_brinquedo():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]
    m = classification_metrics(y, p, base_rate=0.5)
    assert m["n"] == 4 and m["taxa_positiva"] == 0.5
    for k in ("acuracia", "precisao", "revocacao", "f1", "auc"):
        assert m[k] == 1.0, k
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.16 + 0.01) / 4)
    assert m["brier_ingenuo"] == pytest.approx(0.25)
    assert m["brier_skill"] == pytest.approx(1 - 0.085 / 0.25)
    assert m["log_loss"] == pytest.approx(-(2 * math.log(0.9) + 2 * math.log(0.6)) / 4)


def test_limiar_de_decisao_e_respeitado():
    m = classification_metrics([0, 1, 1], [0.6, 0.7, 0.8], base_rate=0.5, threshold=0.95)
    assert m["precisao"] == 0.0 and m["revocacao"] == 0.0 and m["acuracia"] == pytest.approx(1 / 3)


def test_prever_sempre_a_taxa_base_nao_tem_habilidade():
    y = np.array([0, 1, 1, 0, 1])
    m = classification_metrics(y, np.full(5, 0.6), base_rate=0.6)
    assert m["brier_skill"] == pytest.approx(0.0)
    assert m["auc"] == pytest.approx(0.5)


def test_auc_indefinida_com_uma_so_classe_e_log_loss_continua_finita():
    m = classification_metrics([1, 1, 1], [0.7, 0.8, 0.9], base_rate=0.5)
    assert math.isnan(m["auc"]) and math.isfinite(m["log_loss"])


def test_probabilidades_extremas_nao_geram_perda_infinita():
    m = classification_metrics([1, 0], [0.0, 1.0], base_rate=0.5)  # erra com certeza absoluta
    assert math.isfinite(m["log_loss"]) and m["log_loss"] > 5


# ----------------------------------------------------------------------------
# metrics: bootstrap por cluster
# ----------------------------------------------------------------------------

def _clusters(n_jog=20, por_jog=5):
    rng = np.random.default_rng(0)
    return np.repeat(np.arange(n_jog), por_jog), rng.random(n_jog * por_jog)


def test_bootstrap_detecta_melhoria_constante():
    clusters, base = _clusters()
    r = paired_cluster_bootstrap(base - 1.0, base, clusters, n_boot=500)
    assert r["delta"] == pytest.approx(-1.0)
    assert r["ic_sup"] < 0 and r["p_valor"] == 0.0
    assert r["n_clusters"] == 20 and r["n_boot"] == 500


def test_bootstrap_de_perdas_identicas_nao_acha_diferenca():
    clusters, base = _clusters()
    r = paired_cluster_bootstrap(base, base.copy(), clusters, n_boot=200)
    assert r["delta"] == 0 and r["ic_inf"] == 0 and r["ic_sup"] == 0 and r["p_valor"] == 1.0


def test_bootstrap_e_reprodutivel_e_depende_da_semente():
    clusters, base = _clusters()
    ruido = base + np.random.default_rng(5).normal(0, 0.3, len(base))
    a = paired_cluster_bootstrap(ruido, base, clusters, n_boot=300, seed=1)
    assert a == paired_cluster_bootstrap(ruido, base, clusters, n_boot=300, seed=1)
    assert a["ic_inf"] != paired_cluster_bootstrap(ruido, base, clusters, n_boot=300, seed=2)["ic_inf"]


def test_bootstrap_de_estatistica_nao_aditiva():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(30), 10)
    y = rng.integers(0, 2, 300)
    score = y + rng.normal(0, 1, 300)
    r = cluster_bootstrap_statistic(y, score, clusters, roc_auc_score, n_boot=200)
    assert r["valor"] == pytest.approx(roc_auc_score(y, score))
    assert r["ic_inf"] <= r["valor"] <= r["ic_sup"]
    assert 0 < r["n_boot"] <= 200
    assert r == cluster_bootstrap_statistic(y, score, clusters, roc_auc_score, n_boot=200)


# ----------------------------------------------------------------------------
# modeling: splits e travas
# ----------------------------------------------------------------------------

def _splits(mudancas: dict | None = None) -> pd.DataFrame:
    linhas = [("train", "g1", "2024-01-01"), ("train", "g2", "2024-01-02"), ("val", "g3", "2024-02-01"), ("test", "g4", "2024-03-01")]
    df = pd.DataFrame(linhas, columns=["split", "GAME_ID", "GAME_DATE"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    for idx, (col, valor) in (mudancas or {}).items():
        df.loc[idx, col] = pd.Timestamp(valor) if col == "GAME_DATE" else valor
    return df


def test_splits_validos_passam():
    assert_disjoint_temporal(_splits())


def test_trava_de_splits_recusa_jogo_repetido():
    with pytest.raises(AssertionError, match="aparecem em split anterior"):
        assert_disjoint_temporal(_splits({2: ("GAME_ID", "g2")}))


def test_trava_de_splits_recusa_ordem_temporal_invertida():
    with pytest.raises(AssertionError, match="comeca antes do fim"):
        assert_disjoint_temporal(_splits({2: ("GAME_DATE", "2024-01-02")}))


def test_trava_de_splits_recusa_split_vazio_e_linhas_soltas():
    with pytest.raises(AssertionError, match="split test vazio"):
        assert_disjoint_temporal(_splits().query("split != 'test'"))
    solto = pd.concat([_splits(), pd.DataFrame({"split": ["unassigned"], "GAME_ID": ["g9"], "GAME_DATE": [pd.Timestamp("2025-01-01")]})])
    with pytest.raises(AssertionError, match="fora dos splits"):
        assert_disjoint_temporal(solto)


def test_get_split_recusa_nome_desconhecido():
    with pytest.raises(ValueError, match="split desconhecido"):
        get_split(_splits(), "validacao")
    assert len(get_split(_splits(), "train")) == 2


def test_dobras_do_cv_validam_na_temporada_seguinte():
    treino = pd.DataFrame({"SEASON": ["A"] * 2 + ["B"] * 3 + ["C"] * 4 + ["D"] * 5})
    dobras = season_folds(treino)
    assert len(dobras) == 3
    for k, (tr, va) in enumerate(dobras):
        esperada = ["B", "C", "D"][k]
        assert set(treino["SEASON"].iloc[va]) == {esperada}
        assert len(va) == [3, 4, 5][k]
        assert set(treino["SEASON"].iloc[tr]) == set("ABCD"[: k + 1])
        assert not set(tr) & set(va)


def test_limite_de_dias_de_descanso():
    X = pd.DataFrame({"rest_days": [1, 5, 10, 50, 130], "outra": [1, 2, 3, 4, 5]})
    antes = X.copy()
    cap = cap_rest_days(X)
    assert cap["rest_days"].tolist() == [1, 5, 10, REST_DAYS_CAP, REST_DAYS_CAP]
    pd.testing.assert_frame_equal(X, antes)  # sem efeito colateral
    sem_coluna = pd.DataFrame({"outra": [1]})
    assert cap_rest_days(sem_coluna) is sem_coluna
    matriz = np.array([[100.0]])
    assert cap_rest_days(matriz) is matriz


def test_features_relativas_a_linha_de_valor_conhecido():
    df = pd.DataFrame({
        "line_synthetic": [10.0, 10.5], "pts_ewm_prev": [12.0, 10.0], "pts_ma5_prev": [11.0, 10.0],
        "pts_season_mean_prev": [9.0, 10.0], "pts_std5_prev": [1.0, 0.0],
    })
    antes = df.copy()
    out = add_clf_features(df)
    pd.testing.assert_frame_equal(df, antes)
    assert out["diff_ewm_linha"].tolist() == [2.0, -0.5]
    assert out["diff_ma5_linha"].tolist() == [1.0, -0.5]
    assert out["diff_media_temporada_linha"].tolist() == [-1.0, -0.5]
    assert out["z_ewm_linha"].tolist() == [1.0, -0.5]  # (12-10)/(1+1) e (10-10,5)/(0+1)
    assert out["linha_inteira"].tolist() == [1.0, 0.0]


def test_conjuntos_de_features_da_classificacao():
    completo = clf_feature_set("completo")
    assert completo == PRE_GAME_FEATURES + CLF_EXTRA_FEATURES and len(set(completo)) == len(completo)
    assert "linha_inteira" not in clf_feature_set("sem_linha_inteira") and "line_synthetic" in clf_feature_set("sem_linha_inteira")
    assert clf_feature_set("so_pre_jogo") == PRE_GAME_FEATURES
    with pytest.raises(ValueError):
        clf_feature_set("inexistente")


def test_trava_de_colunas_pos_jogo_recusa_alvos_e_estatisticas_do_jogo():
    for coluna in ("PTS", "pts", "MIN", "y_pts", "y_over", "FGA", "plus_minus", "WL"):
        with pytest.raises(AssertionError, match="pos-jogo"):
            assert_no_post_game(PRE_GAME_FEATURES + [coluna])
    for variante in ("completo", "sem_linha_inteira", "so_pre_jogo"):
        assert_no_post_game(clf_feature_set(variante))


def test_baselines_de_valor_conhecido():
    n = 20
    ewm = np.arange(n, dtype=float)
    cols = {c: ewm for c in set(REG_BASELINES.values())}
    treino = pd.DataFrame({**cols, "y_pts": 2 * ewm + 1})
    teste = pd.DataFrame({**{c: np.array([5.0, 6.0]) for c in cols}})
    p = predict_baselines(teste, treino)
    assert set(p) == set(REG_BASELINES) | {"constante_media_treino", "constante_mediana_treino", "linear_sobre_exponencial"}
    assert all(len(v) == 2 for v in p.values())
    assert p["media_5_jogos"].tolist() == [5.0, 6.0]
    assert p["constante_media_treino"].tolist() == [treino["y_pts"].mean()] * 2
    assert p["constante_mediana_treino"].tolist() == [treino["y_pts"].median()] * 2
    assert p["linear_sobre_exponencial"] == pytest.approx([11.0, 13.0])  # recupera y = 2x + 1


# ----------------------------------------------------------------------------
# modeling: catalogo de modelos
# ----------------------------------------------------------------------------

def test_catalogo_de_regressao_e_de_classificacao():
    assert list(regression_zoo()) == ["ridge", "lasso", "poisson_glm", "random_forest", "hgb_quadratico", "hgb_poisson", "hgb_absoluto"]
    assert list(classification_zoo()) == ["logistica_l2", "logistica_l1", "random_forest", "hgb"]


def test_toda_grade_so_cita_hiperparametros_que_existem():
    for zoo in (regression_zoo(), classification_zoo()):
        for nome, (pipe, grade) in zoo.items():
            assert grade, nome
            for parametro, valores in grade.items():
                clone(pipe).set_params(**{parametro: valores[0]})  # AttributeError/ValueError se o nome estiver errado


def test_gradient_boosting_nao_usa_early_stopping_aleatorio():
    """O early stopping do HGB separaria uma validacao aleatoria e quebraria a disciplina temporal."""
    for zoo in (regression_zoo(), classification_zoo()):
        for nome, (pipe, _) in zoo.items():
            if "hgb" in nome:
                assert pipe.named_steps["model"].early_stopping is False, nome


def test_modelos_com_aleatoriedade_usam_a_semente_do_projeto():
    """So os que sorteiam algo: floresta, boosting e a logistica L1 (saga). Ridge e Lasso sao deterministicos."""
    estocasticos = {"random_forest", "hgb_quadratico", "hgb_poisson", "hgb_absoluto", "logistica_l1", "hgb"}
    verificados = []
    for zoo in (regression_zoo(), classification_zoo()):
        for nome, (pipe, _) in zoo.items():
            if nome in estocasticos:
                assert pipe.named_steps["model"].get_params()["random_state"] == SEED, nome
                verificados.append(nome)
    assert len(verificados) == 6 + 1  # random_forest aparece nas duas familias


def test_so_modelos_lineares_padronizam():
    for nome, (pipe, _) in regression_zoo().items():
        assert ("scaler" in pipe.named_steps) == (nome in {"ridge", "lasso", "poisson_glm"}), nome


@pytest.fixture(scope="module")
def amostra():
    df = load_dataset()
    treino = df[df["split"] == "train"].sample(1500, random_state=0)
    teste = df[df["split"] == "val"].sample(300, random_state=0)
    return add_clf_features(treino), add_clf_features(teste)


def test_todos_os_modelos_de_regressao_treinam_e_preveem(amostra):
    treino, teste = amostra
    for nome, (pipe, _) in regression_zoo().items():
        modelo = clone(pipe).fit(treino[PRE_GAME_FEATURES], treino["y_pts"])
        p = modelo.predict(teste[PRE_GAME_FEATURES])
        assert p.shape == (300,) and np.isfinite(p).all(), nome
        if nome == "poisson_glm":
            assert (p >= 0).all()
        assert 2 < p.mean() < 25, nome  # ordem de grandeza dos pontos por jogo


def test_todos_os_classificadores_treinam_e_dao_probabilidades(amostra):
    treino, teste = amostra
    feats = clf_feature_set("completo")
    for nome, (pipe, _) in classification_zoo().items():
        modelo = clone(pipe).fit(treino[feats], treino["y_over"].astype(int))
        proba = modelo.predict_proba(teste[feats])
        assert proba.shape == (300, 2) and np.allclose(proba.sum(axis=1), 1.0), nome
        assert ((proba >= 0) & (proba <= 1)).all(), nome


# ----------------------------------------------------------------------------
# uncertainty
# ----------------------------------------------------------------------------

def test_intervalo_conformal_de_valor_conhecido():
    banda = ConformalBand(alpha=0.2, q=2.0, n_calibracao=10)
    lo, hi = banda.predict([16.0, 0.25])
    assert lo[0] == pytest.approx(16 - 2 * 4) and hi[0] == pytest.approx(16 + 2 * 4)  # sqrt(16) = 4
    assert lo[1] == 0.0 and hi[1] == pytest.approx(0.25 + 2 * math.sqrt(MIN_SCALE))  # previsao < 1 usa escala 1, e o piso e zero


def test_intervalo_nunca_fica_abaixo_de_zero_e_cresce_com_a_previsao():
    banda = ConformalBand(alpha=0.1, q=3.0, n_calibracao=100)
    mu = np.linspace(0, 40, 200)
    lo, hi = banda.predict(mu)
    assert (lo >= 0).all() and (hi >= mu).all()
    assert (np.diff(hi - lo) >= -1e-12).all()


def test_banda_serializa_e_recupera_igual():
    banda = ConformalBand.fit([1, 2, 3, 4, 5, 6], [2, 2, 3, 3, 4, 4], alpha=0.2)
    assert ConformalBand.from_dict(json.loads(banda.to_json())) == banda


def test_conformal_cobre_ao_menos_o_nominal_na_propria_calibracao():
    rng = np.random.default_rng(0)
    mu = rng.uniform(2, 30, 400)
    y = rng.poisson(mu).astype(float)
    qs = {}
    for alpha in (0.2, 0.1):
        banda = ConformalBand.fit(y, mu, alpha)
        lo, hi = banda.predict(mu)
        assert np.mean((y >= lo) & (y <= hi)) >= 1 - alpha
        qs[alpha] = banda.q
    assert qs[0.1] >= qs[0.2]  # mais confianca, intervalo mais largo


def test_relatorio_de_cobertura_com_valores_conhecidos():
    r = coverage_report([1, 5, 10], lo=np.array([0, 0, 0]), hi=np.array([2, 4, 10]))
    assert r["n"] == 3 and r["cobertura"] == pytest.approx(2 / 3)  # 5 > 4 fica de fora; 10 == 10 entra
    assert r["largura_media"] == pytest.approx(16 / 3) and r["largura_mediana"] == 4.0


def test_cobertura_por_grupo():
    r = coverage_by_group([1, 5, 10], np.array([0, 0, 0]), np.array([2, 4, 10]), pd.Series(["a", "a", "b"]))
    assert r.loc["a", "cobertura"] == 0.5 and r.loc["b", "cobertura"] == 1.0 and r.loc["a", "n"] == 2


def test_oraculo_da_temporada_exclui_o_proprio_jogo():
    df = pd.DataFrame({"PLAYER_ID": [1, 1, 1, 2], "SEASON": "S", "y_pts": [10.0, 20.0, 30.0, 7.0]})
    assert oracle_season_loo(df).tolist() == [25.0, 20.0, 15.0, 7.0]  # o jogador de um jogo so cai no proprio valor


def test_oraculo_local_respeita_a_ordem_original_do_dataframe():
    df = pd.DataFrame({
        "PLAYER_ID": 1, "SEASON": "S", "y_pts": [10.0, 20.0, 30.0, 40.0],
        "GAME_DATE": pd.date_range("2024-01-01", periods=4),
    }).sample(frac=1, random_state=0)  # embaralhado: o resultado tem de seguir a ordem de entrada
    esperado = {10.0: 20.0, 20.0: 20.0, 30.0: 30.0, 40.0: 30.0}
    assert oracle_local(df, half_window=1).tolist() == [esperado[v] for v in df["y_pts"]]


def test_oraculo_em_mediana_de_valor_conhecido():
    um_grupo = pd.DataFrame({"PLAYER_ID": 1, "SEASON": "S", "y_pts": [10.0, 20.0, 30.0], "GAME_DATE": pd.date_range("2024-01-01", periods=3)})
    assert oracle_median(um_grupo).tolist() == [25.0, 20.0, 15.0]
    quatro = pd.DataFrame({"PLAYER_ID": 1, "SEASON": "S", "y_pts": [10.0, 20.0, 30.0, 40.0], "GAME_DATE": pd.date_range("2024-01-01", periods=4)})
    assert oracle_median(quatro, half_window=1).tolist() == [20.0, 20.0, 30.0, 30.0]
