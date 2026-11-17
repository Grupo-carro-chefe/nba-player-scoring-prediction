# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_artefatos.py
# SINTESE: Métricas do JSON recalculadas das previsões e travas do protocolo de teste único.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: métricas recalculadas e travas do protocolo.
# === FIM DO CABECALHO ===

"""Integridade dos resultados versionados e das travas do protocolo.

test_resultados.py confere que README e relatorio CITAM os numeros de
test_final.json. Aqui se confere que os numeros do JSON estao CERTOS: cada
metrica e recalculada, de forma independente, a partir das previsoes salvas em
results/predictions_*.parquet e dos dados, sem passar por train.py/evaluate.py.
"""

from __future__ import annotations

import json
import math
import re
import struct
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import evaluate, train
from src.metrics import classification_metrics, regression_metrics
from src.modeling import load_dataset, regression_zoo
from src.uncertainty import ConformalBand

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURAS = ROOT / "results" / "figures"
PNG = b"\x89PNG\r\n\x1a\n"


def _precisa(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} ainda nao gerado")
    return path


@pytest.fixture(scope="module")
def final() -> dict:
    return json.loads(_precisa(METRICS / "test_final.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tv() -> dict:
    return json.loads(_precisa(METRICS / "treino_validacao.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def p_teste() -> pd.DataFrame:
    return pd.read_parquet(_precisa(ROOT / "results" / "predictions_test.parquet"))


@pytest.fixture(scope="module")
def p_val() -> pd.DataFrame:
    return pd.read_parquet(_precisa(ROOT / "results" / "predictions_val.parquet"))


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return load_dataset()


def _chaves(df: pd.DataFrame) -> set:
    return set(zip(df["PLAYER_ID"], df["GAME_ID"]))


# ----------------------------------------------------------------------------
# previsoes salvas x dataset
# ----------------------------------------------------------------------------

def test_previsoes_do_teste_cobrem_exatamente_o_split_de_teste(final, p_teste, dataset):
    teste = dataset[dataset["split"] == "test"]
    assert len(p_teste) == len(teste) == final["linhas_teste"]
    assert _chaves(p_teste) == _chaves(teste)
    junto = p_teste.merge(teste[["PLAYER_ID", "GAME_ID", "y_pts", "y_over"]], on=["PLAYER_ID", "GAME_ID"], suffixes=("", "_ds"))
    assert (junto["y_pts"] == junto["y_pts_ds"]).all() and (junto["y_over"] == junto["y_over_ds"]).all()
    assert set(p_teste["SEASON"]) == {"2025-26"}


def test_previsoes_da_validacao_cobrem_exatamente_o_split_de_validacao(tv, p_val, dataset):
    val = dataset[dataset["split"] == "val"]
    assert len(p_val) == len(val) == tv["linhas"]["val"]
    assert _chaves(p_val) == _chaves(val)
    assert set(p_val["SEASON"]) == {"2024-25"}


def test_validacao_nao_traz_colunas_do_teste(p_val, p_teste):
    """Oraculos e intervalos so existem no teste: a validacao ficou livre deles."""
    assert not [c for c in p_val.columns if c.startswith(("oraculo", "intervalo", "modelo_oraculo"))]
    assert {"oraculo_temporada_media", "intervalo80_inf", "modelo_oraculo_nivel"} <= set(p_teste.columns)


def test_validacao_vem_antes_do_teste_no_tempo(p_val, p_teste):
    assert p_val["GAME_DATE"].max() < p_teste["GAME_DATE"].min()


# ----------------------------------------------------------------------------
# metricas do JSON recalculadas das previsoes
# ----------------------------------------------------------------------------

def test_metricas_de_regressao_do_json_sao_recalculaveis(final, p_teste):
    y = p_teste["y_pts"]
    for nome, m in final["regressao"]["teste"].items():
        novo = regression_metrics(y, p_teste[f"reg__{nome}"])
        for k in ("mae", "rmse", "r2", "vies"):
            assert m[k] == pytest.approx(novo[k], rel=1e-9, abs=1e-12), f"{nome}.{k}"
        assert m["n"] == len(p_teste)


def test_metricas_de_classificacao_do_json_sao_recalculaveis(final, tv, p_teste):
    base = tv["classificacao"]["taxa_positiva_treino"]
    y = p_teste["y_over"].astype(int).to_numpy()
    sem_empate = (p_teste["y_pts"] != p_teste["line_synthetic"]).to_numpy()
    for nome, m in final["classificacao"]["teste"].items():
        proba = p_teste[f"clf__{nome}"].to_numpy()
        novo = classification_metrics(y, proba, base)
        for k in ("acuracia", "auc", "brier", "brier_ingenuo", "brier_skill", "log_loss", "ece"):
            assert m[k] == pytest.approx(novo[k], rel=1e-9, abs=1e-12), f"{nome}.{k}"
        sem = classification_metrics(y[sem_empate], proba[sem_empate], base)
        assert final["classificacao"]["teste_sem_empates"][nome]["auc"] == pytest.approx(sem["auc"], rel=1e-9)


def test_diferencas_do_bootstrap_sao_recalculaveis_e_os_ic_as_contem(final, p_teste):
    y = p_teste["y_pts"]
    n_jogadores = p_teste["PLAYER_ID"].nunique()
    comparacoes = final["regressao"]["comparacoes_bootstrap"]
    assert len(comparacoes) >= 30
    for chave, v in comparacoes.items():
        a, b = chave.split("_vs_")
        delta = float(np.mean(np.abs(y - p_teste[f"reg__{a}"])) - np.mean(np.abs(y - p_teste[f"reg__{b}"])))
        assert v["delta"] == pytest.approx(delta, rel=1e-9, abs=1e-12), chave
        assert v["ic_inf"] <= v["delta"] <= v["ic_sup"], chave
        assert 0.0 <= v["p_valor"] <= 1.0 and v["n_clusters"] == n_jogadores, chave


def test_cobertura_dos_intervalos_recalculada_e_proxima_do_nominal(final, p_teste):
    y = p_teste["y_pts"]
    for nivel in ("80", "90"):
        lo, hi = p_teste[f"intervalo{nivel}_inf"], p_teste[f"intervalo{nivel}_sup"]
        cobertura = float(((y >= lo) & (y <= hi)).mean())
        informada = final["regressao"]["intervalos"][nivel]["global"]["cobertura"]
        assert informada == pytest.approx(cobertura, abs=1e-12)
        assert abs(cobertura - int(nivel) / 100) < 0.02, f"cobertura de {nivel}% = {cobertura:.3f}"
        assert (lo >= 0).all() and (lo <= hi).all()
    assert (p_teste["intervalo90_inf"] <= p_teste["intervalo80_inf"]).all()
    assert (p_teste["intervalo80_sup"] <= p_teste["intervalo90_sup"]).all()


def test_quantil_conformal_foi_calibrado_so_na_validacao(final, tv, p_val):
    modelo = tv["regressao"]["modelo_final"]
    conformal = json.loads((ROOT / "results" / "models" / "conformal.json").read_text(encoding="utf-8"))
    assert conformal == tv["regressao"]["conformal"]
    for nivel, alpha in (("80", 0.2), ("90", 0.1)):
        banda = ConformalBand.fit(p_val["y_pts"], p_val[f"reg__{modelo}"], alpha)
        assert banda.q == pytest.approx(conformal[nivel]["q"], rel=1e-12)
        assert conformal[nivel]["n_calibracao"] == len(p_val) == banda.n_calibracao
        assert final["regressao"]["intervalos"][nivel]["q"] == pytest.approx(conformal[nivel]["q"], rel=1e-12)


def test_modelo_final_e_o_mesmo_na_validacao_e_no_teste(final, tv):
    assert final["regressao"]["modelo_final"] == tv["regressao"]["modelo_final"]
    assert final["classificacao"]["modelo_final"] == tv["classificacao"]["modelo_final"]
    assert final["regressao"]["modelo_final"] in regression_zoo()
    assert not final["regressao"]["modelo_final"].startswith("oraculo")


def test_modelo_de_classificacao_foi_escolhido_pelo_menor_log_loss_na_validacao(tv):
    val = {k: v["log_loss"] for k, v in tv["classificacao"]["validacao"].items() if v["tipo"] == "modelo"}
    assert tv["classificacao"]["modelo_final"] == min(val, key=val.get)


def test_resultados_sao_numeros_finitos(final, tv):
    def percorrer(no, caminho=""):
        if isinstance(no, dict):
            for k, v in no.items():
                yield from percorrer(v, f"{caminho}/{k}")
        elif isinstance(no, list):
            for i, v in enumerate(no):
                yield from percorrer(v, f"{caminho}[{i}]")
        elif isinstance(no, float) and not math.isfinite(no):
            yield caminho

    nao_finitos = [c for doc in (final, tv) for c in percorrer(doc)]
    assert not nao_finitos, f"NaN/inf nos resultados: {nao_finitos[:5]}"


# ----------------------------------------------------------------------------
# modelos e figuras versionados
# ----------------------------------------------------------------------------

def test_modelos_finais_carregam_e_reproduzem_uma_previsao(dataset, p_teste):
    reg = joblib.load(ROOT / "results" / "models" / "regressao_final.joblib")
    clf = joblib.load(ROOT / "results" / "models" / "classificacao_final.joblib")
    from src.modeling import add_clf_features, clf_feature_set
    from src.features import PRE_GAME_FEATURES

    amostra = add_clf_features(dataset[dataset["split"] == "test"].head(50))
    p = reg.predict(amostra[PRE_GAME_FEATURES])
    junto = amostra[["PLAYER_ID", "GAME_ID"]].assign(p=p).merge(p_teste, on=["PLAYER_ID", "GAME_ID"])
    assert np.allclose(junto["p"], junto["reg__hgb_absoluto"], atol=1e-6)
    proba = clf.predict_proba(amostra[clf_feature_set("completo")])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_figuras_sao_pngs_validos_e_nao_triviais():
    figuras = sorted(FIGURAS.glob("*.png"))
    assert len(figuras) == 9
    assert "f_previsto_vs_observado_marginais.png" in {f.name for f in figuras}
    for f in figuras:
        cabeca = f.read_bytes()[:24]
        assert cabeca[:8] == PNG, f.name
        largura, altura = struct.unpack(">II", cabeca[16:24])
        assert largura >= 600 and altura >= 300, f"{f.name}: {largura}x{altura}"
        assert f.stat().st_size > 10_000, f.name


# ----------------------------------------------------------------------------
# travas do protocolo (nao pode treinar de novo nem reavaliar o teste sem querer)
# ----------------------------------------------------------------------------

def test_hash_do_codigo_ignora_o_cabecalho_e_os_fins_de_linha(tmp_path, monkeypatch):
    codigo = "x = 1\nprint(x)\n"
    arquivo = tmp_path / "a.py"
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "CODIGO_DOS_RESULTADOS", ("a.py",))

    def hash_de(texto: str) -> str:
        arquivo.write_bytes(texto.encode("utf-8"))
        return train.hash_codigo()

    base = hash_de("# === CABECALHO ACADEMICO ===\n# historico v1\n# === FIM DO CABECALHO ===\n" + codigo)
    assert re.fullmatch(r"[0-9a-f]{64}", base)
    assert hash_de("# === CABECALHO ACADEMICO ===\n# historico v2, mais longo\n# === FIM DO CABECALHO ===\n" + codigo) == base
    assert hash_de(("# === FIM DO CABECALHO ===\n" + codigo).replace("\n", "\r\n")) == base
    assert hash_de("# === FIM DO CABECALHO ===\n" + codigo.replace("1", "2")) != base


def test_hash_do_codigo_real_e_estavel_e_cobre_os_arquivos_certos():
    assert train.hash_codigo() == train.hash_codigo()
    assert set(train.CODIGO_DOS_RESULTADOS) == {
        "src/features.py", "src/target.py", "src/modeling.py", "src/metrics.py",
        "src/uncertainty.py", "src/train.py", "src/evaluate.py",
    }
    assert all((ROOT / rel).exists() for rel in train.CODIGO_DOS_RESULTADOS)


def test_informacoes_do_ambiente_registram_o_que_reproduz_o_resultado():
    info = train.environment_info()
    assert {"gerado_em_utc", "git_commit", "hash_codigo", "python", "sklearn", "pandas", "numpy", "seed"} <= set(info)
    assert info["seed"] == 42 and info["hash_codigo"] == train.hash_codigo()
    assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"]) or info["git_commit"] == "desconhecido"


def test_treino_recusa_rodar_com_o_teste_ja_avaliado(tmp_path, monkeypatch):
    (tmp_path / "test_final.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(train, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(train, "train", lambda fast=False: pytest.fail("nao deveria treinar"))
    monkeypatch.setattr(sys, "argv", ["train"])
    with pytest.raises(SystemExit, match="ja existe"):
        train.main()
    monkeypatch.setattr(sys, "argv", ["train", "--force", "--fast"])
    with pytest.raises(SystemExit, match="--fast e so para depurar"):
        train.main()


def test_treino_com_force_segue_adiante(tmp_path, monkeypatch):
    (tmp_path / "test_final.json").write_text("{}", encoding="utf-8")
    chamadas = []
    monkeypatch.setattr(train, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(train, "train", lambda fast=False: chamadas.append(fast))
    monkeypatch.setattr(sys, "argv", ["train", "--force"])
    train.main()
    assert chamadas == [False]


def test_avaliacao_recusa_reavaliar_o_teste(tmp_path, monkeypatch):
    final = tmp_path / "test_final.json"
    final.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate, "FINAL_JSON", final)
    with pytest.raises(SystemExit, match="ja foi avaliado"):
        evaluate.evaluate(force=False)


def test_avaliacao_exige_o_treino_feito_antes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "FINAL_JSON", tmp_path / "test_final.json")
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="python -m src.train"):
        evaluate.evaluate(force=False)


def test_avaliacao_exige_a_confirmacao_de_escolhas_congeladas(monkeypatch):
    monkeypatch.setattr(evaluate, "evaluate", lambda force=False: pytest.fail("nao deveria avaliar"))
    monkeypatch.setattr(sys, "argv", ["evaluate"])
    with pytest.raises(SystemExit, match="--final"):
        evaluate.main()


def test_importancia_agrupada_distingue_familia_informativa_de_ruido():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"util": rng.normal(size=400), "util_irma": rng.normal(size=400), "ruido": rng.normal(size=400)})
    y = 3 * X["util"] + 3 * X["util_irma"] + rng.normal(0, 0.1, 400)
    modelo = LinearRegression().fit(X, y)
    r = evaluate.grouped_permutation_importance(
        modelo, X, y, {"informativa": ["util", "util_irma"], "irrelevante": ["ruido"]}, n_repeats=3
    )
    assert r["mae_base"] < 0.2
    assert r["familias"]["informativa"]["aumento_mae_medio"] > 1.0
    ruido = abs(r["familias"]["irrelevante"]["aumento_mae_medio"])
    assert ruido < 0.01 * r["familias"]["informativa"]["aumento_mae_medio"]  # coeficiente ~0: embaralhar quase nao muda nada
    assert r == evaluate.grouped_permutation_importance(
        modelo, X, y, {"informativa": ["util", "util_irma"], "irrelevante": ["ruido"]}, n_repeats=3
    )  # mesma semente, mesmo resultado


def test_escolha_de_calibracao_prefere_nao_calibrar_um_modelo_ja_calibrado():
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.random(2000)})
    y = (rng.random(2000) < X["x"]).astype(int).to_numpy()  # P(y=1|x)=x: o proprio x ja e a probabilidade

    class Calibrado(DummyClassifier):
        def predict_proba(self, X):
            p = X["x"].to_numpy()
            return np.column_stack([1 - p, p])

    modelo = Calibrado().fit(X, y)
    datas = pd.Series(pd.date_range("2025-01-01", periods=2000))
    escolha = train._choose_calibration(modelo, X, y, datas)
    assert set(escolha["brier_segunda_metade"]) == {"sem_calibracao", "sigmoid", "isotonic"}
    assert escolha["metodo"] in escolha["brier_segunda_metade"]
    menor = min(escolha["brier_segunda_metade"], key=escolha["brier_segunda_metade"].get)
    assert escolha["metodo"] == menor
    assert escolha["data_corte"] == str(datas.sort_values().iloc[1000].date())
