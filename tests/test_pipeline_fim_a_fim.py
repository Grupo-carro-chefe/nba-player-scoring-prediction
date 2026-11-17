# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_pipeline_fim_a_fim.py
# SINTESE: Treino, teste único e figuras com dados reduzidos; treino não lê o teste.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: pipeline reduzido de ponta a ponta.
# === FIM DO CABECALHO ===

"""Pipeline de ponta a ponta (treino, teste unico, figuras) com dados reduzidos.

Roda train -> evaluate -> figures do zero numa pasta temporaria, com uma amostra
do dataset e o modo --fast. Nao substitui os resultados versionados (que levam
minutos para sair): prova que o pipeline roda inteiro num clone limpo, que o
formato dos resultados e o mesmo do arquivo entregue e, principalmente, que o
treino NAO le o conjunto de teste.

Marcado como lento (dezenas de segundos): pule com `pytest -m "not slow"`.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import evaluate, figures, modeling, train
from src.features import PRE_GAME_FEATURES
from src.modeling import load_dataset
from src.uncertainty import ConformalBand

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
POR_TEMPORADA = 700
RUIDO = {"ambiente", "tempo_s", "tempo_total_s"}


def _amostra() -> pd.DataFrame:
    df = load_dataset()
    amostra = df.groupby("SEASON", group_keys=False).sample(POR_TEMPORADA, random_state=7)
    return amostra.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)


def _redirecionar(mp, raiz: Path, df: pd.DataFrame) -> None:
    """Faz train, evaluate e figures lerem `df` e gravarem em `raiz`, sem tocar em results/."""
    pastas = {"RESULTS": raiz, "METRICS_DIR": raiz / "metrics", "MODELS_DIR": raiz / "models",
              "CACHE_DIR": raiz / "cache", "FIG_DIR": raiz / "figures"}
    for modulo in (train, evaluate, figures):
        for nome, valor in pastas.items():
            if hasattr(modulo, nome):
                mp.setattr(modulo, nome, valor)
    mp.setattr(evaluate, "FINAL_JSON", raiz / "metrics" / "test_final.json")
    for modulo in (train, evaluate, modeling):
        mp.setattr(modulo, "load_dataset", lambda: df.copy())


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("pipeline")
    df = _amostra()
    with pytest.MonkeyPatch.context() as mp:
        _redirecionar(mp, raiz, df)
        tv = train.train(fast=True)
        final = evaluate.evaluate()
        figuras = figures.make_all()
    return SimpleNamespace(raiz=raiz, df=df, tv=tv, final=final, figuras=figuras)


def _sem_ruido(no):
    if isinstance(no, dict):
        return {k: _sem_ruido(v) for k, v in no.items() if k not in RUIDO}
    if isinstance(no, list):
        return [_sem_ruido(v) for v in no]
    return no


def _igual(a, b, caminho="") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{caminho}: chaves diferentes"
        for k in a:
            _igual(a[k], b[k], f"{caminho}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{caminho}: tamanhos diferentes"
        for i, (x, y) in enumerate(zip(a, b)):
            _igual(x, y, f"{caminho}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), f"{caminho}: {a} != {b}"
    else:
        assert a == b, f"{caminho}: {a!r} != {b!r}"


def test_pipeline_gera_todos_os_artefatos(pipeline):
    raiz = pipeline.raiz
    esperados = [
        "metrics/treino_validacao.json", "metrics/test_final.json", "predictions_val.parquet", "predictions_test.parquet",
        "models/regressao_final.joblib", "models/classificacao_final.joblib", "models/conformal.json",
    ]
    assert [e for e in esperados if not (raiz / e).exists()] == []
    assert len(pipeline.figuras) == 9
    assert "f_previsto_vs_observado_marginais.png" in pipeline.figuras
    assert (raiz / "cache").is_dir() and any((raiz / "cache").glob("reg__*.joblib"))


def test_amostra_usa_as_cinco_temporadas_e_os_tres_splits(pipeline):
    assert pipeline.df["SEASON"].nunique() == 5
    assert set(pipeline.df["split"]) == {"train", "val", "test"}
    assert pipeline.tv["linhas"] == {"train": 3 * POR_TEMPORADA, "val": POR_TEMPORADA}
    assert pipeline.final["linhas_teste"] == POR_TEMPORADA


def test_modo_rapido_pula_floresta_e_variantes_do_boosting(pipeline):
    assert set(pipeline.tv["regressao"]["cv"]) == {"ridge", "lasso", "poisson_glm", "hgb_quadratico"}
    assert set(pipeline.tv["classificacao"]["cv"]) == {"logistica_l2", "hgb"}


def test_modelo_final_e_o_de_menor_erro_na_validacao(pipeline):
    reg = pipeline.tv["regressao"]
    val = {k: v["mae"] for k, v in reg["validacao"].items() if v["tipo"] == "modelo"}
    assert reg["modelo_final"] == min(val, key=val.get)
    clf = pipeline.tv["classificacao"]
    ll = {k: v["log_loss"] for k, v in clf["validacao"].items() if v["tipo"] == "modelo"}
    assert clf["modelo_final"] == min(ll, key=ll.get)


def test_conformal_calibrado_na_validacao_cobre_ao_menos_o_nominal(pipeline):
    pv = pd.read_parquet(pipeline.raiz / "predictions_val.parquet")
    modelo = pipeline.tv["regressao"]["modelo_final"]
    conformal = json.loads((pipeline.raiz / "models" / "conformal.json").read_text(encoding="utf-8"))
    for nivel, alpha in (("80", 0.2), ("90", 0.1)):
        banda = ConformalBand.from_dict(conformal[nivel])
        lo, hi = banda.predict(pv[f"reg__{modelo}"])
        assert float(((pv["y_pts"] >= lo) & (pv["y_pts"] <= hi)).mean()) >= 1 - alpha
        assert banda.q == pytest.approx(ConformalBand.fit(pv["y_pts"], pv[f"reg__{modelo}"], alpha).q)


def test_avaliacao_gerou_previsoes_e_intervalos_para_todo_o_teste(pipeline):
    pt = pd.read_parquet(pipeline.raiz / "predictions_test.parquet")
    assert len(pt) == POR_TEMPORADA and set(pt["SEASON"]) == {"2025-26"}
    modelo = pipeline.final["regressao"]["modelo_final"]
    assert np.isfinite(pt[f"reg__{modelo}"]).all()
    assert (pt["intervalo80_inf"] <= pt["intervalo80_sup"]).all() and (pt["intervalo80_inf"] >= 0).all()
    assert {"oraculo_temporada_media", "clf__final_calibrado", "reg__mediana_sobre_exponencial"} <= set(pt.columns)


def test_formato_do_json_e_o_mesmo_do_arquivo_entregue(pipeline):
    real = json.loads((ROOT / "results" / "metrics" / "test_final.json").read_text(encoding="utf-8"))
    novo = pipeline.final
    assert set(novo) == set(real)
    variavel = {"coeficientes_padronizados"}  # so existe se o modelo final for linear
    for secao in ("regressao", "classificacao"):
        assert set(novo[secao]) - variavel == set(real[secao]) - variavel, secao
    assert set(novo["regressao"]["intervalos"]["80"]) == set(real["regressao"]["intervalos"]["80"])
    assert set(novo["regressao"]["intervalos"]["80"]["global"]) == set(real["regressao"]["intervalos"]["80"]["global"])
    assert set(novo["classificacao"]["teste"]["final_calibrado"]) == set(real["classificacao"]["teste"]["final_calibrado"])
    um_novo = next(iter(novo["regressao"]["comparacoes_bootstrap"].values()))
    um_real = next(iter(real["regressao"]["comparacoes_bootstrap"].values()))
    assert set(um_novo) == set(um_real)
    for secao in ("regressao", "classificacao"):
        assert set(pipeline.tv[secao]) == set(json.loads((ROOT / "results" / "metrics" / "treino_validacao.json").read_text(encoding="utf-8"))[secao])


def test_avaliacao_final_se_recusa_a_rodar_de_novo(pipeline, monkeypatch):
    _redirecionar(monkeypatch, pipeline.raiz, pipeline.df)
    with pytest.raises(SystemExit, match="ja foi avaliado"):
        evaluate.evaluate()


def test_figuras_geradas_sao_pngs_validos(pipeline):
    for nome in pipeline.figuras:
        cabeca = (pipeline.raiz / "figures" / nome).read_bytes()[:24]
        assert cabeca[:8] == b"\x89PNG\r\n\x1a\n", nome
        largura, altura = struct.unpack(">II", cabeca[16:24])
        assert largura >= 600 and altura >= 300, nome


def test_o_treino_nao_depende_do_conjunto_de_teste(pipeline, tmp_path, monkeypatch):
    """Embaralha alvos e destroi as features do teste: treino e validacao tem de sair identicos."""
    envenenado = pipeline.df.copy()
    teste = (envenenado["split"] == "test").to_numpy()
    rng = np.random.default_rng(0)
    for alvo in ("y_pts", "y_over"):
        envenenado.loc[teste, alvo] = rng.permutation(envenenado.loc[teste, alvo].to_numpy())
    envenenado.loc[teste, PRE_GAME_FEATURES + ["line_synthetic"]] = -1e6

    _redirecionar(monkeypatch, tmp_path, envenenado)
    tv2 = train.train(fast=True)

    _igual(_sem_ruido(tv2), _sem_ruido(pipeline.tv))
    a = pd.read_parquet(pipeline.raiz / "predictions_val.parquet")
    b = pd.read_parquet(tmp_path / "predictions_val.parquet")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)
