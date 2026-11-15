# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/train.py
# SINTESE: Treino com validação cruzada temporal e escolhas feitas na validação.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: CV temporal, seleção, calibração e conformal.
#   2026-11-15 | Rodrigo    | Hash do código nos resultados; JSON com fim de linha LF.
#   2026-11-15 | Rodrigo    | Trava: não retreina com o teste já avaliado sem --force.
# === FIM DO CABECALHO ===

"""Treino, selecao de hiperparametros e escolhas feitas na validacao.

    python -m src.train

Protocolo (nada aqui le o conjunto de teste):

1. Hiperparametros: busca em grade com validacao cruzada TEMPORAL dentro do
   treino (2021-22 -> 2022-23; 2021-23 -> 2023-24). Melhor configuracao de cada
   familia e reajustada no treino inteiro.
2. Validacao (2024-25): compara familias e baselines, escolhe o modelo final de
   regressao e o de classificacao, escolhe e ajusta o calibrador de
   probabilidade e calibra o intervalo conformal.
3. O teste (2025-26) so e lido por `python -m src.evaluate --final`, uma vez.

Saidas:
  results/metrics/treino_validacao.json   escolhas, CV e metricas de validacao
  results/predictions_val.parquet         previsoes de todos os modelos na validacao
  results/models/                         modelos finais (pequenos, versionados)
  results/cache/                          todos os modelos ajustados (nao versionado)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.features import PRE_GAME_FEATURES
from src.metrics import classification_metrics, regression_metrics
from src.modeling import (
    LINE,
    ROOT,
    SEED,
    TARGET_CLF,
    TARGET_REG,
    add_clf_features,
    assert_disjoint_temporal,
    assert_no_post_game,
    classification_zoo,
    clf_feature_set,
    get_split,
    load_dataset,
    predict_baselines,
    regression_zoo,
    season_folds,
)
from src.uncertainty import ConformalBand

RESULTS = ROOT / "results"
METRICS_DIR = RESULTS / "metrics"
MODELS_DIR = RESULTS / "models"
CACHE_DIR = RESULTS / "cache"
FIG_DIR = RESULTS / "figures"

CONFORMAL_ALPHAS = (0.2, 0.1)  # intervalos de 80% e 90%
HGB_SEARCH_ITER = 12           # sorteios por funcao de perda do HGB
ID_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "GAME_ID", "GAME_DATE", "SEASON"]


def git_commit() -> str:
    """Hash do commit atual; le .git direto se o executavel git nao existir."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_file = ROOT / ".git" / ref
            if ref_file.exists():
                return ref_file.read_text().strip()
            packed = (ROOT / ".git" / "packed-refs").read_text().splitlines()
            return next(line.split()[0] for line in packed if line.endswith(ref))
        return head
    except Exception:
        return "desconhecido"


# Arquivos cujo conteudo determina os resultados. O hash deles fica gravado nos
# JSONs; tests/test_resultados.py falha se o codigo mudar sem que os resultados
# sejam regenerados (o hash do commit sozinho nao detecta arquivos alterados e
# ainda nao commitados).
CODIGO_DOS_RESULTADOS = (
    "src/features.py", "src/target.py", "src/modeling.py", "src/metrics.py",
    "src/uncertainty.py", "src/train.py", "src/evaluate.py",
)


def hash_codigo() -> str:
    """SHA-256 do codigo que gera os resultados, ignorando o cabecalho academico.

    O cabecalho muda a cada entrada de historico sem alterar o comportamento;
    por isso ele e removido antes do hash. Fins de linha sao normalizados.
    """
    import hashlib

    h = hashlib.sha256()
    for rel in CODIGO_DOS_RESULTADOS:
        texto = (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        fim = texto.find("# === FIM DO CABECALHO ===")
        if fim != -1:
            texto = texto[fim + len("# === FIM DO CABECALHO ==="):]
        h.update(rel.encode() + b"\0" + texto.strip().encode("utf-8") + b"\0")
    return h.hexdigest()


def environment_info() -> dict:
    commit = git_commit()
    return {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit or "desconhecido",
        "hash_codigo": hash_codigo(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "seed": SEED,
    }


def _search(name: str, pipe, grid: dict, folds, scoring: str):
    """Grade completa para modelos baratos; sorteio para o HGB."""
    # HGB e RF ja usam todos os nucleos internamente; paralelizar a busca por
    # cima disso so disputaria CPU.
    n_jobs = 1 if ("hgb" in name or "forest" in name) else -1
    n_comb = int(np.prod([len(v) for v in grid.values()]))
    if "hgb" in name and n_comb > HGB_SEARCH_ITER:
        return RandomizedSearchCV(
            pipe, grid, n_iter=HGB_SEARCH_ITER, cv=folds, scoring=scoring,
            n_jobs=n_jobs, random_state=SEED, refit=True,
        )
    return GridSearchCV(pipe, grid, cv=folds, scoring=scoring, n_jobs=n_jobs, refit=True)


def _cv_summary(search, n_folds: int) -> dict:
    res = search.cv_results_
    best = search.best_index_
    return {
        "melhores_parametros": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "score_cv_medio": float(res["mean_test_score"][best]),
        "score_cv_por_dobra": [float(res[f"split{k}_test_score"][best]) for k in range(n_folds)],
        "combinacoes_testadas": int(len(res["params"])),
    }


def _choose_calibration(model, X_val: pd.DataFrame, y_val: np.ndarray, dates: pd.Series) -> dict:
    """Escolhe entre sem calibracao, sigmoid (Platt) e isotonica SEM usar o teste.

    Ajusta cada calibrador na primeira metade (no tempo) da validacao e compara o
    Brier na segunda metade. O metodo vencedor e reajustado na validacao inteira.
    """
    corte = dates.sort_values().iloc[len(dates) // 2]
    a = (dates < corte).to_numpy()
    b = ~a
    raw_b = model.predict_proba(X_val[b])[:, 1]
    briers = {"sem_calibracao": float(brier_score_loss(y_val[b], raw_b))}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val[a], y_val[a])
        briers[method] = float(brier_score_loss(y_val[b], cal.predict_proba(X_val[b])[:, 1]))
    escolhido = min(briers, key=briers.get)
    return {"brier_segunda_metade": briers, "metodo": escolhido, "data_corte": str(corte.date())}


def train(fast: bool = False) -> dict:
    t0 = time.time()
    for d in (METRICS_DIR, MODELS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = add_clf_features(load_dataset())
    assert_disjoint_temporal(df)
    train_df, val_df = get_split(df, "train"), get_split(df, "val")
    folds = season_folds(train_df)
    out: dict = {"ambiente": environment_info(), "linhas": {"train": len(train_df), "val": len(val_df)}}

    # ------------------------------------------------------------------ regressao
    X_tr, y_tr = train_df[PRE_GAME_FEATURES], train_df[TARGET_REG].to_numpy()
    X_va, y_va = val_df[PRE_GAME_FEATURES], val_df[TARGET_REG].to_numpy()
    assert_no_post_game(PRE_GAME_FEATURES)

    preds_val = val_df[ID_COLS + ["min_ma5_prev", "pts_season_mean_prev", LINE, TARGET_REG, TARGET_CLF]].copy()
    reg = {"cv": {}, "validacao": {}, "tempo_s": {}}

    for nome, p in predict_baselines(val_df, train_df).items():
        preds_val[f"reg__{nome}"] = p
        reg["validacao"][nome] = regression_metrics(y_va, p) | {"tipo": "baseline"}

    for nome, (pipe, grid) in regression_zoo().items():
        if fast and nome in {"random_forest", "hgb_poisson", "hgb_absoluto"}:
            continue
        t = time.time()
        search = _search(nome, pipe, grid, folds, "neg_mean_absolute_error").fit(X_tr, y_tr)
        reg["tempo_s"][nome] = round(time.time() - t, 1)
        reg["cv"][nome] = _cv_summary(search, len(folds))
        model = search.best_estimator_
        p = model.predict(X_va)
        preds_val[f"reg__{nome}"] = p
        reg["validacao"][nome] = regression_metrics(y_va, p) | {"tipo": "modelo"}
        joblib.dump(model, CACHE_DIR / f"reg__{nome}.joblib", compress=3)
        print(f"[reg] {nome:16s} CV MAE {-reg['cv'][nome]['score_cv_medio']:.3f}  val MAE {reg['validacao'][nome]['mae']:.3f}  ({reg['tempo_s'][nome]}s)")

    modelos_reg = [k for k, v in reg["validacao"].items() if v["tipo"] == "modelo"]
    final_reg = min(modelos_reg, key=lambda k: reg["validacao"][k]["mae"])
    reg["modelo_final"] = final_reg
    print(f"[reg] modelo final (menor MAE na validacao): {final_reg}")

    # Intervalo conformal calibrado nos residuos de validacao do modelo final.
    mu_val = preds_val[f"reg__{final_reg}"].to_numpy()
    bands = {f"{int((1 - a) * 100)}": ConformalBand.fit(y_va, mu_val, a) for a in CONFORMAL_ALPHAS}
    reg["conformal"] = {k: vars(b) for k, b in bands.items()}
    (MODELS_DIR / "conformal.json").write_text(json.dumps(reg["conformal"], indent=2), encoding="utf-8", newline="\n")
    joblib.dump(joblib.load(CACHE_DIR / f"reg__{final_reg}.joblib"), MODELS_DIR / "regressao_final.joblib", compress=3)
    out["regressao"] = reg

    # -------------------------------------------------------------- classificacao
    feats = clf_feature_set("completo")
    assert_no_post_game(feats)
    yc_tr = train_df[TARGET_CLF].astype(int).to_numpy()
    yc_va = val_df[TARGET_CLF].astype(int).to_numpy()
    base_rate = float(yc_tr.mean())
    clf = {"cv": {}, "validacao": {}, "tempo_s": {}, "taxa_positiva_treino": base_rate}

    preds_val["clf__constante_treino"] = base_rate
    clf["validacao"]["constante_treino"] = classification_metrics(yc_va, np.full(len(yc_va), base_rate), base_rate) | {"tipo": "baseline"}

    best_params: dict = {}
    for nome, (pipe, grid) in classification_zoo().items():
        if fast and nome in {"random_forest", "logistica_l1"}:
            continue
        t = time.time()
        search = _search(nome, pipe, grid, folds, "neg_log_loss").fit(train_df[feats], yc_tr)
        clf["tempo_s"][nome] = round(time.time() - t, 1)
        clf["cv"][nome] = _cv_summary(search, len(folds))
        best_params[nome] = search.best_params_
        model = search.best_estimator_
        proba = model.predict_proba(val_df[feats])[:, 1]
        preds_val[f"clf__{nome}"] = proba
        clf["validacao"][nome] = classification_metrics(yc_va, proba, base_rate) | {"tipo": "modelo"}
        joblib.dump(model, CACHE_DIR / f"clf__{nome}.joblib", compress=3)
        print(f"[clf] {nome:16s} val logloss {clf['validacao'][nome]['log_loss']:.4f}  AUC {clf['validacao'][nome]['auc']:.3f}  ({clf['tempo_s'][nome]}s)")

    modelos_clf = [k for k, v in clf["validacao"].items() if v["tipo"] == "modelo"]
    final_clf = min(modelos_clf, key=lambda k: clf["validacao"][k]["log_loss"])
    clf["modelo_final"] = final_clf
    print(f"[clf] modelo final (menor log loss na validacao): {final_clf}")

    # Ablacao de features da classificacao com os hiperparametros vencedores.
    zoo = classification_zoo()
    clf["ablacao_validacao"] = {}
    for variant in ("sem_linha_inteira", "so_pre_jogo"):
        fv = clf_feature_set(variant)
        m = clone(zoo[final_clf][0]).set_params(**best_params[final_clf]).fit(train_df[fv], yc_tr)
        proba = m.predict_proba(val_df[fv])[:, 1]
        clf["ablacao_validacao"][variant] = classification_metrics(yc_va, proba, base_rate)
        joblib.dump(m, CACHE_DIR / f"clf_ablacao__{variant}.joblib", compress=3)

    # Calibracao: escolhida na validacao (metade 1 ajusta, metade 2 julga).
    final_model = joblib.load(CACHE_DIR / f"clf__{final_clf}.joblib")
    cal_choice = _choose_calibration(final_model, val_df[feats], yc_va, val_df["GAME_DATE"])
    if cal_choice["metodo"] == "sem_calibracao":
        calibrated = final_model
    else:
        calibrated = CalibratedClassifierCV(FrozenEstimator(final_model), method=cal_choice["metodo"]).fit(val_df[feats], yc_va)
    clf["calibracao"] = cal_choice
    joblib.dump(calibrated, MODELS_DIR / "classificacao_final.joblib", compress=3)
    print(f"[clf] calibracao escolhida: {cal_choice['metodo']}  {cal_choice['brier_segunda_metade']}")
    out["classificacao"] = clf

    out["features"] = {"regressao": PRE_GAME_FEATURES, "classificacao": feats}
    out["tempo_total_s"] = round(time.time() - t0, 1)

    preds_val.to_parquet(RESULTS / "predictions_val.parquet", index=False)
    (METRICS_DIR / "treino_validacao.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8", newline="\n"
    )
    print(f"ok: {METRICS_DIR / 'treino_validacao.json'} ({out['tempo_total_s']}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Treina e seleciona modelos (nao toca o teste).")
    ap.add_argument("--fast", action="store_true", help="pula RF e variantes do HGB (para depurar)")
    ap.add_argument("--force", action="store_true",
                    help="retreina mesmo com o teste ja avaliado (rode depois src.evaluate --final --force)")
    args = ap.parse_args()
    # Trava: retreinar sobrescreve modelos, quantis e treino_validacao.json. Com
    # o teste ja avaliado, isso deixaria test_final.json descrevendo artefatos
    # que nao existem mais. O caminho seguro e scripts/run_all.py --retrain.
    final = METRICS_DIR / "test_final.json"
    if final.exists() and not args.force:
        raise SystemExit(
            f"{final} ja existe: retreinar agora deixaria o teste inconsistente com os modelos. "
            "Use python scripts/run_all.py --retrain (treina e reavalia juntos) ou, manualmente, "
            "python -m src.train --force seguido de python -m src.evaluate --final --force."
        )
    if args.fast and final.exists():
        raise SystemExit("--fast e so para depurar antes da avaliacao: grava nos mesmos arquivos dos modelos finais.")
    train(fast=args.fast)


if __name__ == "__main__":
    main()
