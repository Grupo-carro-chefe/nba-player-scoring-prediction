# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/evaluate.py
# SINTESE: Avaliação final, única, no conjunto de teste (2025-26).
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: avaliação única no teste, ablação e teto.
# === FIM DO CABECALHO ===

"""Avaliacao final no conjunto de teste (temporada 2025-26).

    python -m src.evaluate --final

Este e o UNICO ponto do projeto que le o teste, e deve rodar uma vez, depois
que `python -m src.train` congelou todas as escolhas. Se `test_final.json` ja
existir, o script se recusa a rodar de novo (use --force so para refazer a
mesma avaliacao, nunca para trocar de modelo depois de ver o teste).

Responde, com numeros, aos resultados esperados do relatorio N1:
  (a) o modelo reduz o MAE em relacao ao baseline? (IC por bootstrap de jogador)
  (b) descanso, mando e adversario contribuem? (ablacao por familia)
  (c) a media exponencial supera as medias moveis simples?
  (d) as probabilidades sao calibradas, com Brier abaixo do ingenuo?
  (e) qual o teto preditivo? (oraculos que usam o futuro, so como regua)

Saidas: results/metrics/test_final.json e results/predictions_test.parquet.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.features import PRE_GAME_FEATURES
from src.metrics import (
    classification_metrics,
    cluster_bootstrap_statistic,
    paired_cluster_bootstrap,
    regression_by_group,
    regression_by_minutes,
    regression_metrics,
    reliability_table,
)
from src.modeling import (
    FEATURE_FAMILIES,
    LINE,
    SEED,
    TARGET_CLF,
    TARGET_REG,
    add_clf_features,
    assert_disjoint_temporal,
    clf_feature_set,
    get_split,
    load_dataset,
    minute_band,
    predict_baselines,
    regression_zoo,
)
from src.train import CACHE_DIR, ID_COLS, METRICS_DIR, MODELS_DIR, RESULTS, environment_info
from src.uncertainty import (
    ConformalBand,
    coverage_by_group,
    coverage_report,
    oracle_local,
    oracle_median,
    oracle_season_loo,
)

FINAL_JSON = METRICS_DIR / "test_final.json"
N_PERM = 5


def _abs_err(y, p) -> np.ndarray:
    return np.abs(np.asarray(y, float) - np.asarray(p, float))


def _df_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.reset_index().to_json(orient="records", force_ascii=False))


def grouped_permutation_importance(model, X: pd.DataFrame, y, families: dict, n_repeats: int = N_PERM, seed: int = SEED) -> dict:
    """Aumento de MAE ao embaralhar uma FAMILIA inteira de features de uma vez.

    Embaralhar so uma coluna de um grupo de colunas quase identicas (ewm, ma10,
    media da temporada: |corr| ~0,98) subestima a importancia, porque o modelo
    recupera a informacao pelas colunas irmas. Embaralhar a familia junta (a
    mesma permutacao de linhas para todas as colunas dela) mede a contribuicao
    do grupo.
    """
    rng = np.random.default_rng(seed)
    base = float(np.mean(_abs_err(y, model.predict(X))))
    out = {}
    for fam, cols in families.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            deltas.append(float(np.mean(_abs_err(y, model.predict(Xp)))) - base)
        out[fam] = {"aumento_mae_medio": float(np.mean(deltas)), "desvio": float(np.std(deltas))}
    return {"mae_base": base, "familias": out}


def evaluate(force: bool = False) -> dict:
    if FINAL_JSON.exists() and not force:
        raise SystemExit(
            f"{FINAL_JSON} ja existe: o teste ja foi avaliado. Rodar de novo para "
            "escolher outro modelo seria usar o teste como validacao. Use --force "
            "apenas para refazer exatamente a mesma avaliacao."
        )
    tv_path = METRICS_DIR / "treino_validacao.json"
    if not tv_path.exists():
        raise SystemExit("rode antes: python -m src.train")
    tv = json.loads(tv_path.read_text(encoding="utf-8"))
    t0 = time.time()

    df = add_clf_features(load_dataset())
    assert_disjoint_temporal(df)
    train_df, val_df, test_df = (get_split(df, s) for s in ("train", "val", "test"))
    y_te = test_df[TARGET_REG].to_numpy(dtype=float)
    players = test_df["PLAYER_ID"].to_numpy()
    out: dict = {"ambiente": environment_info(), "linhas_teste": int(len(test_df))}

    preds = test_df[ID_COLS + ["min_ma5_prev", "pts_season_mean_prev", LINE, TARGET_REG, TARGET_CLF]].copy()

    # ============================================================ REGRESSAO
    reg_tv = tv["regressao"]
    final_reg = reg_tv["modelo_final"]
    reg: dict = {"modelo_final": final_reg, "teste": {}}

    for nome, p in predict_baselines(test_df, train_df).items():
        preds[f"reg__{nome}"] = p
        reg["teste"][nome] = regression_metrics(y_te, p) | {"tipo": "baseline"}
    for nome, info in reg_tv["validacao"].items():
        if info["tipo"] != "modelo":
            continue
        model = joblib.load(CACHE_DIR / f"reg__{nome}.joblib")
        preds[f"reg__{nome}"] = model.predict(test_df[PRE_GAME_FEATURES])
        reg["teste"][nome] = regression_metrics(y_te, preds[f"reg__{nome}"]) | {"tipo": "modelo"}

    mu = preds[f"reg__{final_reg}"].to_numpy()
    err = {k[5:]: _abs_err(y_te, preds[k]) for k in preds.columns if k.startswith("reg__")}

    # (a) cada modelo contra a baseline prometida (media de 5) e contra a exponencial
    reg["comparacoes_bootstrap"] = {}
    for nome in err:
        for ref in ("media_5_jogos", "media_exponencial", "linear_sobre_exponencial"):
            if nome == ref:
                continue
            reg["comparacoes_bootstrap"][f"{nome}_vs_{ref}"] = paired_cluster_bootstrap(err[nome], err[ref], players)

    # (c) media exponencial contra as medias moveis simples
    reg["exponencial_vs_simples"] = {
        ref: paired_cluster_bootstrap(err["media_exponencial"], err[ref], players)
        for ref in ("media_3_jogos", "media_5_jogos", "media_10_jogos", "media_da_temporada")
    }

    # Metricas por faixa de minutos (pre-jogo) para o modelo final e as baselines
    reg["por_faixa_de_minutos"] = {
        nome: _df_records(regression_by_minutes(test_df, preds[f"reg__{nome}"]))
        for nome in (final_reg, "media_5_jogos", "media_exponencial")
    }
    # Estrelas vs demais; jogador visto ou nao no treino
    estrela = np.where(test_df["pts_season_mean_prev"] >= 20, "media >= 20 pts", "media < 20 pts")
    novo = np.where(test_df["PLAYER_ID"].isin(set(train_df["PLAYER_ID"])), "jogador presente no treino", "jogador ausente do treino")
    reg["por_perfil"] = {
        "estrela": {n: _df_records(regression_by_group(test_df, preds[f"reg__{n}"], pd.Series(estrela))) for n in (final_reg, "media_exponencial")},
        "novo_no_treino": {n: _df_records(regression_by_group(test_df, preds[f"reg__{n}"], pd.Series(novo))) for n in (final_reg, "media_exponencial")},
    }

    # Deriva por temporada: dobras do CV (fora da amostra), validacao e teste
    por_temp = {}
    cv_final = reg_tv["cv"].get(final_reg, {}).get("score_cv_por_dobra", [])
    temporadas_cv = sorted(train_df["SEASON"].unique())[1:]
    for s, score in zip(temporadas_cv, cv_final):
        part = train_df[train_df["SEASON"] == s]
        por_temp[s] = {
            "modelo_final_mae": -score,
            "media_exponencial_mae": float(np.mean(_abs_err(part[TARGET_REG], part["pts_ewm_prev"]))),
            "media_5_jogos_mae": float(np.mean(_abs_err(part[TARGET_REG], part["pts_ma5_prev"]))),
            "origem": "dobra da validacao cruzada",
        }
    por_temp["2024-25"] = {
        "modelo_final_mae": reg_tv["validacao"][final_reg]["mae"],
        "media_exponencial_mae": reg_tv["validacao"]["media_exponencial"]["mae"],
        "media_5_jogos_mae": reg_tv["validacao"]["media_5_jogos"]["mae"],
        "origem": "validacao",
    }
    por_temp["2025-26"] = {
        "modelo_final_mae": reg["teste"][final_reg]["mae"],
        "media_exponencial_mae": reg["teste"]["media_exponencial"]["mae"],
        "media_5_jogos_mae": reg["teste"]["media_5_jogos"]["mae"],
        "origem": "teste",
    }
    reg["por_temporada"] = por_temp

    # (b) ablacao por familia: retreina o modelo final sem cada familia
    zoo = regression_zoo()
    params = reg_tv["cv"][final_reg]["melhores_parametros"]
    base_pipe = zoo[final_reg][0]
    ablacao = {}
    for fam, cols in FEATURE_FAMILIES.items():
        keep = [c for c in PRE_GAME_FEATURES if c not in cols]
        m = clone(base_pipe).set_params(**params).fit(train_df[keep], train_df[TARGET_REG])
        e_abl = _abs_err(y_te, m.predict(test_df[keep]))
        ablacao[fam] = {
            "features_removidas": cols,
            "mae_sem_familia": float(e_abl.mean()),
            "aumento_vs_completo": paired_cluster_bootstrap(e_abl, err[final_reg], players),
        }
    reg["ablacao_por_familia"] = ablacao

    final_model = joblib.load(MODELS_DIR / "regressao_final.joblib")
    reg["importancia_permutacao_agrupada"] = grouped_permutation_importance(
        final_model, test_df[PRE_GAME_FEATURES], y_te, FEATURE_FAMILIES
    )
    if hasattr(final_model[-1], "coef_"):
        coefs = pd.Series(final_model[-1].coef_, index=PRE_GAME_FEATURES)
        reg["coeficientes_padronizados"] = coefs.sort_values(key=np.abs, ascending=False).to_dict()

    # (e) teto preditivo: oraculos que usam o futuro (so como regua).
    # Em MAE a regua certa e a MEDIANA: oraculos em media ficam artificialmente
    # piores porque os pontos sao assimetricos. Reportamos os dois e a
    # diferenca direta modelo - oraculo (negativa = modelo ja erra menos que o
    # oraculo daquele tipo).
    preds["oraculo_temporada_media"] = oracle_season_loo(test_df)
    preds["oraculo_local_media"] = oracle_local(test_df)
    preds["oraculo_temporada_mediana"] = oracle_median(test_df)
    preds["oraculo_local_mediana"] = oracle_median(test_df, half_window=5)
    mae_final = reg["teste"][final_reg]["mae"]
    teto = {"nota": (
        "Oraculos usam jogos futuros do proprio jogador (temporada inteira ou 5 antes/5 depois, "
        "sem o proprio jogo). Nao sao utilizaveis na pratica: medem quanto erro sobra mesmo "
        "conhecendo o nivel real do atleta. Versao em mediana acrescentada apos a primeira "
        "execucao (2026-11-15); nao altera nenhuma escolha de modelo."
    )}
    for o in ("oraculo_temporada_media", "oraculo_local_media", "oraculo_temporada_mediana", "oraculo_local_mediana"):
        e_o = _abs_err(y_te, preds[o])
        teto[o] = {
            "mae": float(e_o.mean()),
            "modelo_menos_oraculo": paired_cluster_bootstrap(err[final_reg], e_o, players),
        }
    melhor = min((k for k in teto if k.startswith("oraculo")), key=lambda k: teto[k]["mae"])
    teto["melhor_oraculo"] = melhor
    teto["ganho_sobre_exponencial_que_o_melhor_oraculo_teria"] = reg["teste"]["media_exponencial"]["mae"] - teto[melhor]["mae"]
    teto["ganho_sobre_exponencial_do_modelo"] = reg["teste"]["media_exponencial"]["mae"] - mae_final
    reg["teto_preditivo"] = teto

    # Incerteza: cobertura do intervalo conformal no teste
    conf = json.loads((MODELS_DIR / "conformal.json").read_text(encoding="utf-8"))
    reg["intervalos"] = {}
    terc = pd.Series(pd.qcut(mu, 3, labels=["previsao baixa", "previsao media", "previsao alta"]))
    for nivel, d in conf.items():
        band = ConformalBand.from_dict(d)
        lo, hi = band.predict(mu)
        preds[f"intervalo{nivel}_inf"], preds[f"intervalo{nivel}_sup"] = lo, hi
        reg["intervalos"][nivel] = {
            "nominal": 1 - band.alpha,
            "q": band.q,
            "global": coverage_report(y_te, lo, hi),
            "por_faixa_de_minutos": _df_records(coverage_by_group(y_te, lo, hi, minute_band(test_df))),
            "por_tercil_da_previsao": _df_records(coverage_by_group(y_te, lo, hi, terc)),
        }

    # Analise de erro: maiores residuos
    res = preds.assign(previsto=mu, erro=y_te - mu)
    top = res.reindex(res["erro"].abs().sort_values(ascending=False).index).head(20)
    reg["maiores_erros"] = json.loads(top[["PLAYER_NAME", "GAME_DATE", "OPPONENT", "min_ma5_prev", "pts_season_mean_prev", "previsto", TARGET_REG, "erro"]]
                                      .assign(GAME_DATE=lambda d: d["GAME_DATE"].dt.strftime("%Y-%m-%d")).to_json(orient="records", force_ascii=False))
    reg["vies_por_decil_da_previsao"] = _df_records(
        res.assign(decil=pd.qcut(mu, 10, labels=False) + 1).groupby("decil").agg(
            previsto_medio=("previsto", "mean"), real_medio=(TARGET_REG, "mean"), n=(TARGET_REG, "size")
        )
    )
    out["regressao"] = reg

    # ========================================================= CLASSIFICACAO
    clf_tv = tv["classificacao"]
    base_rate = clf_tv["taxa_positiva_treino"]
    yc = test_df[TARGET_CLF].astype(int).to_numpy()
    sem_empate = (test_df[TARGET_REG] != test_df[LINE]).to_numpy()
    feats = clf_feature_set("completo")
    clf: dict = {"modelo_final": clf_tv["modelo_final"], "calibracao": clf_tv["calibracao"], "teste": {}, "teste_sem_empates": {}}

    def _both(nome: str, proba) -> None:
        clf["teste"][nome] = classification_metrics(yc, proba, base_rate)
        clf["teste_sem_empates"][nome] = classification_metrics(yc[sem_empate], np.asarray(proba)[sem_empate], base_rate)

    preds["clf__constante_treino"] = base_rate
    _both("constante_treino", np.full(len(yc), base_rate))
    for nome, info in clf_tv["validacao"].items():
        if info["tipo"] != "modelo":
            continue
        m = joblib.load(CACHE_DIR / f"clf__{nome}.joblib")
        preds[f"clf__{nome}"] = m.predict_proba(test_df[feats])[:, 1]
        _both(nome, preds[f"clf__{nome}"])
    final_clf = joblib.load(MODELS_DIR / "classificacao_final.joblib")
    preds["clf__final_calibrado"] = final_clf.predict_proba(test_df[feats])[:, 1]
    _both("final_calibrado", preds["clf__final_calibrado"])

    # Regra sem treino: "a media exponencial esta acima da linha?"
    regra = test_df["diff_ewm_linha"].to_numpy()
    clf["regra_exponencial_acima_da_linha"] = {
        "acuracia": float(np.mean((regra > 0).astype(int) == yc)),
        "auc": float(roc_auc_score(yc, regra)),
        "acuracia_sem_empates": float(np.mean((regra[sem_empate] > 0).astype(int) == yc[sem_empate])),
        "auc_sem_empates": float(roc_auc_score(yc[sem_empate], regra[sem_empate])),
    }

    p_final = preds["clf__final_calibrado"].to_numpy()
    clf["auc_ic"] = cluster_bootstrap_statistic(yc, p_final, players, roc_auc_score)
    clf["auc_ic_regra"] = cluster_bootstrap_statistic(yc, regra, players, roc_auc_score)
    clf["brier_vs_ingenuo"] = paired_cluster_bootstrap((p_final - yc) ** 2, (base_rate - yc) ** 2, players)
    clf["curva_calibracao"] = json.loads(reliability_table(yc, p_final).to_json(orient="records"))
    raw_final = preds[f"clf__{clf_tv['modelo_final']}"].to_numpy()
    clf["curva_calibracao_sem_ajuste"] = json.loads(reliability_table(yc, raw_final).to_json(orient="records"))

    clf["ablacao"] = {"completo": clf["teste"][clf_tv["modelo_final"]]}
    for variant in ("sem_linha_inteira", "so_pre_jogo"):
        m = joblib.load(CACHE_DIR / f"clf_ablacao__{variant}.joblib")
        clf["ablacao"][variant] = classification_metrics(yc, m.predict_proba(test_df[clf_feature_set(variant)])[:, 1], base_rate)
    clf["empates_no_teste"] = {"n": int((~sem_empate).sum()), "fracao": float((~sem_empate).mean())}
    out["classificacao"] = clf
    out["tempo_s"] = round(time.time() - t0, 1)

    preds.to_parquet(RESULTS / "predictions_test.parquet", index=False)
    FINAL_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    r = reg["teste"]
    print(f"[teste] {final_reg}: MAE {r[final_reg]['mae']:.3f} | media 5 {r['media_5_jogos']['mae']:.3f} | exponencial {r['media_exponencial']['mae']:.3f}")
    c = clf["teste"]["final_calibrado"]
    print(f"[teste] classificacao: AUC {c['auc']:.3f} Brier {c['brier']:.4f} (ingenuo {c['brier_ingenuo']:.4f}) acc {c['acuracia']:.3f}")
    print(f"ok: {FINAL_JSON} ({out['tempo_s']}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliacao final no teste (roda uma vez).")
    ap.add_argument("--final", action="store_true", help="confirma que as escolhas estao congeladas")
    ap.add_argument("--force", action="store_true", help="refaz a mesma avaliacao se o JSON ja existir")
    args = ap.parse_args()
    if not args.final:
        raise SystemExit("use --final para confirmar que o treino esta congelado")
    evaluate(force=args.force)


if __name__ == "__main__":
    main()
