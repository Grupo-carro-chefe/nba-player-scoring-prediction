# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/predict.py
# SINTESE: Artefato de demonstração: pontos previstos, intervalo e probabilidade.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: demonstração com intervalo e aviso de escopo.
#   2026-11-15 | Rodrigo    | Saída em UTF-8 também quando redirecionada (Windows).
#   2026-11-15 | Rodrigo    | --team sem data mostra só o jogo mais recente.
#   2026-11-15 | Rodrigo    | Recusa nome ambíguo e --sample fora do intervalo.
#   2026-11-17 | Rodrigo    | --sample 0 passa a ser recusado com a mensagem de intervalo (antes caía em 'informe --player').
# === FIM DO CABECALHO ===

"""Artefato de demonstracao: estimativa de pontos com incerteza para um jogo.

    python -m src.predict --player "Jalen Brunson" --date 2026-01-11
    python -m src.predict --sample 5            # 5 jogos sorteados do teste
    python -m src.predict --team BOS --date 2026-01-10

Le apenas o dataset versionado e os modelos finais em results/models/. Para o
jogo pedido, mostra:

- pontos previstos pelo modelo final e o intervalo conformal de 80% e 90%;
- a baseline (media dos 5 jogos anteriores), para comparacao;
- a probabilidade de o jogador superar o proprio patamar recente (mediana dos
  10 jogos anteriores);
- o que de fato aconteceu, ja que a demonstracao usa jogos passados.

Escopo: so jogadores de rotacao (media >= 15 min nos 5 jogos anteriores, com
10+ jogos na temporada) em temporada regular. Fora disso o dataset nao tem a
linha e o script recusa a consulta em vez de inventar um numero.

Isto e uma estimativa estatistica de desempenho esportivo. Nao e recomendacao
de aposta: nao ha odds, valor esperado nem gestao de banca, e o "patamar
recente" nao e uma linha de mercado.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd

from src.features import PRE_GAME_FEATURES
from src.modeling import SEED, add_clf_features, clf_feature_set, load_dataset
from src.train import MODELS_DIR
from src.uncertainty import ConformalBand

AVISO = (
    "Estimativa estatística de desempenho, com incerteza. "
    "Não é recomendação de aposta: o patamar recente é a mediana do próprio "
    "jogador, não uma linha de mercado."
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().strip()


def load_artifacts():
    reg = joblib.load(MODELS_DIR / "regressao_final.joblib")
    clf = joblib.load(MODELS_DIR / "classificacao_final.joblib")
    conf = {k: ConformalBand.from_dict(v) for k, v in json.loads((MODELS_DIR / "conformal.json").read_text(encoding="utf-8")).items()}
    return reg, clf, conf


def predict_rows(rows: pd.DataFrame, reg, clf, conf) -> pd.DataFrame:
    """Previsao, intervalos e probabilidade para linhas do dataset."""
    rows = add_clf_features(rows)
    out = rows[["PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "GAME_DATE", "split",
                "min_ma5_prev", "pts_ma5_prev", "line_synthetic", "y_pts"]].copy()
    out["previsto"] = reg.predict(rows[PRE_GAME_FEATURES])
    for nivel, band in conf.items():
        lo, hi = band.predict(out["previsto"])
        out[f"ic{nivel}_inf"], out[f"ic{nivel}_sup"] = lo, hi
    out["prob_superar_patamar"] = clf.predict_proba(rows[clf_feature_set("completo")])[:, 1]
    return out


def render(r: pd.Series) -> str:
    def f(x: float, nd: int = 1) -> str:
        return f"{x:.{nd}f}".replace(".", ",")

    dentro = r["ic80_inf"] <= r["y_pts"] <= r["ic80_sup"]
    superou = r["y_pts"] > r["line_synthetic"]
    linhas = [
        f"{r['PLAYER_NAME']} ({r['TEAM_ABBREVIATION']}) x {r['OPPONENT']}  |  {r['GAME_DATE']:%d/%m/%Y}  |  split: {r['split']}",
        "-" * 78,
        f"  Pontos previstos ............ {f(r['previsto'])}",
        f"  Intervalo de 80% ............ {f(r['ic80_inf'])} a {f(r['ic80_sup'])}",
        f"  Intervalo de 90% ............ {f(r['ic90_inf'])} a {f(r['ic90_sup'])}",
        f"  Baseline (média 5 jogos) .... {f(r['pts_ma5_prev'])}",
        f"  Patamar recente (mediana 10)  {f(r['line_synthetic'])}",
        f"  P(superar o patamar) ........ {f(100 * r['prob_superar_patamar'], 0)}%",
        f"  Minutos médios (5 jogos) .... {f(r['min_ma5_prev'])}",
        "-" * 78,
        f"  Real: {int(r['y_pts'])} pontos  |  erro do modelo: {f(r['y_pts'] - r['previsto'])}  |  "
        f"erro da baseline: {f(r['y_pts'] - r['pts_ma5_prev'])}",
        f"  Dentro do intervalo de 80%? {'sim' if dentro else 'não'}  |  superou o patamar? {'sim' if superou else 'não'}",
    ]
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimativa de pontos com incerteza (jogos passados).")
    ap.add_argument("--player", help="nome do jogador (parcial, sem acento)")
    ap.add_argument("--team", help="sigla do time, ex.: BOS")
    ap.add_argument("--date", help="data do jogo AAAA-MM-DD")
    ap.add_argument("--sample", type=int, help="sorteia N jogos do conjunto de teste")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    # Acentos corretos tambem quando a saida e redirecionada (no Windows o
    # padrao de um pipe e cp1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    df = load_dataset()
    if args.sample is not None:  # 0 tambem entra aqui, para a guarda abaixo recusa-lo com a mensagem certa
        teste = df[df["split"] == "test"]
        if not 0 < args.sample <= len(teste):
            raise SystemExit(f"--sample deve estar entre 1 e {len(teste)}")
        rows = teste.sample(args.sample, random_state=args.seed)
    else:
        rows = df
        if args.player:
            alvo = _norm(args.player)
            rows = rows[rows["PLAYER_NAME"].map(_norm).str.contains(alvo, regex=False)]
        if args.team:
            rows = rows[rows["TEAM_ABBREVIATION"] == args.team.upper()]
        if args.date:
            rows = rows[rows["GAME_DATE"] == pd.Timestamp(args.date)]
        if not (args.player or args.team):
            raise SystemExit("informe --player, --team ou --sample")
        if rows.empty:
            raise SystemExit(
                "Jogo fora do escopo do modelo ou inexistente. O modelo só cobre jogadores de "
                "rotação (média >= 15 min nos 5 jogos anteriores) com 10+ jogos na temporada, "
                "em temporada regular de 2021-22 a 2025-26."
            )
        if args.player and rows["PLAYER_ID"].nunique() > 1:
            nomes = sorted(rows["PLAYER_NAME"].unique())
            raise SystemExit("o nome bate com mais de um jogador; seja mais específico: " + ", ".join(nomes[:15]))
        if not args.date:
            if args.player:
                # sem data: os 5 jogos mais recentes do jogador
                rows = rows.sort_values("GAME_DATE").tail(5)
            else:
                # time sem data: so o jogo mais recente do time, nao as cinco temporadas
                rows = rows[rows["GAME_DATE"] == rows["GAME_DATE"].max()]

    reg, clf, conf = load_artifacts()
    res = predict_rows(rows.reset_index(drop=True), reg, clf, conf)
    for _, r in res.iterrows():
        print(render(r))
        print()
    if len(res) > 1:
        cob = np.mean((res["y_pts"] >= res["ic80_inf"]) & (res["y_pts"] <= res["ic80_sup"]))
        print(f"Resumo: {len(res)} jogos, {cob:.0%} dentro do intervalo de 80%.")
    print(AVISO)


if __name__ == "__main__":
    main()
