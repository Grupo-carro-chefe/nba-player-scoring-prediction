# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/figures.py
# SINTESE: Figuras do relatório a partir dos resultados salvos.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: figuras do relatório.
#   2026-11-15 | Rodrigo    | Figura da distribuição dos pontos (EDA) e baseline em mediana na escada.
#   2026-11-15 | Rodrigo    | Título e eixo calculados a partir dos dados.
#   2026-11-17 | Rodrigo    | Figura de dispersão previsto x observado com distribuições marginais.
# === FIM DO CABECALHO ===

"""Figuras do relatorio, geradas a partir dos JSONs e previsoes salvos.

    python -m src.figures

Nao treina nada: le results/metrics/*.json e results/predictions_test.parquet.
As cores seguem uma paleta categorica fixa (azul = modelo final, laranja =
media de 5 jogos, aqua = media exponencial), sempre na mesma ordem, para que a
mesma serie tenha a mesma cor em todas as figuras.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.train import FIG_DIR, METRICS_DIR, RESULTS  # noqa: E402

AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CINZA, TINTA, TINTA_2, GRADE = "#a3a29d", "#0b0b0b", "#52514e", "#e6e5e1"
SUPERFICIE = "#fcfcfb"

NOMES = {
    "constante_media_treino": "Constante (média do treino)",
    "constante_mediana_treino": "Constante (mediana do treino)",
    "media_3_jogos": "Média 3 jogos",
    "media_5_jogos": "Média 5 jogos (baseline N1)",
    "media_10_jogos": "Média 10 jogos",
    "mediana_10_jogos": "Mediana 10 jogos (linha)",
    "media_da_temporada": "Média da temporada",
    "media_exponencial": "Média exponencial (0,9)",
    "linear_sobre_exponencial": "Linear sobre a exponencial",
    "mediana_sobre_exponencial": "Mediana sobre a exponencial",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "poisson_glm": "GLM Poisson",
    "random_forest": "Floresta aleatória",
    "hgb_quadratico": "Gradient boosting (quadrático)",
    "hgb_poisson": "Gradient boosting (Poisson)",
    "hgb_absoluto": "Gradient boosting (absoluto)",
    "historico_pontos": "Histórico de pontos",
    "minutos_e_papel": "Minutos e papel",
    "volume_de_jogo": "Volume de jogo",
    "descanso_e_mando": "Descanso e mando",
    "adversario": "Adversário",
    "calendario_do_jogador": "Jogos na temporada",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRADE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def fig_distribuicao_pontos(df: pd.DataFrame) -> None:
    """Histograma de pontos por jogo com media e mediana (assimetria a direita)."""
    y = df["y_pts"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(y, bins=np.arange(0, y.max() + 2) - 0.5, color=AZUL, edgecolor=SUPERFICIE, linewidth=0.4)
    ax.axvline(y.median(), color=AQUA, linewidth=2, label=f"Mediana: {y.median():.0f}".replace(".", ","))
    ax.axvline(y.mean(), color=LARANJA, linewidth=2, label=f"Média: {y.mean():.2f}".replace(".", ","))
    ax.set_xlim(-1, 60)
    ax.set_xlabel("Pontos marcados no jogo")
    ax.set_ylabel("Jogos")
    temporadas = sorted(df["SEASON"].unique())
    ax.set_title(f"Distribuição do alvo ({len(df):,} jogos, {temporadas[0]} a {temporadas[-1]})".replace(",", "."))
    ax.legend(loc="upper right")
    _save(fig, "f_distribuicao_pontos")


def fig_escada_mae(res: dict) -> None:
    """MAE no teste de todas as baselines e modelos, do pior ao melhor."""
    t = dict(res["regressao"]["teste"])
    diag = res["regressao"].get("diagnosticos_posteriores", {}).get("mediana_sobre_exponencial")
    if diag:
        t["mediana_sobre_exponencial"] = {"mae": diag["mae_teste"]}
    final = res["regressao"]["modelo_final"]
    items = sorted(t.items(), key=lambda kv: kv[1]["mae"], reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (nome, m) in enumerate(items):
        cor = AZUL if nome == final else (LARANJA if nome == "media_5_jogos" else (AQUA if nome == "media_exponencial" else CINZA))
        ax.barh(i, m["mae"], color=cor, height=0.62)
        ax.text(m["mae"] + 0.03, i, f"{m['mae']:.3f}".replace(".", ","), va="center", fontsize=8.5, color=TINTA_2)
    ax.set_yticks(range(len(items)), [NOMES.get(n, n) + ("  (final)" if n == final else "") for n, _ in items], fontsize=8.5)
    ax.set_xlabel("Erro absoluto médio no teste (pontos)")
    maes = [m["mae"] for _, m in items]
    ax.set_xlim(np.floor(min(maes) * 2) / 2 - 0.5, max(maes) + 0.4)  # nunca corta a menor barra
    ax.grid(axis="y", visible=False)
    ax.set_title("Escada de previsores: baselines e modelos (teste 2025-26)")
    _save(fig, "f_escada_mae")


def fig_mae_por_faixa(res: dict) -> None:
    pf = res["regressao"]["por_faixa_de_minutos"]
    final = res["regressao"]["modelo_final"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for nome, cor in ((final, AZUL), ("media_5_jogos", LARANJA), ("media_exponencial", AQUA)):
        d = pd.DataFrame(pf[nome])
        ax.plot(d["grupo"], d["mae"], marker="o", markersize=7, linewidth=2, color=cor, label=NOMES.get(nome, nome))
    ax.set_xlabel("Média de minutos nos 5 jogos anteriores")
    ax.set_ylabel("MAE (pontos)")
    ax.set_title("Erro por faixa de minutos (teste)")
    ax.legend(loc="upper left")
    _save(fig, "f_mae_por_faixa_minutos")


def fig_previsto_observado(preds: pd.DataFrame, final: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5))
    hb = ax.hexbin(preds[f"reg__{final}"], preds["y_pts"], gridsize=45, cmap="Blues", mincnt=1, bins="log")
    lim = [0, max(preds["y_pts"].max(), preds[f"reg__{final}"].max()) + 2]
    ax.plot(lim, lim, color=TINTA_2, linewidth=1, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Pontos previstos")
    ax.set_ylabel("Pontos marcados")
    ax.set_title("Previsto vs. observado (teste)")
    fig.colorbar(hb, ax=ax, label="linhas (escala log)")
    _save(fig, "f_previsto_vs_observado")


def fig_previsto_marginais(preds: pd.DataFrame, final: str) -> None:
    """Dispersao previsto x observado com a distribuicao de cada eixo nas laterais.

    O centro e densidade (hexbin), nao pontos: sao ~17 mil jogos e pontos soltos
    virariam uma mancha. As laterais mostram o achado do relatorio: as previsoes
    ficam espremidas num intervalo estreito, enquanto os pontos marcados se
    espalham muito mais; a distancia entre os dois histogramas e a variacao de
    jogo a jogo que nenhuma informacao pre-jogo explica. Os dois eixos usam a
    mesma escala, para que a diagonal tracejada seja a previsao perfeita.
    """
    mu, y = preds[f"reg__{final}"].to_numpy(float), preds["y_pts"].to_numpy(float)
    lim = float(np.ceil(max(np.quantile(y, 0.999), np.quantile(mu, 0.999)))) + 2
    fora = int(((y > lim) | (mu > lim)).sum())
    mae = float(np.mean(np.abs(y - mu)))

    fig = plt.figure(figsize=(6.6, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    canto = fig.add_subplot(gs[0, 1])
    canto.axis("off")

    hb = ax.hexbin(mu, y, gridsize=38, extent=(0, lim, 0, lim), cmap="Blues", mincnt=1, bins="log", linewidths=0.2, edgecolors=SUPERFICIE)
    ax.plot([0, lim], [0, lim], color=TINTA_2, linewidth=1, linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Pontos previstos pelo modelo")
    ax.set_ylabel("Pontos marcados no jogo")
    ax.text(lim * 0.97, lim * 0.93, "previsão perfeita", color=TINTA_2, fontsize=8.5, ha="right", rotation=0)

    bins = np.arange(0, lim + 1, 1.0)
    ax_x.hist(mu, bins=bins, color=AZUL, rwidth=0.85)
    ax_y.hist(y, bins=bins, color=CINZA, rwidth=0.85, orientation="horizontal")
    # Os eixos das laterais compartilham escala com o grafico central, entao so se
    # esconde o que e decoracao (ticks e linhas), nunca a escala em si.
    for lateral in (ax_x, ax_y):
        lateral.grid(False)
    ax_x.set_yticks([])
    ax_x.spines["left"].set_visible(False)
    ax_x.tick_params(labelbottom=False, bottom=False)
    ax_y.set_xticks([])
    ax_y.spines["bottom"].set_visible(False)
    ax_y.tick_params(labelleft=False, left=False)
    ax_x.text(0.99, 0.80, "distribuição das previsões", transform=ax_x.transAxes, color=TINTA_2, fontsize=8.5, ha="right")
    ax_y.text(0.97, 0.99, "distribuição\ndo que\nocorreu", transform=ax_y.transAxes, color=TINTA_2, fontsize=8.5, ha="right", va="top")

    cax = canto.inset_axes([0.62, 0.05, 0.14, 0.72])
    cb = fig.colorbar(hb, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("jogos (log)", fontsize=8.5)

    nota = f"{len(y):,} jogos de teste (2025-26)".replace(",", ".")
    if fora:
        nota += f"; {fora} fora do recorte dos eixos"
    titulo = "Previsto vs. observado · MAE " + f"{mae:.2f}".replace(".", ",") + " pontos"
    ax_x.set_title(titulo, loc="left", fontsize=11, fontweight="bold", pad=22)
    ax_x.text(0.0, 1.07, nota, transform=ax_x.transAxes, fontsize=8.5, color=TINTA_2, ha="left")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "f_previsto_vs_observado_marginais.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_calibracao(res: dict) -> None:
    c = res["classificacao"]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=TINTA_2, linewidth=1, linestyle="--", label="Calibração perfeita")
    d = pd.DataFrame(c["curva_calibracao"])
    ax.plot(d["prob_media_prevista"], d["frequencia_observada"], marker="o", markersize=7, linewidth=2, color=AZUL, label="Modelo final")
    ax.axhline(res["classificacao"]["teste"]["constante_treino"]["taxa_positiva"], color=LARANJA, linewidth=1.5, label="Taxa observada no teste")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("Probabilidade prevista (média da faixa)")
    ax.set_ylabel("Frequência observada de superação")
    ax.set_title("Curva de calibração (teste, 10 faixas)")
    ax.legend(loc="upper left")
    _save(fig, "f_curva_calibracao")


def fig_cobertura(res: dict) -> None:
    iv = res["regressao"]["intervalos"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for nivel, cor in (("80", AZUL), ("90", LARANJA)):
        d = pd.DataFrame(iv[nivel]["por_faixa_de_minutos"])
        ax.plot(d["grupo"], d["cobertura"], marker="o", markersize=7, linewidth=2, color=cor, label=f"Intervalo de {nivel}%")
        ax.axhline(int(nivel) / 100, color=cor, linewidth=1, linestyle=":")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Média de minutos nos 5 jogos anteriores")
    ax.set_ylabel("Cobertura observada")
    ax.set_title("Cobertura dos intervalos conformais (teste; pontilhado = nominal)")
    ax.legend(loc="lower left")
    _save(fig, "f_cobertura_intervalos")


def fig_ablacao(res: dict) -> None:
    ab = res["regressao"]["ablacao_por_familia"]
    items = sorted(ab.items(), key=lambda kv: kv[1]["aumento_vs_completo"]["delta"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (fam, v) in enumerate(items):
        b = v["aumento_vs_completo"]
        ax.errorbar(b["delta"], i, xerr=[[b["delta"] - b["ic_inf"]], [b["ic_sup"] - b["delta"]]],
                    fmt="o", color=AZUL, markersize=7, capsize=4, linewidth=2)
    ax.axvline(0, color=TINTA_2, linewidth=1)
    ax.set_yticks(range(len(items)), [NOMES.get(f, f) for f, _ in items])
    ax.set_xlabel("Aumento do MAE ao remover a família (pontos; IC 95% por jogador)")
    ax.set_title("Ablação por família de features (teste)")
    _save(fig, "f_ablacao_familias")


def fig_deriva(res: dict) -> None:
    pt = pd.DataFrame(res["regressao"]["por_temporada"]).T
    final = res["regressao"]["modelo_final"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for col, cor, nome in (("modelo_final_mae", AZUL, NOMES.get(final, final)), ("media_5_jogos_mae", LARANJA, NOMES["media_5_jogos"]), ("media_exponencial_mae", AQUA, NOMES["media_exponencial"])):
        ax.plot(pt.index, pt[col].astype(float), marker="o", markersize=7, linewidth=2, color=cor, label=nome)
    ax.set_ylabel("MAE (pontos)")
    ax.set_title("Erro por temporada (dobras de CV, validação e teste)")
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, "f_mae_por_temporada")


def make_all() -> list[str]:
    _style()
    res = json.loads((METRICS_DIR / "test_final.json").read_text(encoding="utf-8"))
    preds = pd.read_parquet(RESULTS / "predictions_test.parquet")
    from src.modeling import load_dataset

    fig_distribuicao_pontos(load_dataset())
    fig_escada_mae(res)
    fig_mae_por_faixa(res)
    fig_previsto_observado(preds, res["regressao"]["modelo_final"])
    fig_previsto_marginais(preds, res["regressao"]["modelo_final"])
    fig_calibracao(res)
    fig_cobertura(res)
    fig_ablacao(res)
    fig_deriva(res)
    return sorted(p.name for p in FIG_DIR.glob("*.png"))


if __name__ == "__main__":
    print(make_all())
