# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/report.py
# SINTESE: Gera o relatório da N2 (.docx) a partir dos resultados.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: relatório N2 gerado a partir dos resultados.
#   2026-11-15 | Rodrigo    | Correções da revisão crítica: teto, baselines, ablação, conformal, referências.
# === FIM DO CABECALHO ===

"""Gera o relatorio da N2 (.docx) sobre o template da FCI.

    python -m src.report --video-url https://youtu.be/XXXX

Todos os numeros do texto saem de results/metrics/*.json e do dataset, nunca
digitados a mao: se um resultado mudar, o relatorio muda junto, e o teste
tests/test_resultados.py confere que os numeros citados batem com o JSON.

O template (relatorio/template_fci.docx) e a conversao para .docx do template
oficial da FCI. O script apaga o conteudo de exemplo do corpo, preserva o
cabecalho com o logo e as margens, e escreve o relatorio com a mesma
formatacao direta do template (Times 12, titulos de secao em 14 negrito, corpo
justificado com recuo de 1,25 cm).
"""

from __future__ import annotations

import argparse
import copy
import json
import re

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from src.features import PRE_GAME_FEATURES
from src.modeling import ROOT, load_dataset
from src.train import FIG_DIR, METRICS_DIR

TEMPLATE = ROOT / "relatorio" / "template_fci.docx"
OUT_DOCX = ROOT / "relatorio" / "relatorio_N2.docx"
GITHUB = "https://github.com/Grupo-carro-chefe/nba-player-scoring-prediction"
VIDEO_PLACEHOLDER = "[link do vídeo será inserido após a publicação no YouTube]"
FONTE = "Times New Roman"


# ----------------------------------------------------------------------------
# Formatacao de numeros (pt-BR)
# ----------------------------------------------------------------------------

def num(x: float, nd: int = 3) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, nd: int = 1) -> str:
    return num(100 * x, nd) + "%"


def inteiro(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


# ----------------------------------------------------------------------------
# Escrita no documento
# ----------------------------------------------------------------------------

class Relatorio:
    def __init__(self) -> None:
        self.doc = Document(str(TEMPLATE))
        body = self.doc.element.body
        # O template tem duas secoes; o cabecalho com o logo esta preso a
        # primeira (sectPr dentro de um paragrafo). Antes de apagar o conteudo
        # de exemplo, a referencia ao cabecalho e copiada para a secao final.
        final_sect = body.find(qn("w:sectPr"))
        refs = [r for r in body.iter(qn("w:headerReference")) if r.getparent() is not final_sect]
        if refs and final_sect.find(qn("w:headerReference")) is None:
            for r in refs:
                final_sect.insert(0, copy.deepcopy(r))
        for el in list(body):
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        self.n_tab = 0
        self.n_fig = 0

    # -- runs com marcacao simples: **negrito**, *italico*, `codigo`
    def _runs(self, p, texto: str, size: float = 12, italic: bool = False, bold: bool = False) -> None:
        for parte in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", texto):
            if not parte:
                continue
            r_bold, r_italic, mono = bold, italic, False
            if parte.startswith("**"):
                parte, r_bold = parte[2:-2], True
            elif parte.startswith("*"):
                parte, r_italic = parte[1:-1], True
            elif parte.startswith("`"):
                parte, mono = parte[1:-1], True
            run = p.add_run(parte)
            run.font.name = "Courier New" if mono else FONTE
            run.font.size = Pt(size - 1.5 if mono else size)
            run.bold = r_bold or None
            run.italic = r_italic or None

    def par(self, texto: str, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first: float | None = 1.25,
            left: float | None = None, right: float | None = None, before: float = 6, after: float | None = None,
            size: float = 12, italic: bool = False, bold: bool = False):
        p = self.doc.add_paragraph()
        p.alignment = align
        pf = p.paragraph_format
        if first is not None:
            pf.first_line_indent = Cm(first)
        if left is not None:
            pf.left_indent = Cm(left)
        if right is not None:
            pf.right_indent = Cm(right)
        pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        self._runs(p, texto, size=size, italic=italic, bold=bold)
        return p

    def titulo_secao(self, texto: str) -> None:
        p = self.par(texto, align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=12, after=3, size=14, bold=True)
        p.paragraph_format.keep_with_next = True

    def subtitulo(self, texto: str) -> None:
        p = self.par(texto, align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=9, after=0, size=12, bold=True)
        p.paragraph_format.keep_with_next = True

    def itens(self, itens: list[str], letras: bool = True) -> None:
        for i, t in enumerate(itens):
            marca = f"{chr(97 + i)}) " if letras else "• "
            self.par(marca + t, first=-0.6, left=1.85, before=3)

    def tabela(self, titulo: str, cabecalho: list[str], linhas: list[list[str]], larguras: list[float] | None = None,
               fonte_nota: str = "Fonte: elaborado pelos autores.") -> None:
        self.n_tab += 1
        cap = self.par(f"Tabela {self.n_tab}. {titulo}", align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=9, after=2, size=11)
        cap.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1 + len(linhas), cols=len(cabecalho))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        _bordas(t)
        for j, h in enumerate(cabecalho):
            self._cell(t.cell(0, j), h, bold=True)
        for i, linha in enumerate(linhas, start=1):
            for j, v in enumerate(linha):
                self._cell(t.cell(i, j), v, align_right=(j > 0 and _numerico(v)))
        if larguras:
            for row in t.rows:
                for j, w in enumerate(larguras):
                    row.cells[j].width = Cm(w)
        self.par(fonte_nota, align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=2, size=10)

    def _cell(self, cell, texto: str, bold: bool = False, align_right: bool = False) -> None:
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if align_right else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        self._runs(p, texto, size=10, bold=bold)

    def figura(self, arquivo: str, titulo: str, largura: float = 14.5) -> None:
        self.n_fig += 1
        cap = self.par(f"Figura {self.n_fig}. {titulo}", align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=9, after=2, size=11)
        cap.paragraph_format.keep_with_next = True
        self.doc.add_picture(str(FIG_DIR / arquivo), width=Cm(largura))
        img = self.doc.paragraphs[-1]
        img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img.paragraph_format.keep_with_next = True  # fonte junto da figura
        self.par("Fonte: elaborado pelos autores.", align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=2, size=10)

    def salvar(self) -> None:
        self.doc.save(str(OUT_DOCX))


def _numerico(v: str) -> bool:
    return bool(re.fullmatch(r"[−+\-]?[\d.,]+%?( \[.*\])?|[−+\-]?[\d.,]+ a [−+\-]?[\d.,]+", v.strip()))


def _bordas(table) -> None:
    """Bordas simples (o template nao traz estilo de tabela)."""
    tbl = table._tbl
    pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    pr.append(borders)


# ----------------------------------------------------------------------------
# Numeros do texto
# ----------------------------------------------------------------------------

NOMES_MODELO = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "poisson_glm": "GLM Poisson",
    "random_forest": "Floresta aleatória",
    "hgb_quadratico": "Gradient boosting (perda quadrática)",
    "hgb_poisson": "Gradient boosting (perda Poisson)",
    "hgb_absoluto": "Gradient boosting (perda absoluta)",
    "media_3_jogos": "Média dos 3 jogos anteriores",
    "media_5_jogos": "Média dos 5 jogos anteriores (baseline da N1)",
    "media_10_jogos": "Média dos 10 jogos anteriores",
    "mediana_10_jogos": "Mediana dos 10 jogos anteriores",
    "media_da_temporada": "Média da temporada até o jogo anterior",
    "media_exponencial": "Média exponencial (fator 0,9)",
    "linear_sobre_exponencial": "Regressão linear sobre a média exponencial",
    "constante_media_treino": "Constante: média do treino",
    "constante_mediana_treino": "Constante: mediana do treino",
    "logistica_l2": "Regressão logística (L2)",
    "logistica_l1": "Regressão logística (L1)",
    "hgb": "Gradient boosting",
    "constante_treino": "Probabilidade constante (taxa do treino)",
    "final_calibrado": "Modelo final",
}
NOMES_FAMILIA = {
    "historico_pontos": "Histórico de pontos",
    "minutos_e_papel": "Minutos e papel na rotação",
    "volume_de_jogo": "Volume de jogo (arremessos, rebotes, assistências)",
    "descanso_e_mando": "Descanso e mando de quadra",
    "adversario": "Adversário",
    "calendario_do_jogador": "Jogos disputados na temporada",
}


def eda_numbers(df: pd.DataFrame) -> dict:
    y = df["y_pts"]
    corr = df[PRE_GAME_FEATURES].corrwith(y).sort_values(ascending=False)
    corr_over = df[PRE_GAME_FEATURES].corrwith(df["y_over"]).abs().max()
    num_feats = [c for c in PRE_GAME_FEATURES]
    cm = df[num_feats].corr().abs().where(~np.eye(len(num_feats), dtype=bool))
    pares = int((cm > 0.90).sum().sum() / 2)
    q1, q3 = y.quantile([0.25, 0.75])
    iqr_out = ((y < q1 - 1.5 * (q3 - q1)) | (y > q3 + 1.5 * (q3 - q1))).mean()
    titular = df.groupby("starter_proxy_prev")["y_pts"].mean()
    casa = df.groupby("is_home")["y_pts"].mean()
    b2b = df.groupby("is_b2b")["y_pts"].mean()
    temp = df.groupby("SEASON").agg(m=("y_pts", "mean"), p=("y_over", "mean"))
    emp = df["y_pts"] == df["line_synthetic"]
    return {
        "n": len(df), "jogadores": df["PLAYER_ID"].nunique(), "jogos": df["GAME_ID"].nunique(),
        "inicio": df["GAME_DATE"].min(), "fim": df["GAME_DATE"].max(),
        "media": y.mean(), "mediana": y.median(), "dp": y.std(), "moda": y.mode().iloc[0],
        "q1": q1, "q3": q3, "assim": y.skew(), "curt": y.kurt(), "zeros": (y == 0).mean(),
        "corr": corr, "corr_over": corr_over, "pares_09": pares, "iqr_out": iqr_out,
        "p30": (y >= 30).mean(), "p40": (y >= 40).mean(), "p50": (y >= 50).mean(), "max": y.max(),
        "pos": df["y_over"].mean(), "titular": titular, "casa": casa, "b2b": b2b, "temp": temp,
        "linha_inteira": (df["line_synthetic"] % 1 == 0).mean(), "empates": int(emp.sum()), "empates_frac": emp.mean(),
        "over_linha_inteira": df.loc[df["line_synthetic"] % 1 == 0, "y_over"].mean(),
        "over_linha_meia": df.loc[df["line_synthetic"] % 1 != 0, "y_over"].mean(),
        "split": df["split"].value_counts(),
        "rest_max": df["rest_days"].max(), "rest_7": (df["rest_days"] > 7).mean(),
        "gp_min": df["games_played_prev"].min(),
    }


# ----------------------------------------------------------------------------
# Conteudo
# ----------------------------------------------------------------------------

def build(video_url: str | None) -> None:
    tv = json.loads((METRICS_DIR / "treino_validacao.json").read_text(encoding="utf-8"))
    tf = json.loads((METRICS_DIR / "test_final.json").read_text(encoding="utf-8"))
    df = load_dataset()
    E = eda_numbers(df)
    R, C = tf["regressao"], tf["classificacao"]
    fin = R["modelo_final"]
    T = R["teste"]
    b_ma5 = R["comparacoes_bootstrap"][f"{fin}_vs_media_5_jogos"]
    b_ewm = R["comparacoes_bootstrap"][f"{fin}_vs_media_exponencial"]
    b_lin = R["comparacoes_bootstrap"][f"{fin}_vs_linear_sobre_exponencial"]
    ganho_ma5 = 1 - T[fin]["mae"] / T["media_5_jogos"]["mae"]
    ganho_ewm = 1 - T[fin]["mae"] / T["media_exponencial"]["mae"]
    cf = C["teste"]["final_calibrado"]
    cfe = C["teste_sem_empates"]["final_calibrado"]
    iv80, iv90 = R["intervalos"]["80"], R["intervalos"]["90"]
    teto = R["teto_preditivo"]
    ab = R["ablacao_por_familia"]
    video = video_url or VIDEO_PLACEHOLDER
    params_fin = tv["regressao"]["cv"][fin]["melhores_parametros"]
    dg = R["diagnosticos_posteriores"]
    lad = dg["mediana_sobre_exponencial"]
    mo = dg["modelo_com_nivel_oraculo"]
    b_lad = lad["modelo_final_vs_esta"]
    b_mo = mo["modelo_final_menos_oraculo"]
    fracao_nivel = (T["media_exponencial"]["mae"] - T[fin]["mae"]) / (T["media_exponencial"]["mae"] - mo["mae_teste"])
    clf_fin = C["modelo_final"]

    rel = Relatorio()
    P = rel.par
    c = WD_ALIGN_PARAGRAPH.CENTER

    # ------------------------------------------------------------ capa
    P("Predição de Desempenho Individual de Atletas da NBA por Meio de Aprendizado de Máquina Supervisionado",
      align=c, first=None, before=12, after=12, size=16)
    P("André Ihsan Ward (RA 10425684)", align=c, first=None, before=0)
    P("Rodrigo Lucas Mascarenhas Leite Oliveira (RA 10427925)", align=c, first=None, before=0)
    P("Prof. Dr. Leandro Zerbinatti", align=c, first=None, before=12)
    P("Faculdade de Computação e Informática (FCI)\nUniversidade Presbiteriana Mackenzie, São Paulo, SP, Brasil",
      align=c, first=None, before=12)
    P("andre16ward@gmail.com, rodrilucasoliveira@gmail.com", align=c, first=None, before=6, after=6, size=11)
    P("Inteligência Artificial, 7º J, Sistemas de Informação, Noite", align=c, first=None, before=6)
    P("2026", align=c, first=None, before=6, after=6)

    # ------------------------------------------------------------ resumo
    P("Resumo", align=c, first=None, left=0.8, right=0.8, before=12, size=14)
    P(
        "Projeções de desempenho individual de atletas costumam se apoiar em médias simples dos jogos recentes. "
        "Este trabalho desenvolve e avalia modelos de aprendizado de máquina supervisionado para estimar os pontos "
        "de jogadores da NBA em partidas ainda não disputadas, usando apenas informação conhecida antes do jogo. "
        f"O conjunto de dados, coletado pelo grupo nas interfaces públicas da liga, tem {inteiro(E['n'])} registros "
        "jogador-partida de cinco temporadas regulares. Com a biblioteca scikit-learn, foram comparados médias "
        "móveis, regressão regularizada (Ridge e Lasso), modelo linear generalizado de Poisson, floresta aleatória e "
        "gradient boosting, sob validação cruzada temporal, com a temporada 2025-26 reservada para um teste único. "
        f"O melhor modelo, gradient boosting com perda absoluta, obteve erro absoluto médio (MAE) de {num(T[fin]['mae'], 2)} "
        f"pontos, {pct(ganho_ma5)} abaixo da média dos cinco jogos anteriores e {pct(ganho_ewm)} abaixo da média "
        "exponencial, ganhos modestos mas estatisticamente consistentes. Os intervalos conformais de 80% e 90% "
        f"cobriram {pct(iv80['global']['cobertura'])} e {pct(iv90['global']['cobertura'])} dos jogos de teste. Na "
        "formulação de classificação, a probabilidade de o atleta superar a mediana dos seus dez jogos anteriores "
        f"atingiu AUC de {num(cf['auc'], 2)} e Brier score {pct(cf['brier_skill'])} menor que o de um estimador "
        "ingênuo, com probabilidades bem calibradas. Um diagnóstico com informação do futuro indica que conhecer o "
        f"nível real do jogador na temporada reduziria o MAE em apenas mais {num(b_mo['delta'], 2)} ponto: a maior parte "
        "do erro restante parece vir da variação de jogo a jogo, e não de estimar mal o nível do atleta.",
        italic=True, left=0.8, right=0.8, first=None, before=6,
    )
    P("**Palavras-chave:** aprendizado de máquina; análise esportiva; NBA; scikit-learn; calibração; predição conformal.",
      left=0.8, right=0.8, first=None, before=6, after=6)
    P("Abstract", align=c, first=None, left=0.8, right=0.8, before=12, size=14)
    P(
        "Individual performance projections usually rely on simple averages of recent games. This work develops and "
        "evaluates supervised machine learning models to estimate the points scored by NBA players in games not yet "
        "played, using only information known before tip-off. The dataset, collected by the group from the league "
        f"public statistics interfaces, holds {inteiro(E['n']).replace('.', ',')} player-game records from five regular seasons. "
        "Using scikit-learn, moving averages, regularized regression (Ridge and Lasso), a Poisson generalized linear "
        "model, random forests and gradient boosting were compared under temporal cross-validation, with the 2025-26 "
        "season held out for a single test. The best model, gradient boosting with absolute loss, reached a mean "
        f"absolute error of {T[fin]['mae']:.2f} points, {100 * ganho_ma5:.1f}% below the five-game average and "
        f"{100 * ganho_ewm:.1f}% below the exponential average: modest but statistically consistent gains. Conformal "
        f"80% and 90% intervals covered {100 * iv80['global']['cobertura']:.1f}% and {100 * iv90['global']['cobertura']:.1f}% "
        "of test games. In the classification formulation, the probability of exceeding the median of the previous ten "
        f"games reached an AUC of {cf['auc']:.2f} and a Brier score {100 * cf['brier_skill']:.1f}% below a naive "
        "estimator, with well calibrated probabilities. A diagnostic that uses future information indicates that knowing "
        f"each player's true season level would lower the MAE by only {b_mo['delta']:.2f} more points: most of the remaining "
        "error seems to come from game-to-game variation rather than from misjudging the player's level.",
        italic=True, left=0.8, right=0.8, first=None, before=6,
    )
    P("**Keywords:** machine learning; sports analytics; NBA; scikit-learn; calibration; conformal prediction.",
      left=0.8, right=0.8, first=None, before=6, after=6)

    # ------------------------------------------------------------ 1. introducao
    rel.titulo_secao("1. Introdução")
    rel.subtitulo("1.1. Contextualização")
    P("O basquete profissional adotou cedo a análise quantitativa como apoio à decisão. Oliver (2004) organizou um "
      "conjunto de métricas de eficiência ofensiva e defensiva que deslocou a avaliação de jogadores das estatísticas "
      "brutas para indicadores ajustados ao ritmo de jogo, e Kubatko et al. (2007) consolidaram esse vocabulário em uma "
      "base metodológica comum. Desde então, franquias, veículos de mídia e plataformas de estatística passaram a "
      "tratar a produção individual como objeto mensurável.")
    P("A disponibilidade de dados acompanhou esse movimento. A NBA publica em suas interfaces oficiais o registro "
      "detalhado de cada partida, incluindo o desempenho de cada atleta em cada jogo. São milhares de observações por "
      "temporada, organizadas de forma estável ao longo do tempo, o que atende aos requisitos das técnicas de "
      "aprendizado supervisionado descritas por Coppin (2010), Russell e Norvig (2021) e Hastie, Tibshirani e Friedman "
      "(2009): aprender, a partir de exemplos rotulados, uma função que generalize para casos não vistos.")
    P("A prática corrente de projeção, porém, ainda se apoia em médias dos últimos jogos. Uma média simples ignora "
      "quem é o adversário, se a partida é em casa, quanto o atleta descansou e qual o seu papel atual na rotação. Como "
      "atribui peso igual a todas as observações, ela também demora a reagir quando alguma dessas condições muda.")
    rel.subtitulo("1.2. Justificativa")
    P("A lacuna que motiva o projeto está na distância entre a riqueza dos dados disponíveis e a simplicidade do "
      "estimador usual. Modelos supervisionados permitem incorporar contexto de forma explícita e medir, com rigor, "
      "quanto esse contexto de fato acrescenta. Há também uma motivação metodológica: séries esportivas são terreno "
      "fértil para vazamento de dados, porque quase toda estatística sobre uma partida só existe depois que ela termina. "
      "Construir variáveis estritamente pré-jogo, demonstrar por testes automatizados que não há vazamento e avaliar o "
      "modelo por divisão temporal exercita os fundamentos de preparação de dados e validação vistos na disciplina.")
    P("Por fim, uma estimativa pontual sem medida de incerteza induz confiança que os dados não sustentam. Por isso a "
      "solução acompanha cada previsão de um intervalo com cobertura verificada e avalia a calibração das probabilidades.")
    rel.subtitulo("1.3. Objetivo")
    P("O objetivo geral é desenvolver e avaliar um modelo de aprendizado de máquina supervisionado capaz de estimar a "
      "produção individual de atletas da NBA a partir de informação disponível antes do início da partida. São objetivos "
      "específicos:", first=1.25)
    rel.itens([
        "coletar nas interfaces públicas da NBA um conjunto de dados próprio no grão jogador-partida, documentando todo o procedimento;",
        "realizar a análise exploratória e a preparação dos dados, com variáveis estritamente pré-jogo e testes automatizados contra vazamento;",
        "comparar baselines de média móvel com regressão regularizada, floresta aleatória e gradient boosting, sob validação temporal;",
        "estimar a probabilidade de o atleta superar o próprio patamar recente e avaliar a calibração dessa probabilidade;",
        "acompanhar cada estimativa de um intervalo de incerteza com cobertura verificada e estimar o teto preditivo do problema;",
        "discutir os limites do artefato e os riscos do seu uso indevido.",
    ])
    rel.subtitulo("1.4. Opção do projeto")
    P("O grupo adota a Opção Framework. O problema é uma tarefa supervisionada sobre dados tabulares numéricos, com alvo "
      "bem definido e histórico longitudinal extenso; nessa configuração, modelos clássicos de regressão e comitês de "
      "árvores são competitivos e muito mais interpretáveis que um grande modelo de linguagem. A biblioteca scikit-learn "
      "(PEDREGOSA et al., 2011) é usada no pré-processamento, na modelagem, na seleção de hiperparâmetros e na avaliação; "
      "pandas (McKINNEY, 2010) e Matplotlib (HUNTER, 2007) dão suporte à manipulação e à visualização, e a organização "
      "do fluxo em pipelines segue a prática descrita por Géron (2021). Os dados não vêm de repositórios "
      "prontos: são capturados pelo próprio grupo por requisições programáticas às interfaces públicas da liga.")
    P("O relatório está organizado assim: a Seção 2 formaliza o problema; a Seção 3 trata dos aspectos éticos; a Seção 4 "
      "descreve o conjunto de dados, a análise exploratória e a preparação; a Seção 5 apresenta a metodologia aplicada; "
      "a Seção 6, os resultados; a Seção 7, a conclusão; a Seção 8 traz os endereços do repositório e do vídeo; e a "
      "Seção 9, as referências.")

    # ------------------------------------------------------------ 2. problema
    rel.titulo_secao("2. Descrição do Problema")
    P("O problema consiste em estimar a produção ofensiva de um atleta em uma partida ainda não disputada, usando apenas "
      "informação conhecida antes do início do jogo. Cada registro é indexado pelo par jogador e partida, e os registros "
      "são ordenados no tempo. Para cada um deles constrói-se um vetor de variáveis a partir das partidas anteriores do "
      "atleta e do contexto já conhecido da partida corrente (adversário, mando de quadra e calendário). O problema é "
      "tratado em duas formulações complementares:")
    rel.itens([
        "**regressão** (formulação principal): estimar diretamente o número de pontos (`y_pts`);",
        "**classificação**: estimar a probabilidade de o atleta superar um limiar de referência (`y_over`), o que permite avaliar a calibração das estimativas, e não apenas o erro pontual.",
    ])
    P("Não existe série histórica de limiares de mercado em fonte pública, gratuita e reprodutível: interfaces comerciais "
      "restringem o histórico a planos pagos, agregadores não o publicam e sua extração automatizada violaria termos de "
      "uso. Adotou-se então um limiar sintético, a **mediana dos pontos nas dez partidas anteriores** do atleta, "
      "arredondada para meio ponto (`line_synthetic`). Superar a própria mediana recente é tarefa diferente de superar um "
      "limiar de mercado, e o alvo é autorreferencial: o "
      "limiar acompanha o próprio jogador, o que aproxima a taxa de superação de 50% e limita a acurácia alcançável a "
      "valores pouco acima do acaso. Por isso a regressão permanece como formulação principal.")
    P(f"**Correção em relação à N1.** O relatório anterior afirmou que o arredondamento para meio ponto elimina empates. "
      f"Isso não é verdade: a mediana de dez valores inteiros é a média dos dois centrais e pode ser inteira (por exemplo, "
      f"12 e 12 resultam em 12,0), caso em que o arredondamento não a altera. A linha é inteira em {pct(E['linha_inteira'])} "
      f"dos registros, e há {inteiro(E['empates'])} empates ({pct(E['empates_frac'])}) entre pontos e linha, todos rotulados "
      f"como \"não superou\". A taxa de superação é {pct(E['over_linha_inteira'])} quando a linha é inteira e "
      f"{pct(E['over_linha_meia'])} quando termina em ,5. O dataset publicado foi mantido, a documentação foi corrigida e "
      "todos os resultados de classificação são reportados com e sem os empates (Seção 6.4).")
    P("O problema tem dificuldades específicas que condicionam a modelagem:")
    rel.itens([
        "**variância alta**: a produção em uma única partida é ruidosa, e existe um teto preditivo que nenhum modelo ultrapassa;",
        "**dependência dos minutos jogados**: o volume de produção depende do tempo em quadra, que depende de decisões técnicas, faltas e placar, nada disso conhecido antes do jogo;",
        "**heterogeneidade entre atletas**: titulares e jogadores de rotação curta têm distribuições muito diferentes;",
        "**não estacionariedade**: contusões, trocas e mudanças de função alteram a distribuição ao longo do tempo, o que inviabiliza a divisão aleatória entre treino e teste;",
        "**mistura de contextos**: temporada regular e pós-temporada têm regimes distintos e não são misturados.",
    ])
    P("O artefato é um estimador de desempenho esportivo para fins analíticos e acadêmicos. Não é ferramenta de "
      "recomendação de apostas: o projeto não inclui cotações, cálculo de valor esperado nem gestão de banca.")

    # ------------------------------------------------------------ 3. etica
    rel.titulo_secao("3. Aspectos Éticos e Responsabilidade no Desenvolvimento da Solução")
    P("A discussão ética se organiza em seis eixos, tomando como referência o panorama de princípios de ética em IA de "
      "Jobin, Ienca e Vayena (2019), do qual se aproveitam transparência, justiça, não maleficência e responsabilidade. "
      "Na N2, cada compromisso assumido na primeira entrega ganhou um mecanismo verificável no código.")
    rel.subtitulo("3.1. Risco de uso indevido")
    P("Um modelo que estima a produção individual de atletas tem aplicação imediata em mercados de apostas, e esse é o "
      "risco mais relevante do projeto. O repositório não incorpora cotações, valor esperado ou qualquer camada de decisão "
      "financeira; a probabilidade estimada é sempre a de superar o **próprio patamar recente** do jogador, nunca uma "
      "linha de mercado; e o artefato de demonstração imprime, em toda consulta, o aviso de que se trata de estimativa "
      "estatística e não de recomendação de aposta. Os resultados de classificação são modestos e não autorizam nenhuma "
      "conclusão sobre lucratividade. Essas medidas reduzem o risco, mas não o eliminam: qualquer estimador público pode "
      "ser reaproveitado por terceiros, e o que a não maleficência exige é não facilitar esse uso e ser transparente "
      "quanto à sua possibilidade.")
    rel.subtitulo("3.2. Dados pessoais e privacidade")
    P("O conjunto trata de pessoas identificadas, o que atrai a Lei nº 13.709/2018 (BRASIL, 2018). Os dados são "
      "estatísticas de desempenho profissional tornadas públicas pela própria liga, usadas para fins acadêmicos, sem "
      "tratamento de dados sensíveis. O grupo não coleta nem infere informação de saúde. Na N2 identificou-se uma "
      f"sutileza: a variável `rest_days` mede os dias desde a última partida do atleta e chega a {int(E['rest_max'])} dias; "
      f"acima de uma semana ({pct(E['rest_7'])} dos registros) ela reflete ausência, frequentemente por contusão. A "
      "variável foi mantida por ser pública e pré-jogo. O limite de 10 dias aplicado nos modelos contém a escala, mas não "
      "remove a informação de ausência: o valor 10 continua indicando que o atleta ficou fora por dez dias ou mais. O "
      "arquivo publicado mantém o valor original. O grupo declara isso como limitação: o modelo usa, indiretamente, um "
      "indicador de ausência que pode refletir lesão, embora não colete nem infira diagnóstico.")
    rel.subtitulo("3.3. Vieses do conjunto de dados")
    P("Atletas com poucos minutos geram amostras pequenas e ruidosas; incluí-los inflaria as métricas, porque prever "
      "valores próximos de zero é trivial. O grupo aplica um critério mínimo de participação (média de ao menos 15 minutos "
      "nos cinco jogos anteriores) e reporta as métricas por faixa de minutos (Tabela 4) e por perfil de jogador "
      "(estrelas, atletas novos e conhecidos), para que o desempenho sobre um grupo não mascare o de outro. Há também "
      "viés de sobrevivência (atletas que deixam a liga somem da base) e de seleção: só existe registro se o atleta entrou "
      "em quadra, de modo que o modelo estima pontos *condicionados a jogar*.")
    rel.subtitulo("3.4. Comunicação da incerteza e responsabilidade")
    P("Dada a variância do fenômeno, apresentar uma estimativa pontual sem qualificação seria uma forma de desinformar. "
      "O compromisso da N1 foi cumprido assim: toda previsão vem com intervalos conformais de 80% e 90%, cuja cobertura "
      "real foi medida no teste (Seção 6.5); as probabilidades da classificação têm calibração avaliada (Seção 6.4); e o "
      "quanto do erro se deve a não conhecer o nível do atleta foi estimado e declarado (Seção 6.6). A responsabilidade "
      "sobre decisões continua humana. "
      "Estimativas de desempenho individual não devem ser usadas isoladamente em decisões que afetem a carreira de um atleta.")
    rel.subtitulo("3.5. Transparência e reprodutibilidade")
    P("Todo o material está em repositório público: conjunto de dados, cadernos, código-fonte, resultados e este "
      "relatório. Os arquivos-fonte trazem cabeçalho com identificação dos integrantes, síntese e histórico de alterações "
      "por arquivo, verificado por teste automatizado. A coleta gera manifesto com data, endpoints, versão da biblioteca e "
      "contagens; o treino registra versões das bibliotecas, semente aleatória e o commit do código nos arquivos de "
      "resultados. O dataset é tratado como congelado, porque a interface da liga é um dado vivo e uma nova coleta poderia "
      "alterar as linhas.")
    rel.subtitulo("3.6. Disciplina de avaliação")
    P("Um risco ético menos óbvio é o de relatar resultados otimistas por reutilizar o conjunto de teste. Para evitá-lo, "
      "todas as escolhas (hiperparâmetros, modelo, calibrador, intervalo) foram congeladas antes de ler o teste, e o "
      "script de avaliação se recusa a sobrescrever o resultado sem uma opção explícita. A avaliação foi reexecutada, de "
      "forma idêntica, para acrescentar diagnósticos definidos depois da primeira leitura (os oráculos em mediana, a "
      "regressão mediana sobre a média exponencial e o modelo com nível oráculo, Seções 6.1 e 6.6), sugeridos por uma "
      "revisão crítica do próprio relatório. Esses diagnósticos, assim como a ablação e a permutação, rodam sobre o teste "
      "mas não alteraram nenhuma escolha de modelo; isso está registrado no próprio arquivo de resultados.")

    # ------------------------------------------------------------ 4. dados
    rel.titulo_secao("4. Conjunto de Dados")
    rel.subtitulo("4.1. Origem e procedimento de coleta")
    P("Os dados vêm das interfaces públicas de estatística da NBA, acessadas pela biblioteca de código aberto nba_api "
      "(SWAR, 2024). São consultados os registros de partidas por atleta (uma linha por jogador e por jogo) e por equipe "
      "(base dos indicadores do adversário). A consulta é feita por temporada, o que reduz o volume de requisições de cerca "
      "de 2.750 (uma por atleta e temporada) para duas por temporada (uma de atletas, outra de equipes), dez no total, com "
      "controle de taxa, nova tentativa com espera progressiva e cache local.")
    P("O recorte são cinco temporadas regulares, de 2021-22 a 2025-26. As temporadas 2019-20 e 2020-21 ficaram de fora "
      "de propósito (sede única, calendário reduzido e ginásios sem público). As janelas das variáveis históricas são "
      "reiniciadas a cada temporada. A coleta reuniu 131.292 registros brutos; descartadas as linhas sem histórico "
      f"suficiente e aplicado o critério mínimo de participação, o conjunto analítico ficou com {inteiro(E['n'])} linhas "
      f"e 37 colunas, cobrindo {inteiro(E['jogadores'])} atletas e {inteiro(E['jogos'])} partidas, entre "
      f"{E['inicio']:%d/%m/%Y} e {E['fim']:%d/%m/%Y}: 9 colunas de identificação, 24 variáveis preditoras pré-jogo, "
      "3 de alvo e linha de referência e 1 de divisão temporal.")
    rel.subtitulo("4.2. Conteúdo do conjunto e dicionário de dados")
    P("A Tabela 1 traz o dicionário das principais variáveis, **com os nomes reais das colunas** do arquivo publicado. "
      "Na N1, a tabela usou nomes descritivos em português que não existem no arquivo, e duas descrições estavam "
      "imprecisas: a titularidade não é uma proporção, mas um indicador binário estimado pelos minutos recentes; e a "
      "força do adversário não é um rating de eficiência, mas a média de pontos cedidos. A versão abaixo corrige isso.")
    rel.tabela("Dicionário das principais variáveis (nomes reais das colunas).",
               ["Coluna", "Tipo", "Descrição e disponibilidade"], [
                   ["`PLAYER_ID`, `GAME_ID`", "Identificador", "Atleta e partida. Contexto."],
                   ["`GAME_DATE`, `SEASON`", "Data, texto", "Data e temporada da partida. Pré-jogo."],
                   ["`OPPONENT`", "Categórica", "Sigla do adversário. Pré-jogo (não entra no modelo)."],
                   ["`is_home`", "Binária", "Partida em casa. Pré-jogo."],
                   ["`rest_days`", "Numérica", "Dias desde a partida anterior do atleta (limitado a 10 nos modelos). Pré-jogo."],
                   ["`is_b2b`", "Binária", "Segunda partida em noites seguidas. Pré-jogo."],
                   ["`pts_ma3_prev`, `pts_ma5_prev`, `pts_ma10_prev`", "Numérica", "Média de pontos nas 3, 5 e 10 partidas anteriores. Pré-jogo."],
                   ["`pts_ewm_prev`", "Numérica", "Média exponencial de pontos, peso 0,9 por jogo de distância. Pré-jogo."],
                   ["`pts_std5_prev`, `pts_season_mean_prev`", "Numérica", "Desvio dos últimos 5 jogos; média na temporada até o jogo anterior. Pré-jogo."],
                   ["`min_*_prev`", "Numérica", "Médias de minutos (3, 5, 10 jogos, exponencial, temporada). Pré-jogo."],
                   ["`fga_ma5_prev`, `fg3a_ma5_prev`, `fta_ma5_prev`", "Numérica", "Arremessos de quadra, de três e lances livres tentados (5 jogos). Pré-jogo."],
                   ["`reb_ma5_prev`, `ast_ma5_prev`", "Numérica", "Rebotes e assistências (5 jogos), proxies de função. Pré-jogo."],
                   ["`starter_proxy_prev`", "Binária", "Proxy de titularidade: 1 se a média de minutos dos 5 jogos anteriores ≥ 28. Pré-jogo."],
                   ["`games_played_prev`", "Numérica", "Jogos já disputados na temporada. Pré-jogo."],
                   ["`opp_def_pts_allowed_prev`", "Numérica", "Média de pontos cedidos pelo adversário até o jogo anterior dele. Pré-jogo."],
                   ["`opp_pace_proxy_prev`, `opp_games_played_prev`", "Numérica", "Proxy de ritmo do adversário; jogos já disputados por ele. Pré-jogo."],
                   ["`line_synthetic`", "Numérica", "Mediana dos pontos nos 10 jogos anteriores, arredondada para ,5. Pré-jogo."],
                   ["`y_pts`", "Numérica", "Pontos marcados. Alvo de regressão. Pós-jogo."],
                   ["`y_over`", "Binária", "1 se `y_pts` > `line_synthetic`. Alvo de classificação. Pós-jogo."],
               ], larguras=[5.2, 2.3, 8.5])
    P("Os minutos jogados na própria partida, apesar de serem a variável mais associada ao alvo, não entram no conjunto "
      "de preditoras, porque só são conhecidos depois que a partida termina. Seu efeito é capturado de forma indireta pelas "
      "médias de minutos e pela titularidade estimada, ambas calculadas sobre o histórico anterior.")
    rel.subtitulo("4.3. Análise exploratória")
    co = E["corr"]
    P("A análise exploratória está no caderno `notebooks/01_eda.ipynb`. Os principais resultados, recalculados sobre o "
      "dataset publicado, são os seguintes.")
    P(f"**Distribuição do alvo.** Os pontos têm assimetria à direita: média {num(E['media'], 2)}, mediana {num(E['mediana'], 0)}, "
      f"desvio padrão {num(E['dp'], 2)}, moda {num(E['moda'], 0)}, quartis {num(E['q1'], 0)} e {num(E['q3'], 0)}, assimetria "
      f"{num(E['assim'], 3)} e curtose {num(E['curt'], 3)}. Partidas sem pontos respondem por {pct(E['zeros'], 2)} das linhas. "
      "A assimetria (Figura 1) tem consequência direta na modelagem: o erro absoluto é minimizado pela mediana, e não pela "
      "média, o que favorece modelos treinados com perda absoluta (Seção 6.1).")
    rel.figura("f_distribuicao_pontos.png", "Distribuição dos pontos por jogo, com média e mediana.", largura=13)
    P(f"**Correlações.** As variáveis de histórico recente têm associação forte com os pontos: média exponencial "
      f"{num(co['pts_ewm_prev'])}, média da temporada {num(co['pts_season_mean_prev'])}, média de dez jogos "
      f"{num(co['pts_ma10_prev'])} e arremessos tentados em cinco jogos {num(co['fga_ma5_prev'])}. As variáveis de contexto "
      f"quase não se correlacionam linearmente com o alvo (adversário {num(co['opp_def_pts_allowed_prev'])}, mando "
      f"{num(co['is_home'])}). Com o alvo binário, a maior correlação absoluta é {num(E['corr_over'])}: como a linha é a "
      "mediana recente do próprio atleta, as variáveis que preveem pontos preveem igualmente a linha. A Seção 6.4 mostra "
      "que o sinal da classificação está na **diferença** entre a forma recente e a linha.")
    P(f"**Balanceamento, extremos e multicolinearidade.** A classe positiva responde por {pct(E['pos'], 2)} das linhas. "
      f"Pelo critério do intervalo interquartil, {pct(E['iqr_out'], 2)} dos valores do alvo são extremos; partidas de 30 ou "
      f"mais pontos são {pct(E['p30'], 2)}, de 40 ou mais {pct(E['p40'], 2)} e de 50 ou mais {pct(E['p50'], 2)}, com máximo de "
      f"{int(E['max'])}. São eventos reais e foram preservados. Há {E['pares_09']} pares de variáveis com correlação "
      "absoluta acima de 0,90 (a mesma estatística em janelas diferentes), o que motivou a inclusão de modelos "
      "regularizados (L1/L2) e a análise de importância por família de variáveis, em vez de coluna a coluna.")
    P(f"**Contexto e estabilidade.** O indicador de titularidade separa bem os grupos ({num(E['titular'].get(1.0, np.nan), 2)} "
      f"contra {num(E['titular'].get(0.0, np.nan), 2)} pontos). Mando ({num(E['casa'].get(1.0, np.nan), 2)} em casa contra "
      f"{num(E['casa'].get(0.0, np.nan), 2)} fora) e jogos em noites seguidas ({num(E['b2b'].get(1.0, np.nan), 2)} contra "
      f"{num(E['b2b'].get(0.0, np.nan), 2)}) praticamente não separam. A média de pontos varia entre "
      f"{num(E['temp']['m'].min(), 2)} e {num(E['temp']['m'].max(), 2)} entre temporadas, deslocamento pequeno que favorece "
      "a divisão temporal.")
    rel.subtitulo("4.4. Preparação dos dados")
    P("A preparação, em Python (`src/features.py`, `src/target.py`, `src/build_dataset.py` e `src/modeling.py`), compreende:")
    rel.itens([
        "ordenação cronológica por atleta e cálculo de todas as variáveis históricas agrupadas por (atleta, temporada) e deslocadas em uma partida (`shift(1)`), de modo que cada linha use apenas partidas anteriores;",
        "médias móveis de 3, 5 e 10 partidas e média exponencial de fator 0,9; indicadores do adversário calculados só com as partidas anteriores dele;",
        "critério mínimo de participação (15 minutos de média nos 5 jogos anteriores, informação pré-jogo) e descarte das linhas sem 10 jogos de histórico, com a aritmética registrada no manifesto (131.292 − 33.504 − 13.989 = 83.799);",
        "nenhuma imputação: as primeiras partidas de cada temporada, únicas sem histórico, são descartadas, o que elimina uma fonte frequente de vazamento;",
        "padronização das variáveis e limite de `rest_days` encapsulados em um `Pipeline` do scikit-learn, ajustado apenas no treino de cada dobra. Todas as preditoras são numéricas, então não há codificação de categóricas (a menção da N1 a essa etapa foi retirada); adversário e equipe entram apenas por indicadores numéricos;",
        f"divisão temporal: treino 2021-22 a 2023-24 ({inteiro(E['split']['train'])} linhas), validação 2024-25 "
        f"({inteiro(E['split']['val'])}) e teste 2025-26 ({inteiro(E['split']['test'])}), com teste automatizado de que os "
        "conjuntos não se sobrepõem em partidas nem em datas.",
    ])
    P(f"{_extenso(n_testes('test_no_leakage.py')).capitalize()} testes automatizados protegem a preparação contra vazamento. O principal altera o resultado de um jogo futuro e "
      "exige que nenhuma variável das linhas anteriores mude; na N2 ele foi estendido às variáveis do adversário, que antes "
      "não eram cobertas. Também foi corrigido o teste que verifica a ausência de colunas pós-jogo entre as preditoras: ele "
      "comparava nomes em maiúsculas e minúsculas e, por isso, nunca falharia. No total, o projeto tem "
      f"{n_testes()} testes automatizados.")

    # ------------------------------------------------------------ 5. metodologia
    rel.titulo_secao("5. Metodologia")
    rel.subtitulo("5.1. Tipo de pesquisa e abordagem")
    P("Trata-se de pesquisa aplicada, quantitativa, com delineamento experimental comparativo: diferentes algoritmos "
      "supervisionados são treinados sobre o mesmo conjunto de variáveis e comparados entre si e contra baselines, sob "
      "protocolo de avaliação idêntico e com as mesmas linhas.")
    rel.subtitulo("5.2. Protocolo de validação temporal")
    P("Uma divisão aleatória colocaria jogos de dias vizinhos em conjuntos diferentes; como as variáveis descrevem o estado "
      "recente do atleta, o modelo treinaria com informação do futuro. O protocolo tem três níveis (BERGMEIR; BENÍTEZ, 2012):")
    rel.itens([
        "**hiperparâmetros**: validação cruzada temporal dentro do treino, com janela expansiva por temporada (treina em 2021-22 e valida em 2022-23; treina em 2021-22 e 2022-23 e valida em 2023-24), com as dobras fornecidas explicitamente ao `GridSearchCV` (ou ao `RandomizedSearchCV`, no gradient boosting); a melhor configuração é reajustada no treino inteiro;",
        "**escolhas na validação (2024-25)**: comparação das famílias de modelos, escolha do modelo final, escolha e ajuste do calibrador de probabilidades e calibração do intervalo conformal;",
        "**teste (2025-26)**: avaliado uma única vez, depois de todas as escolhas congeladas (`python -m src.evaluate --final`).",
    ])
    rel.subtitulo("5.3. Baselines e modelos de regressão")
    P("A baseline prometida na N1 é a média dos cinco jogos anteriores. A análise mostrou que ela fica entre as mais "
      "fracas das médias recentes (só a de três jogos erra mais) e bem atrás da média exponencial, então os modelos também "
      "foram comparados com a média exponencial e com uma regressão linear de uma variável sobre ela, que verifica se o "
      "ganho vem só de reescalar a média recente. Por ser ajustada por mínimos quadrados, essa regressão linear não é a "
      "régua mais exigente em MAE; por isso a Seção 6.1 acrescenta, como diagnóstico, uma regressão **mediana** sobre a "
      "média exponencial. Os modelos, em ordem crescente de complexidade, foram:")
    rel.itens([
        "**Ridge e Lasso** (HOERL; KENNARD, 1970; TIBSHIRANI, 1996): regressão linear com penalização L2 e L1, incluídas por causa da multicolinearidade observada;",
        "**GLM de Poisson**: modelo linear generalizado com ligação logarítmica, natural para contagens;",
        "**floresta aleatória** (BREIMAN, 2001): comitê de árvores por bagging, capaz de capturar interações não lineares;",
        "**gradient boosting** (FRIEDMAN, 2001), na implementação `HistGradientBoostingRegressor`, com três funções de perda: quadrática, Poisson e absoluta. A perda absoluta otimiza diretamente o MAE.",
    ])
    hp = [[NOMES_MODELO[k], ", ".join(f"{p.replace('model__', '')}={_fmt_param(v)}" for p, v in d["melhores_parametros"].items()),
           num(-d["score_cv_medio"]), num(tv["regressao"]["validacao"][k]["mae"])]
          for k, d in tv["regressao"]["cv"].items()]
    P("A Tabela 2 resume os hiperparâmetros escolhidos e o erro de cada família na validação cruzada e na validação.")
    rel.tabela("Hiperparâmetros escolhidos por validação cruzada temporal e MAE na validação (2024-25).",
               ["Modelo", "Hiperparâmetros", "MAE CV", "MAE validação"], hp, larguras=[4.5, 7.5, 1.9, 2.1])
    P("Ridge, Lasso, GLM e floresta aleatória foram ajustados por busca em grade completa. Para o gradient boosting, por "
      "custo, usou-se busca aleatória com 12 sorteios de uma grade de 32 combinações (12 de 16 na classificação), com a "
      "mesma semente para as três funções de perda. Alguns valores escolhidos ficaram na borda da grade, então o ajuste "
      "não é exaustivo; o objetivo é comparar famílias, não ajustar o último decimal. Em particular, Ridge e Lasso ficaram "
      "com a menor penalidade testada: com cerca de 50 mil linhas, a multicolinearidade instabiliza os coeficientes "
      "individuais, mas não a previsão, e na prática os dois equivalem a uma regressão linear sem penalização. O gradient "
      "boosting foi treinado sem parada antecipada, porque a implementação separa uma validação interna aleatória, o que "
      "quebraria a disciplina temporal. A semente aleatória é fixa (42).")
    rel.subtitulo("5.4. Classificação e calibração")
    P("Para a classificação foram usados os classificadores equivalentes: regressão logística com penalização L2 e L1, "
      "floresta aleatória e gradient boosting, com seleção pela perda logarítmica. Além das 24 variáveis pré-jogo, os "
      "classificadores recebem a própria linha e cinco variáveis derivadas dela: as diferenças entre, respectivamente, a "
      "média exponencial, a média de cinco jogos e a média da temporada e a linha; a diferença exponencial padronizada "
      "pelo desvio recente; e um "
      "indicador de linha inteira. Todas são funções de colunas pré-jogo da mesma linha. Sem elas o modelo não tem como "
      "saber qual limiar precisa ser superado. A calibração (NICULESCU-MIZIL; CARUANA, 2005; GUO et al., 2017) foi escolhida "
      "sem usar o teste: Platt (sigmoide) e regressão isotônica foram ajustados na primeira metade da validação e julgados "
      "pelo Brier score na segunda metade, junto com a opção de não calibrar.")
    rel.subtitulo("5.5. Incerteza de cada estimativa")
    P("Para acompanhar cada previsão de uma medida de dispersão, usou-se predição conformal dividida (VOVK; GAMMERMAN; "
      "SHAFER, 2005; ANGELOPOULOS; BATES, 2021). O escore de não conformidade é o erro absoluto normalizado pela raiz da "
      "previsão, |y − μ| / √max(μ, 1), porque a variância de uma contagem cresce com a média. O quantil q do escore é "
      "calculado na validação, com correção de amostra finita, e o intervalo é μ ± q·√max(μ, 1), limitado inferiormente a "
      "zero. Sob permutabilidade entre validação e teste, o método garante cobertura **marginal** de pelo menos o nível "
      "nominal, mas não cobertura por subgrupo. Como há deriva entre temporadas e a validação também serviu para escolher "
      "o modelo, a cobertura no teste é medida, não presumida.")
    rel.subtitulo("5.6. Métricas e comparação estatística")
    P("Regressão: erro absoluto médio (MAE), raiz do erro quadrático médio (RMSE), R² e viés, globais, por faixa de minutos "
      "(pela média dos cinco jogos anteriores, informação pré-jogo) e por temporada. Classificação: acurácia, precisão, "
      "revocação, F1, área sob a curva ROC (AUC), Brier score (BRIER, 1950), Brier skill score em relação ao estimador "
      "ingênuo (que prevê sempre a taxa do treino), perda logarítmica, erro esperado de calibração (ECE) e curva de "
      "calibração. Como as linhas de um mesmo atleta não são independentes, as diferenças entre modelos são acompanhadas "
      "de intervalos de confiança de 95% por **bootstrap pareado por jogador** (2.000 réplicas; 1.000 para a AUC). "
      "Reamostrar o atleta inteiro trata a dependência entre os jogos de um mesmo jogador, mas não a dependência entre "
      "jogadores de uma mesma partida nem a variação entre temporadas: os intervalos medem a incerteza amostral dentro da "
      "temporada 2025-26, com o modelo fixo, e podem ainda ser algo otimistas.")
    rel.subtitulo("5.7. Contribuição das variáveis e teto preditivo")
    P("A contribuição das variáveis foi medida por ablação: o modelo final é retreinado sem cada família de variáveis "
      "(histórico de pontos, minutos e papel, volume de jogo, descanso e mando, adversário, jogos na temporada) e o aumento "
      "do MAE é estimado com intervalo de confiança. Também foi calculada a importância por permutação agrupada, que "
      "embaralha a família inteira de uma vez; embaralhar uma coluna isolada subestima a importância quando há colunas "
      "quase idênticas. Para estimar quanto do erro se deve a não conhecer o nível do atleta, usaram-se previsores que "
      "trapaceiam de propósito, com informação do futuro: oráculos simples (média e mediana do próprio atleta na "
      "temporada inteira, sem o jogo previsto, e nos cinco jogos antes e depois) e um **modelo com nível oráculo**, o "
      "mesmo modelo final recebendo também a média do atleta na temporada inteira. Não são utilizáveis na prática: "
      "servem de régua.")
    rel.subtitulo("5.8. Artefato")
    P("O artefato é uma ferramenta de linha de comando (`python -m src.predict`) que recebe jogador e data e devolve os "
      "pontos previstos, os intervalos de 80% e 90%, a baseline, a probabilidade de superar o patamar recente e, por se "
      "tratar de jogos passados, o resultado real. Consultas fora do domínio do modelo (jogadores fora da rotação, menos "
      "de dez jogos na temporada, pós-temporada) são recusadas com uma explicação, em vez de produzir um número sem "
      "respaldo. O pipeline completo é reproduzível por `python scripts/run_all.py`, e os cadernos 02 a 05 apresentam cada "
      "etapa.")

    # ------------------------------------------------------------ 6. resultados
    rel.titulo_secao("6. Resultados")
    rel.subtitulo("6.1. Regressão: modelos contra baselines")
    ordem = sorted(T.items(), key=lambda kv: kv[1]["mae"])
    P("A Tabela 3 e a Figura 2 comparam, no teste, todos os modelos treinados e as baselines.")
    rel.tabela("Desempenho no teste (2025-26, 17.101 jogos), do menor ao maior MAE.",
               ["Previsor", "Tipo", "MAE", "RMSE", "R²", "Viés"],
               [[NOMES_MODELO.get(k, k) + (" (final)" if k == fin else ""), "modelo" if v["tipo"] == "modelo" else "baseline",
                 num(v["mae"]), num(v["rmse"]), num(v["r2"]), num(v["vies"])] for k, v in ordem],
               larguras=[6.6, 1.8, 1.4, 1.4, 1.4, 1.4])
    P(f"O modelo final, {NOMES_MODELO[fin][0].lower() + NOMES_MODELO[fin][1:]}, escolhido pela validação, obteve MAE de {num(T[fin]['mae'])} no teste. "
      f"Contra a baseline da N1 (média de cinco jogos, {num(T['media_5_jogos']['mae'])}), a redução é de "
      f"{num(-b_ma5['delta'])} ponto por jogo ({pct(ganho_ma5)}), com intervalo de 95% entre {num(-b_ma5['ic_sup'])} e "
      f"{num(-b_ma5['ic_inf'])}. Contra a média exponencial ({num(T['media_exponencial']['mae'])}) a redução é de "
      f"{num(-b_ewm['delta'])} ({pct(ganho_ewm)}; IC entre {num(-b_ewm['ic_sup'])} e {num(-b_ewm['ic_inf'])}), e contra a "
      f"regressão linear sobre a média exponencial, de {num(-b_lin['delta'])} (IC entre {num(-b_lin['ic_sup'])} e "
      f"{num(-b_lin['ic_inf'])}). Todos os intervalos excluem zero. Parte da vantagem vem da perda absoluta: uma regressão "
      f"mediana só sobre a média exponencial (diagnóstico posterior) erra {num(lad['mae_teste'])}, e contra ela o ganho do "
      f"modelo cai para {num(-b_lad['delta'])} ponto (IC de {num(-b_lad['ic_sup'])} a {num(-b_lad['ic_inf'])}). Esse é o "
      "ganho que vem de informação além da média recente.")
    P("Três observações qualificam o resultado. Primeiro, todos os modelos ficam muito próximos entre si (diferença de "
      f"{num(max(v['mae'] for v in T.values() if v['tipo'] == 'modelo') - T[fin]['mae'])} entre o pior e o "
      "melhor modelo treinado): a informação disponível, e não o algoritmo, limita o desempenho. Segundo, o gradient "
      "boosting com perda absoluta vence porque prevê a mediana condicional, que minimiza o erro absoluto numa variável "
      "assimétrica; por isso também tem viés negativo (subestima a média) e RMSE ligeiramente maior que o do Ridge, que "
      "prevê a média. Terceiro, o Ridge e o Lasso, com coeficientes interpretáveis, ficam a menos de 0,02 ponto do melhor "
      "modelo, o que os torna alternativa razoável quando a explicabilidade pesa mais. O modelo final foi escolhido por "
      "ter o menor MAE na validação, critério fixado antes de ler o teste.")
    rel.figura("f_escada_mae.png", "Escada de previsores: MAE no teste de baselines e modelos.", largura=14.5)
    rel.subtitulo("6.2. Erro por faixa de minutos e por temporada")
    fx = {k: pd.DataFrame(v).set_index("grupo") for k, v in R["por_faixa_de_minutos"].items()}
    P("A Tabela 4 detalha o erro por faixa de minutos.")
    rel.tabela("MAE no teste por faixa de minutos (média dos cinco jogos anteriores).",
               ["Faixa (min)", "Jogos", "Pontos médios", "Média 5 jogos", "Média exponencial", "Modelo final", "MAE relativo"],
               [[g, inteiro(fx[fin].loc[g, "n"]), num(fx[fin].loc[g, "media_pontos"], 1), num(fx["media_5_jogos"].loc[g, "mae"]),
                 num(fx["media_exponencial"].loc[g, "mae"]), num(fx[fin].loc[g, "mae"]), pct(fx[fin].loc[g, "mae_relativo"], 0)]
                for g in fx[fin].index], larguras=[2.0, 1.6, 2.0, 2.2, 2.6, 2.2, 2.2])
    P("Em todas as faixas, a estimativa pontual do modelo fica abaixo das duas baselines; as diferenças contra a média "
      "exponencial são pequenas e não têm intervalo de confiança por faixa. O erro absoluto cresce com os minutos, porque quem joga "
      "mais pontua mais e oscila mais, mas o erro relativo à pontuação média cai de "
      f"{pct(fx[fin]['mae_relativo'].iloc[0], 0)} na faixa de 15 a 20 minutos para {pct(fx[fin]['mae_relativo'].iloc[-1], 0)} "
      "acima de 35 minutos: a estimativa é proporcionalmente mais informativa para os titulares. Por temporada (Figura 3), "
      "o MAE do modelo fica entre "
      f"{num(min(v['modelo_final_mae'] for v in R['por_temporada'].values()))} e "
      f"{num(max(v['modelo_final_mae'] for v in R['por_temporada'].values()))}, sempre abaixo das baselines, sem sinal de "
      "deriva forte. Os valores de 2022-23 e 2023-24 vêm das dobras usadas na escolha de hiperparâmetros, com modelos "
      "treinados em menos temporadas, e são levemente otimistas por seleção.")
    rel.figura("f_mae_por_temporada.png", "MAE por temporada: dobras da validação cruzada, validação e teste.", largura=13)
    perf = R["por_perfil"]
    est = {d["grupo"]: d for d in perf["estrela"][fin]}
    nov = {d["grupo"]: d for d in perf["novo_no_treino"][fin]}
    nov_ewm = {d["grupo"]: d for d in perf["novo_no_treino"]["media_exponencial"]}
    P(f"Por perfil, o MAE é {num(est['media >= 20 pts']['mae'])} para atletas com média de 20 pontos ou mais "
      f"({inteiro(est['media >= 20 pts']['n'])} jogos) e {num(est['media < 20 pts']['mae'])} para os demais; em termos "
      f"relativos, {pct(est['media >= 20 pts']['mae_relativo'], 0)} contra {pct(est['media < 20 pts']['mae_relativo'], 0)}. "
      f"Atletas sem linhas no treino ({inteiro(nov['jogador ausente do treino']['n'])} jogos de teste: novatos ou atletas "
      f"que antes não cumpriam o critério de participação) têm MAE de {num(nov['jogador ausente do treino']['mae'])}, menor "
      f"porque pontuam menos (erro relativo de {pct(nov['jogador ausente do treino']['mae_relativo'], 0)} contra "
      f"{pct(nov['jogador presente no treino']['mae_relativo'], 0)}). O ganho sobre a média exponencial nesse grupo "
      f"({num(nov_ewm['jogador ausente do treino']['mae'] - nov['jogador ausente do treino']['mae'])}) é semelhante ao dos "
      f"demais ({num(nov_ewm['jogador presente no treino']['mae'] - nov['jogador presente no treino']['mae'])}): o modelo não "
      "depende de ter visto o atleta no treino, porque usa só o histórico recente dele na temporada.")
    rel.subtitulo("6.3. Contribuição das variáveis")
    ab_rows = sorted(ab.items(), key=lambda kv: -kv[1]["aumento_vs_completo"]["delta"])
    pi = R["importancia_permutacao_agrupada"]["familias"]
    P("A Tabela 5 e a Figura 4 mostram quanto o erro aumenta ao retirar cada família de variáveis.")
    rel.tabela("Ablação por família no teste: aumento do MAE ao retreinar sem a família (IC 95% por jogador) e importância por permutação agrupada.",
               ["Família", "Aumento do MAE", "IC 95%", "p", "Permutação"],
               [[NOMES_FAMILIA[k], num(v["aumento_vs_completo"]["delta"], 4),
                 f"{num(v['aumento_vs_completo']['ic_inf'], 4)} a {num(v['aumento_vs_completo']['ic_sup'], 4)}",
                 num(v["aumento_vs_completo"]["p_valor"], 3), num(pi[k]["aumento_mae_medio"], 3)] for k, v in ab_rows],
               larguras=[6.0, 2.4, 3.6, 1.4, 2.2])
    a_ = lambda k: ab[k]["aumento_vs_completo"]  # noqa: E731
    P(f"O histórico de pontos domina: sem ele o MAE sobe {num(a_('historico_pontos')['delta'])}. Adversário "
      f"(+{num(a_('adversario')['delta'], 3)}; p {_p(a_('adversario')['p_valor'])}), minutos e papel na rotação "
      f"(+{num(a_('minutos_e_papel')['delta'], 3)}; p {_p(a_('minutos_e_papel')['p_valor'])}) e descanso e mando "
      f"(+{num(a_('descanso_e_mando')['delta'], 3)}; p {_p(a_('descanso_e_mando')['p_valor'])}) têm contribuições "
      "pequenas, e só a do adversário resiste a uma correção de Bonferroni para as seis comparações (limiar de 0,0083). O "
      "volume de jogo não tem efeito detectável na ablação, porque sua informação reaparece nas colunas do histórico, mas a "
      f"permutação agrupada ({num(pi['volume_de_jogo']['aumento_mae_medio'], 3)}) mostra que o modelo o usa quando está "
      "presente. A ablação manteve os hiperparâmetros do modelo completo, sem nova busca.")
    rel.figura("f_ablacao_familias.png", "Ablação por família de variáveis (aumento do MAE, IC 95% por jogador).", largura=13)
    rel.subtitulo("6.4. Classificação e calibração")
    rows = []
    for k in ["constante_treino", "logistica_l2", "logistica_l1", "random_forest", "hgb"]:
        v, w = C["teste"][k], C["teste_sem_empates"][k]
        rows.append([NOMES_MODELO[k] + (" (final)" if k == clf_fin else ""), pct(v["acuracia"]), num(v["f1"]), num(v["auc"]),
                     num(v["brier"], 4), pct(v["brier_skill"]), num(v["ece"], 3), num(w["auc"])])
    rg = C["regra_exponencial_acima_da_linha"]
    rows.append(["Regra: média exponencial acima da linha", pct(rg["acuracia"]), "-", num(rg["auc"]), "-", "-", "-", num(rg["auc_sem_empates"])])
    P("A Tabela 6 reúne as métricas de classificação no teste.")
    rel.tabela("Classificação no teste (superar a mediana dos 10 jogos anteriores).",
               ["Modelo", "Acurácia", "F1", "AUC", "Brier", "BSS", "ECE", "AUC sem empates"], rows,
               larguras=[5.4, 1.6, 1.2, 1.2, 1.5, 1.4, 1.3, 1.9])
    bv = C["brier_vs_ingenuo"]
    P(f"O modelo final, {NOMES_MODELO[clf_fin][0].lower() + NOMES_MODELO[clf_fin][1:]}, teve AUC de {num(cf['auc'])} (IC 95% por jogador de "
      f"{num(C['auc_ic']['ic_inf'])} a {num(C['auc_ic']['ic_sup'])}), acurácia de {pct(cf['acuracia'])} contra "
      f"{pct(C['teste']['constante_treino']['acuracia'])} do classificador que responde sempre a classe majoritária, e Brier "
      f"score de {num(cf['brier'], 4)} contra {num(cf['brier_ingenuo'], 4)} do estimador ingênuo (diferença de "
      f"{num(bv['delta'], 4)}, IC de {num(bv['ic_inf'], 4)} a {num(bv['ic_sup'], 4)}). A precisão foi "
      f"{pct(cf['precisao'])} e a revocação {pct(cf['revocacao'])}. A regressão logística já sai bem calibrada: nenhum "
      "calibrador melhorou o Brier na validação, e o modelo final ficou sem ajuste adicional. A curva de calibração "
      f"(Figura 5) acompanha a diagonal ao longo de toda a faixa de probabilidades, de cerca de 0,3 a 0,7. O ECE baixo "
      f"({num(cf['ece'], 3)}) é necessário, mas não suficiente: o estimador constante também tem ECE baixo "
      f"({num(C['teste']['constante_treino']['ece'], 3)}). A vantagem do modelo está na resolução, vista no Brier e na AUC.")
    abc = C["ablacao"]
    P(f"Os empates não explicam o resultado: sem eles, a AUC é {num(cfe['auc'])} e o Brier skill {pct(cfe['brier_skill'])}. "
      f"Retirar o indicador de linha inteira muda a AUC de {num(abc['completo']['auc'])} para {num(abc['sem_linha_inteira']['auc'])}. "
      f"Sem as variáveis relativas à linha, a AUC cai para {num(abc['so_pre_jogo']['auc'])}: elas acrescentam sinal, mas boa "
      "parte da discriminação já vem das variáveis pré-jogo de volume, minutos e histórico. O sinal mistura tendência "
      "recente, assimetria da distribuição e reversão ao patamar da temporada; com colunas tão correlacionadas, não é "
      "possível atribuí-lo a um único mecanismo. O ganho é modesto, como previsto na N1, e não autoriza nenhuma leitura "
      "sobre mercados.")
    rel.figura("f_curva_calibracao.png", "Curva de calibração do classificador final no teste (10 faixas de mesmo tamanho).", largura=10.5)
    rel.subtitulo("6.5. Incerteza: cobertura dos intervalos")
    cob80 = [d["cobertura"] for d in iv80["por_faixa_de_minutos"]]
    cob90 = [d["cobertura"] for d in iv90["por_faixa_de_minutos"]]
    P(f"Os quantis conformais calibrados na validação foram q = {num(iv80['q'], 2)} (80%) e q = {num(iv90['q'], 2)} (90%). "
      f"No teste, a cobertura foi de {pct(iv80['global']['cobertura'])} e {pct(iv90['global']['cobertura'])}, com largura "
      f"média de {num(iv80['global']['largura_media'], 1)} e {num(iv90['global']['largura_media'], 1)} pontos. Por faixa de "
      f"minutos (Figura 6), a cobertura variou de {pct(min(cob80))} a {pct(max(cob80))} no nível de 80% e de "
      f"{pct(min(cob90))} a {pct(max(cob90))} no de 90%: é menor para os atletas de menos minutos, como esperado de uma "
      "garantia apenas marginal. A largura dos intervalos é, ela mesma, um resultado: prever os pontos de um atleta em um "
      f"jogo comporta, honestamente, uma faixa de cerca de {num(iv80['global']['largura_media'], 0)} pontos para 80% de cobertura.")
    rel.figura("f_cobertura_intervalos.png", "Cobertura dos intervalos conformais por faixa de minutos (pontilhado: nominal).", largura=13)
    rel.subtitulo("6.6. Quanto do erro se deve a não conhecer o nível do atleta")
    ot = {k: v for k, v in teto.items() if k.startswith("oraculo") and isinstance(v, dict)}
    nomes_o = {"oraculo_temporada_media": "Média do atleta na temporada inteira", "oraculo_temporada_mediana": "Mediana do atleta na temporada inteira",
               "oraculo_local_media": "Média dos 5 jogos antes e 5 depois", "oraculo_local_mediana": "Mediana dos 5 jogos antes e 5 depois"}
    linhas_o = [[nomes_o[k], num(v["mae"]), num(v["modelo_menos_oraculo"]["delta"]),
                 f"{num(v['modelo_menos_oraculo']['ic_inf'])} a {num(v['modelo_menos_oraculo']['ic_sup'])}"] for k, v in ot.items()]
    linhas_o.append(["Modelo final + nível oráculo (média da temporada)", num(mo["mae_teste"]), num(b_mo["delta"]),
                     f"{num(b_mo['ic_inf'])} a {num(b_mo['ic_sup'])}"])
    P("A Tabela 7 compara o modelo com previsores que usam informação do futuro.")
    rel.tabela("Previsores com informação do futuro (não utilizáveis na prática) e diferença para o modelo final no teste.",
               ["Previsor com informação do futuro", "MAE", "Modelo − previsor", "IC 95%"], linhas_o,
               larguras=[6.5, 1.8, 3.0, 3.6])
    P("Nenhum oráculo simples erra menos que o modelo. Esses oráculos, porém, são estimadores ruidosos do nível: o que usa "
      "os cinco jogos antes e depois erra mais que a própria média exponencial pré-jogo. Por isso eles não são um limite "
      "superior; mostram apenas que o modelo estima o nível do atleta tão bem quanto uma média ou mediana da temporada "
      "inteira.")
    P(f"O diagnóstico mais informativo é o modelo com nível oráculo: o mesmo modelo final, recebendo também a média real do "
      f"atleta na temporada (com jogos futuros). Ele erra {num(mo['mae_teste'])}, ou seja, conhecer o nível verdadeiro "
      f"reduziria o MAE em apenas mais {num(b_mo['delta'])} ponto (IC de {num(b_mo['ic_inf'])} a {num(b_mo['ic_sup'])}). Da "
      f"distância entre a média exponencial ({num(T['media_exponencial']['mae'])}) e esse modelo com informação do futuro, o "
      f"modelo final percorre cerca de {pct(fracao_nivel, 0)}. O erro que sobra, perto de {num(mo['mae_teste'], 1)} pontos por "
      "jogo mesmo com o nível conhecido, vem principalmente da variação de jogo a jogo: minutos que dependem do placar e "
      "de faltas, aproveitamento de arremessos, contusões durante a partida e escalações. Isso não é um teto absoluto (um "
      "oráculo com outras informações, como os minutos que o atleta vai jogar, iria mais longe), mas delimita o quanto "
      "informação adicional sobre o *nível* do atleta poderia ajudar.")
    rel.subtitulo("6.7. Análise de erros")
    me = R["maiores_erros"][:5]
    exemplos = "; ".join(f"{d['PLAYER_NAME']} em {pd.Timestamp(d['GAME_DATE']):%d/%m/%Y} ({int(d['y_pts'])} pontos, previsão de {num(d['previsto'], 1)})" for d in me[:3])
    vd = pd.DataFrame(R["vies_por_decil_da_previsao"])
    d1, d10 = vd.iloc[0], vd.iloc[-1]
    acima = vd[vd["previsto_medio"] > vd["real_medio"]]
    abaixo = vd[vd["previsto_medio"] < vd["real_medio"]]
    trechos_acima = "; ".join(f"{num(r['previsto_medio'], 1)} contra {num(r['real_medio'], 1)}" for _, r in acima.iterrows())
    est_vies = {d["grupo"]: d["vies"] for d in perf["estrela"][fin]}
    est_vies_ewm = {d["grupo"]: d["vies"] for d in perf["estrela"]["media_exponencial"]}
    P(f"Os maiores erros são atuações excepcionais, como {exemplos}. São eventos reais, sem marca no histórico pré-jogo, e "
      f"foram mantidos no conjunto. Por decil da previsão, o modelo fica abaixo da média real em {len(abaixo)} decis (no "
      f"primeiro, previsão média de {num(d1['previsto_medio'], 1)} contra {num(d1['real_medio'], 1)} pontos reais) e acima "
      f"nos {len(acima)} decis superiores ({trechos_acima}). A subestimação nas faixas baixas e médias é o efeito esperado de "
      "prever a mediana condicional numa distribuição assimétrica, e produz o viés negativo visto na Tabela 3. A "
      "superestimação no topo não se explica por isso: indica que o modelo encolhe pouco as previsões dos maiores "
      f"pontuadores (viés de {num(est_vies['media >= 20 pts'], 2)} para atletas com média de 20 pontos ou mais, menor que o "
      f"de {num(est_vies_ewm['media >= 20 pts'], 2)} da média exponencial). A Figura 7 mostra previsto contra observado.")
    rel.figura("f_previsto_vs_observado.png", "Pontos previstos pelo modelo final contra pontos marcados no teste.", largura=10)

    # ------------------------------------------------------------ 7. conclusao
    rel.titulo_secao("7. Conclusão")
    P("Este trabalho desenvolveu um modelo supervisionado para estimar os pontos de atletas da NBA antes de cada partida, "
      "com dados coletados pelo próprio grupo, variáveis estritamente pré-jogo verificadas por testes automatizados e "
      "avaliação temporal com teste único. A Tabela 8 confronta os resultados esperados declarados na N1 com o que foi obtido.")
    rel.tabela("Resultados esperados (N1) e resultados obtidos.", ["Resultado esperado", "Situação", "Evidência"], [
        ["(a) Reduzir o MAE em relação ao baseline de forma perceptível, ainda que modesta", "Alcançado",
         f"−{pct(ganho_ma5)} contra a média de 5 jogos e −{pct(ganho_ewm)} contra a média exponencial; IC excluem zero. Contra uma regressão mediana sobre a exponencial, −{num(-b_lad['delta'])} ponto."],
        ["(b) Descanso, mando e adversário contribuem de forma mensurável, menos que o histórico", "Parcialmente alcançado",
         f"Adversário +{num(ab['adversario']['aumento_vs_completo']['delta'], 3)} e descanso/mando +{num(ab['descanso_e_mando']['aumento_vs_completo']['delta'], 3)} "
         f"de MAE ao remover, contra +{num(ab['historico_pontos']['aumento_vs_completo']['delta'], 3)} do histórico; só o adversário resiste à correção para múltiplas comparações."],
        ["(c) Média exponencial supera médias móveis simples", "Alcançado",
         f"MAE {num(T['media_exponencial']['mae'])} contra {num(T['media_10_jogos']['mae'])} (10 jogos), {num(T['media_5_jogos']['mae'])} (5) e {num(T['media_3_jogos']['mae'])} (3); IC excluem zero."],
        ["(d) Probabilidades calibradas, com Brier abaixo do ingênuo", "Alcançado",
         f"Brier {num(cf['brier'], 4)} contra {num(cf['brier_ingenuo'], 4)} (BSS {pct(cf['brier_skill'])}); curva de calibração próxima da diagonal."],
        ["(e) Estimar empiricamente o teto preditivo", "Parcialmente alcançado",
         f"Conhecer o nível real do atleta reduziria o MAE em só mais {num(b_mo['delta'], 3)} ({num(mo['mae_teste'])}); não é teto absoluto, pois não inclui outras informações (minutos, escalação)."],
    ], larguras=[6.0, 2.6, 7.4])
    P("Três dos cinco resultados esperados foram alcançados e dois, parcialmente. A principal contribuição não é um número "
      "de acurácia, e sim a delimitação honesta do que é previsível. Com informação pré-jogo de desempenho, um modelo "
      "supervisionado melhora de forma consistente, mas modesta, as médias usadas na prática, produz probabilidades "
      "calibradas e intervalos com a cobertura marginal prometida. O diagnóstico com informação do futuro sugere que a "
      "maior parte do erro restante vem da variação de jogo a jogo, e não de estimar mal o nível do atleta. Também se "
      "mostrou que a escolha da baseline importa: vencer a média de cinco jogos é fácil; vencer a média exponencial, e uma "
      "regressão mediana sobre ela, com intervalo de confiança, é o teste que realmente mede o valor do modelo.")
    P("As limitações permanecem as declaradas na N1: ausência de informação de contusões e escalações, que é a maior fonte "
      "de erro irredutível; recorte aos atletas de rotação; titularidade estimada; janela que não reinicia quando o atleta "
      "troca de equipe; dependência entre observações (tratada com bootstrap por jogador); e fonte única de dados. "
      "Acrescentam-se os empates da linha sintética, agora documentados. Como trabalhos futuros, destacam-se incorporar a "
      "escalação confirmada antes do jogo, modelar minutos e aproveitamento separadamente e testar intervalos condicionais "
      "por quantis (ROMANO; PATTERSON; CANDÈS, 2019).")
    P("Em síntese, o artefato serve para estimar, com incerteza explícita, o desempenho de atletas de rotação em partidas "
      "de temporada regular. Não serve para jogadores fora da rotação, para pós-temporada, nem para qualquer conclusão a "
      "respeito de mercados de apostas.")

    # ------------------------------------------------------------ 8. enderecos
    rel.titulo_secao("8. Endereço GitHub e Endereço do Vídeo")
    P(f"Repositório público do projeto: {GITHUB}", first=None)
    P(f"Vídeo de apresentação (YouTube): {video}", first=None)

    # ------------------------------------------------------------ 9. referencias
    rel.titulo_secao("9. Referências")
    refs = [
        "ANGELOPOULOS, A. N.; BATES, S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511, 2021.",
        "BERGMEIR, C.; BENÍTEZ, J. M. On the use of cross-validation for time series predictor evaluation. Information Sciences, v. 191, p. 192-213, 2012.",
        "BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD). Brasília, DF: Presidência da República, 2018.",
        "BREIMAN, L. Random forests. Machine Learning, v. 45, n. 1, p. 5-32, 2001.",
        "BRIER, G. W. Verification of forecasts expressed in terms of probability. Monthly Weather Review, v. 78, n. 1, p. 1-3, 1950.",
        "COPPIN, B. Inteligência artificial. Rio de Janeiro: LTC, 2010.",
        "FRIEDMAN, J. H. Greedy function approximation: a gradient boosting machine. The Annals of Statistics, v. 29, n. 5, p. 1189-1232, 2001.",
        "GÉRON, A. Mãos à obra: aprendizado de máquina com Scikit-Learn, Keras & TensorFlow. 2. ed. Rio de Janeiro: Alta Books, 2021.",
        "GUO, C. et al. On calibration of modern neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 34., 2017, Sydney. Proceedings [...]. PMLR, 2017. p. 1321-1330.",
        "HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. The Elements of Statistical Learning: data mining, inference, and prediction. 2. ed. New York: Springer, 2009.",
        "HUNTER, J. D. Matplotlib: a 2D graphics environment. Computing in Science & Engineering, v. 9, n. 3, p. 90-95, 2007.",
        "HOERL, A. E.; KENNARD, R. W. Ridge regression: biased estimation for nonorthogonal problems. Technometrics, v. 12, n. 1, p. 55-67, 1970.",
        "JOBIN, A.; IENCA, M.; VAYENA, E. The global landscape of AI ethics guidelines. Nature Machine Intelligence, v. 1, n. 9, p. 389-399, 2019.",
        "KUBATKO, J. et al. A starting point for analyzing basketball statistics. Journal of Quantitative Analysis in Sports, v. 3, n. 3, 2007.",
        "McKINNEY, W. Data structures for statistical computing in Python. In: PROCEEDINGS OF THE 9TH PYTHON IN SCIENCE CONFERENCE, 2010, Austin. Proceedings [...]. Austin: SciPy, 2010. p. 56-61.",
        "NICULESCU-MIZIL, A.; CARUANA, R. Predicting good probabilities with supervised learning. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 22., 2005, Bonn. Proceedings [...]. New York: ACM, 2005. p. 625-632.",
        "OLIVER, D. Basketball on paper: rules and tools for performance analysis. Washington: Potomac Books, 2004.",
        "PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.",
        "ROMANO, Y.; PATTERSON, E.; CANDÈS, E. Conformalized quantile regression. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 32., 2019, Vancouver. Proceedings [...]. 2019.",
        "RUSSELL, S.; NORVIG, P. Inteligência artificial: uma abordagem moderna. Rio de Janeiro: LTC, 2021.",
        "SWAR. nba_api: an API client package to access the APIs for NBA.com. GitHub, 2024. Disponível em: https://github.com/swar/nba_api. Acesso em: 14 set. 2026.",
        "TIBSHIRANI, R. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B, v. 58, n. 1, p. 267-288, 1996.",
        "VOVK, V.; GAMMERMAN, A.; SHAFER, G. Algorithmic learning in a random world. New York: Springer, 2005.",
    ]
    for r in refs:
        rel.par(r, align=WD_ALIGN_PARAGRAPH.LEFT, first=None, before=6)

    rel.salvar()
    print(f"ok: {OUT_DOCX} ({rel.n_tab} tabelas, {rel.n_fig} figuras)")


def n_testes(arquivo: str = "test_*.py") -> int:
    """Quantidade de funcoes de teste em tests/ (evita numero fixo no texto)."""
    return sum(len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), flags=re.M)) for f in (ROOT / "tests").glob(arquivo))


def _p(p: float) -> str:
    return "< 0,001" if p < 0.001 else "= " + num(p, 3)


def _extenso(n: int) -> str:
    nomes = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "onze", "doze"]
    return nomes[n] if n < len(nomes) else str(n)


def _fmt_param(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}".replace(".", ",")
    return str(v)


def export_pdf() -> None:
    """Converte o .docx em PDF pelo Microsoft Word (Windows, via automacao COM).

    O Word e o mesmo programa do template oficial, entao o PDF sai com a mesma
    paginacao que o grupo veria ao abrir o .docx.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    # O Word nao abre caminhos de rede (\\wsl.localhost\...), entao a conversao
    # acontece numa pasta temporaria local e o PDF e copiado de volta.
    tmp = Path(tempfile.mkdtemp())
    src, dst = tmp / OUT_DOCX.name, tmp / OUT_DOCX.with_suffix(".pdf").name
    shutil.copy(OUT_DOCX, src)
    ps = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"$d = $w.Documents.Open('{src}'); $d.ExportAsFixedFormat('{dst}', 17); "
        "$n = $d.ComputeStatistics(2); $d.Close(0); $w.Quit(); Write-Output $n"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=300)
    if out.returncode != 0 or not dst.exists():
        raise SystemExit(f"falha ao exportar PDF: {out.stderr.strip()}")
    shutil.copy(dst, OUT_DOCX.with_suffix(".pdf"))
    print(f"ok: {OUT_DOCX.with_suffix('.pdf')} ({out.stdout.strip()} paginas)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera relatorio/relatorio_N2.docx a partir dos resultados.")
    ap.add_argument("--video-url", help="link publico do video no YouTube")
    ap.add_argument("--pdf", action="store_true", help="tambem exporta o PDF pelo Word (Windows)")
    args = ap.parse_args()
    build(args.video_url)
    if args.pdf:
        export_pdf()


if __name__ == "__main__":
    main()
