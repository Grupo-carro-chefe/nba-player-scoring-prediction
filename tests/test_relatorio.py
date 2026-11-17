# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_relatorio.py
# SINTESE: Relatório: formatação pt-BR, fatos verificados, geração do .docx e coerência com o versionado.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: geração e coerência do relatório.
# === FIM DO CABECALHO ===

"""Geracao do relatorio (src/report.py): formatacao, fatos verificados e o .docx.

O relatorio e gerado por codigo a partir de results/. Estes testes provam que a
geracao funciona do zero, que o texto so afirma o que os resultados sustentam e
que o .docx versionado e exatamente o que o codigo gera hoje.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import sys
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

from src import report  # noqa: E402
from src.features import PRE_GAME_FEATURES  # noqa: E402
from src.modeling import load_dataset, regression_zoo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YOUTUBE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]{6,}")


def _textos(path: Path) -> list[str]:
    doc = docx.Document(str(path))
    partes = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        partes += [c.text for row in t.rows for c in row.cells]
    return partes


@pytest.fixture(scope="module")
def resultados() -> tuple[dict, dict]:
    tv = json.loads((ROOT / "results" / "metrics" / "treino_validacao.json").read_text(encoding="utf-8"))
    tf = json.loads((ROOT / "results" / "metrics" / "test_final.json").read_text(encoding="utf-8"))
    return tv, tf


# ----------------------------------------------------------------------------
# formatacao pt-BR
# ----------------------------------------------------------------------------

def test_numeros_em_formato_brasileiro():
    assert report.num(4.9236) == "4,924"
    assert report.num(0.0) == "0,000" and report.num(-1.5, 1) == "-1,5"
    assert report.num(83799, 0) == "83.799"
    assert report.num(1234567.891, 2) == "1.234.567,89"
    assert report.pct(0.8022) == "80,2%" and report.pct(0.5, 0) == "50%"
    assert report.inteiro(131292) == "131.292" and report.inteiro(7) == "7" and report.inteiro(12.0) == "12"


def test_p_valor_e_numeros_por_extenso():
    assert report._p(0.0004) == "< 0,001"
    assert report._p(0.0123) == "= 0,012"
    assert [report._extenso(n) for n in (0, 1, 2, 7, 12)] == ["zero", "um", "dois", "sete", "doze"]
    assert report._extenso(13) == "13"
    assert report._fmt_param(0.0123456) == "0,0123" and report._fmt_param(10) == "10" and report._fmt_param("x") == "x"


# ----------------------------------------------------------------------------
# contagem de testes citada no texto
# ----------------------------------------------------------------------------

def test_contagem_de_testes_conta_funcoes_de_nivel_de_modulo(tmp_path, monkeypatch):
    pasta = tmp_path / "tests"
    pasta.mkdir()
    (pasta / "test_a.py").write_text(
        "def test_um():\n    pass\n\ndef test_dois():\n    pass\n\ndef auxiliar():\n    pass\n\n"
        "class Grupo:\n    def test_metodo(self):\n        pass\n",
        encoding="utf-8",
    )
    (pasta / "test_b.py").write_text("def test_tres():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    assert report.n_testes() == 3
    assert report.n_testes("test_a.py") == 2


def test_contagem_do_relatorio_bate_com_a_analise_da_suite_real():
    por_ast = 0
    for arq in (ROOT / "tests").glob("test_*.py"):
        arvore = ast.parse(arq.read_text(encoding="utf-8"))
        por_ast += sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in arvore.body)
    assert report.n_testes() == por_ast


# ----------------------------------------------------------------------------
# fatos afirmados no texto
# ----------------------------------------------------------------------------

def test_todos_os_fatos_do_texto_valem_com_os_resultados_atuais(resultados):
    tv, tf = resultados
    fatos = report.fatos_do_texto(tv, tf, regression_zoo())
    assert len(fatos) >= 14 and all(isinstance(v, bool) for v in fatos.values())
    assert all(fatos.values()), [k for k, v in fatos.items() if not v]
    report.exigir_fatos(fatos)  # nao levanta


def test_fato_deixa_de_valer_quando_o_resultado_muda(resultados):
    tv, tf = resultados
    alterado = copy.deepcopy(tf)
    modelo = alterado["regressao"]["modelo_final"]
    alterado["regressao"]["comparacoes_bootstrap"][f"{modelo}_vs_media_5_jogos"]["ic_sup"] = 0.1
    fatos = report.fatos_do_texto(tv, alterado, regression_zoo())
    assert fatos["IC do modelo contra media de 5, exponencial e linear excluem zero"] is False
    outro = copy.deepcopy(tf)
    outro["classificacao"]["calibracao"]["metodo"] = "isotonic"
    assert report.fatos_do_texto(tv, outro, regression_zoo())["classificador final e logistico sem calibracao adicional"] is False


def test_geracao_para_se_algum_fato_for_falso():
    with pytest.raises(SystemExit, match="fatos que os resultados atuais nao sustentam") as erro:
        report.exigir_fatos({"vale": True, "nao vale mais": False})
    assert "nao vale mais" in str(erro.value) and "vale\n" not in str(erro.value).replace("nao vale mais", "")


# ----------------------------------------------------------------------------
# numeros da analise exploratoria
# ----------------------------------------------------------------------------

def test_numeros_da_eda_batem_com_o_dataset():
    df = load_dataset()
    e = report.eda_numbers(df)
    assert e["n"] == len(df) and e["split"].sum() == len(df)
    assert e["jogadores"] == df["PLAYER_ID"].nunique() and e["jogos"] == df["GAME_ID"].nunique()
    assert e["inicio"] < e["fim"] and e["max"] == df["y_pts"].max()
    assert e["empates"] == int((df["y_pts"] == df["line_synthetic"]).sum())
    assert 0 < e["pos"] < 1 and 0 <= e["zeros"] < 0.05
    assert set(e["corr"].index) == set(PRE_GAME_FEATURES) and e["corr"].is_monotonic_decreasing
    assert e["gp_min"] >= 10 and e["q1"] < e["mediana"] < e["q3"]


# ----------------------------------------------------------------------------
# geracao do .docx
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def docx_gerado(tmp_path_factory):
    destino = tmp_path_factory.mktemp("relatorio") / "relatorio_teste.docx"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(report, "OUT_DOCX", destino)
        report.build(None)
    return destino


def test_relatorio_e_gerado_com_tabelas_figuras_e_marcador_do_video(docx_gerado):
    assert docx_gerado.exists() and docx_gerado.stat().st_size > 100_000
    doc = docx.Document(str(docx_gerado))
    assert len(doc.tables) >= 8
    citadas = re.findall(r'rel\.figura\("([^"]+)"', (ROOT / "src" / "report.py").read_text(encoding="utf-8"))
    assert len(citadas) >= 5 and len(doc.inline_shapes) == len(citadas)  # cada figura chamada entrou no documento
    assert all((ROOT / "results" / "figures" / f).exists() for f in citadas)
    texto = "\n".join(_textos(docx_gerado))
    assert report.VIDEO_PLACEHOLDER in texto
    assert report.GITHUB in texto.replace(" ", "")
    assert doc.core_properties.title and "NBA" in doc.core_properties.title


def test_link_do_video_substitui_o_marcador(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT_DOCX", tmp_path / "r.docx")
    report.build("https://youtu.be/abc123XYZ")
    texto = "\n".join(_textos(tmp_path / "r.docx"))
    assert "https://youtu.be/abc123XYZ" in texto and report.VIDEO_PLACEHOLDER not in texto


def test_relatorio_versionado_e_exatamente_o_que_o_codigo_gera_hoje(docx_gerado, tmp_path, monkeypatch):
    versionado = ROOT / "relatorio" / "relatorio_N2.docx"
    link = YOUTUBE.search(" ".join(_textos(versionado)))
    if link:  # com o video ja publicado, o texto esperado leva o link
        monkeypatch.setattr(report, "OUT_DOCX", tmp_path / "r.docx")
        report.build(link.group(0))
        esperado = _textos(tmp_path / "r.docx")
    else:
        esperado = _textos(docx_gerado)
    atual = _textos(versionado)
    diferentes = [(a[:70], b[:70]) for a, b in zip(atual, esperado) if a != b]
    assert len(atual) == len(esperado) and not diferentes, (
        f"relatorio_N2.docx desatualizado em relacao ao codigo e aos resultados; rode python -m src.report --pdf. "
        f"Primeira diferenca: {diferentes[:1]}"
    )


def test_linha_de_comando_repassa_link_e_pdf(monkeypatch):
    chamadas = []
    monkeypatch.setattr(report, "build", lambda url: chamadas.append(("build", url)))
    monkeypatch.setattr(report, "export_pdf", lambda: chamadas.append(("pdf",)))
    monkeypatch.setattr(sys, "argv", ["report", "--video-url", "https://youtu.be/abc123XYZ", "--pdf"])
    report.main()
    assert chamadas == [("build", "https://youtu.be/abc123XYZ"), ("pdf",)]
    chamadas.clear()
    monkeypatch.setattr(sys, "argv", ["report"])
    report.main()
    assert chamadas == [("build", None)]
