# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_resultados.py
# SINTESE: Testes que prendem README e relatório aos números de test_final.json.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: números citados batem com test_final.json.
#   2026-11-15 | Rodrigo    | Testes de proveniência (hash do código) e dos fatos do relatório.
# === FIM DO CABECALHO ===

"""Prende README e relatorio aos numeros de results/metrics/test_final.json.

Mesmo padrao de tests/test_documentacao.py: um numero citado na documentacao
que nao bate com o arquivo de resultados e um erro, nao um detalhe de redacao.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "metrics" / "test_final.json"
README = ROOT / "README.md"
REPORT = ROOT / "relatorio" / "relatorio_N2.docx"


def _br(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


@pytest.fixture(scope="module")
def final() -> dict:
    if not FINAL.exists():
        pytest.skip("teste ainda nao avaliado (python -m src.evaluate --final)")
    return json.loads(FINAL.read_text(encoding="utf-8"))


def _citados(f: dict) -> dict[str, str]:
    r, c = f["regressao"], f["classificacao"]
    fin = r["modelo_final"]
    return {
        "MAE do modelo final": _br(r["teste"][fin]["mae"]),
        "MAE da media de 5 jogos": _br(r["teste"]["media_5_jogos"]["mae"]),
        "MAE da media exponencial": _br(r["teste"]["media_exponencial"]["mae"]),
        "reducao vs media de 5": _br(-r["comparacoes_bootstrap"][f"{fin}_vs_media_5_jogos"]["delta"]),
        "reducao vs exponencial": _br(-r["comparacoes_bootstrap"][f"{fin}_vs_media_exponencial"]["delta"]),
        "cobertura 80%": _br(100 * r["intervalos"]["80"]["global"]["cobertura"], 1) + "%",
        "cobertura 90%": _br(100 * r["intervalos"]["90"]["global"]["cobertura"], 1) + "%",
        "AUC": _br(c["teste"]["final_calibrado"]["auc"]),
        "Brier": _br(c["teste"]["final_calibrado"]["brier"], 4),
        "Brier ingenuo": _br(c["teste"]["final_calibrado"]["brier_ingenuo"], 4),
        "ECE": _br(c["teste"]["final_calibrado"]["ece"]),
    }


def test_readme_cita_os_resultados_do_teste(final):
    texto = README.read_text(encoding="utf-8")
    faltando = {k: v for k, v in _citados(final).items() if v not in texto}
    assert not faltando, f"README desatualizado em relacao a test_final.json: {faltando}"


def test_relatorio_cita_os_resultados_do_teste(final):
    docx = pytest.importorskip("docx")
    if not REPORT.exists():
        pytest.skip("relatorio ainda nao gerado (python -m src.report)")
    doc = docx.Document(str(REPORT))
    partes = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        partes += [cell.text for row in t.rows for cell in row.cells]
    texto = "\n".join(partes)
    citados = _citados(final)
    citados.pop("cobertura 80%"), citados.pop("cobertura 90%")  # no relatorio com 1 casa, sem %
    faltando = {k: v for k, v in citados.items() if v not in texto}
    assert not faltando, f"relatorio desatualizado em relacao a test_final.json: {faltando}"


def test_modelo_final_foi_escolhido_sem_o_teste(final):
    """O modelo final do teste e o de menor erro na VALIDACAO, nao no teste."""
    tv = json.loads((ROOT / "results" / "metrics" / "treino_validacao.json").read_text(encoding="utf-8"))
    val = {k: v["mae"] for k, v in tv["regressao"]["validacao"].items() if v["tipo"] == "modelo"}
    assert final["regressao"]["modelo_final"] == min(val, key=val.get)
    assert tv["ambiente"]["gerado_em_utc"] < final["ambiente"]["gerado_em_utc"]


def test_resultados_foram_gerados_pelo_codigo_atual(final):
    """Se o codigo de modelagem mudou, treino e avaliacao precisam ser refeitos."""
    pytest.importorskip("sklearn")
    from src.train import hash_codigo

    tv = json.loads((ROOT / "results" / "metrics" / "treino_validacao.json").read_text(encoding="utf-8"))
    atual = hash_codigo()
    assert tv["ambiente"].get("hash_codigo") == atual, "treino_validacao.json desatualizado: rode python -m src.train"
    assert final["ambiente"].get("hash_codigo") == atual, "test_final.json desatualizado: rode python -m src.evaluate --final --force"


def test_fatos_afirmados_no_relatorio_continuam_valendo(final):
    """As frases qualitativas do relatorio ('todos os IC excluem zero' etc.)."""
    pytest.importorskip("docx")
    from src.modeling import regression_zoo
    from src.report import fatos_do_texto

    tv = json.loads((ROOT / "results" / "metrics" / "treino_validacao.json").read_text(encoding="utf-8"))
    falsos = [k for k, v in fatos_do_texto(tv, final, regression_zoo()).items() if not v]
    assert not falsos, f"o relatorio afirma fatos que os resultados nao sustentam: {falsos}"
