# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_documentacao.py
# SINTESE: Testes que mantêm a documentação coerente com o dataset gerado.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Testes que mantem a documentacao coerente com o dataset gerado.

Existem porque os numeros do data/README.md ja sairam uma vez de uma execucao de
teste com uma temporada, enquanto o dataset publicado tinha cinco. A conta
131.292 - 33.504 - 13.989 = 83.799 e a mais facil de conferir de fora, entao
vale prende-la a um teste.

Os testes sao pulados quando o dataset ainda nao foi gerado, para nao quebrar um
clone limpo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "dataset_manifest.json"
DATA_README = ROOT / "data" / "README.md"


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("dataset ainda nao gerado (rode python -m src.build_dataset)")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _numeros(texto: str) -> set[int]:
    """Numeros do texto, aceitando ponto como separador de milhar."""
    return {int(m.replace(".", "")) for m in re.findall(r"\b\d{1,3}(?:\.\d{3})+\b|\b\d{4,}\b", texto)}


def test_aritmetica_dos_filtros_fecha():
    m = _manifest()
    f = m["filtros"]
    esperado = f["linhas_brutas"] - f["descartadas_por_minutos"] - f["descartadas_sem_alvo"]
    assert esperado == m["linhas"], (
        f"{f['linhas_brutas']} - {f['descartadas_por_minutos']} - "
        f"{f['descartadas_sem_alvo']} = {esperado}, mas o dataset tem {m['linhas']} linhas"
    )


def test_soma_dos_splits_bate_com_o_total():
    m = _manifest()
    assert sum(m["distribuicao_split"].values()) == m["linhas"]


def test_readme_do_dataset_cita_os_numeros_corretos():
    """Os quatro numeros da conta precisam aparecer no data/README.md."""
    if not DATA_README.exists():
        pytest.skip("data/README.md ausente")
    m = _manifest()
    f = m["filtros"]
    citados = _numeros(DATA_README.read_text(encoding="utf-8"))

    for rotulo, valor in (
        ("linhas brutas", f["linhas_brutas"]),
        ("descartadas por minutos", f["descartadas_por_minutos"]),
        ("descartadas sem alvo", f["descartadas_sem_alvo"]),
        ("linhas finais", m["linhas"]),
    ):
        assert valor in citados, f"data/README.md nao cita {rotulo} = {valor:,}".replace(",", ".")


def test_readme_do_dataset_cita_os_splits():
    if not DATA_README.exists():
        pytest.skip("data/README.md ausente")
    m = _manifest()
    citados = _numeros(DATA_README.read_text(encoding="utf-8"))
    for nome, valor in m["distribuicao_split"].items():
        assert valor in citados, f"data/README.md nao cita o split {nome} = {valor:,}".replace(",", ".")
