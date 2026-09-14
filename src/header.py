# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/header.py
# SINTESE: Metadados do grupo para o cabeçalho obrigatório.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Metadados do grupo usados pelo cabecalho obrigatorio.

Preencha INTEGRANTES e rode `python scripts/apply_header.py`.
"""

from __future__ import annotations

INSTITUICAO = "Universidade Presbiteriana Mackenzie"
DISCIPLINA = "Inteligência Artificial"
PROJETO = "Previsão de desempenho de jogadores da NBA (pontos por jogo)"

INTEGRANTES: list[dict[str, str]] = [
    {"nome": "Rodrigo Lucas Mascarenhas Leite Oliveira", "tia": "10427925"},
    {"nome": "André Ihsan Ward", "tia": "10425684"},
]

SINTESES: dict[str, str] = {
    "src/header.py": "Metadados do grupo para o cabeçalho obrigatório.",
    "src/collect.py": "Coleta de game logs da NBA com cache, retry e manifest.",
    "src/features.py": "Features pré-jogo (médias móveis shiftadas, calendário, adversário).",
    "src/target.py": "Alvos de regressão e classificação com linha sintética.",
    "src/build_dataset.py": "Pipeline de coleta bruta até dataset analítico com split temporal.",
    "src/io_utils.py": "Leitura e escrita de tabelas em parquet com fallback para CSV.",
    "scripts/apply_header.py": "Aplica o cabeçalho obrigatório nos arquivos .py e .ipynb.",
    "tests/test_no_leakage.py": "Testes de vazamento temporal nas features.",
    "notebooks/01_eda.ipynb": "Análise exploratória do dataset.",
}

HISTORICO: list[tuple[str, str, str]] = [
    ("2026-09-14", "grupo", "Versão inicial: coleta, features, alvos, testes e EDA."),
]
