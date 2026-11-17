# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/conftest.py
# SINTESE: Configuração compartilhada dos testes (marcador slow).
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: registra o marcador slow.
# === FIM DO CABECALHO ===

"""Configuracao compartilhada da suite de testes."""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: ponta a ponta com modelos reduzidos (dezenas de segundos); pule com -m 'not slow'",
    )
