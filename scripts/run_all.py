# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: scripts/run_all.py
# SINTESE: Executa o pipeline da N2 de ponta a ponta (treino, teste, figuras, notebooks).
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: pipeline da N2 de ponta a ponta.
# === FIM DO CABECALHO ===

"""Executa o pipeline da N2 de ponta a ponta.

    python scripts/run_all.py            # treino, teste (se ainda nao avaliado), figuras e notebooks
    python scripts/run_all.py --no-train # so figuras e notebooks a partir dos resultados salvos

Nao recoleta dados: parte do dataset versionado em data/processed/. A avaliacao
no teste so roda se results/metrics/test_final.json ainda nao existir (ver
src/evaluate.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "02_baselines_e_regressao",
    "03_classificacao_e_calibracao",
    "04_incerteza_teto_e_erros",
    "05_demo_artefato",
]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true", help="nao retreina; usa results/ existentes")
    args = ap.parse_args()

    if not args.no_train:
        run("-m", "src.train")
    if not (ROOT / "results" / "metrics" / "test_final.json").exists():
        run("-m", "src.evaluate", "--final")
    run("-m", "src.figures")
    for nb in NOTEBOOKS:
        run("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=600", f"notebooks/{nb}.ipynb")
    run("scripts/apply_header.py")
    run("-m", "pytest", "-q")


if __name__ == "__main__":
    main()
