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
#   2026-11-15 | Rodrigo    | Não retreina sem --retrain quando o teste já foi avaliado.
#   2026-11-15 | Rodrigo    | Passa --force ao treino e --pdf no Windows.
# === FIM DO CABECALHO ===

"""Executa o pipeline da N2 de ponta a ponta.

    python scripts/run_all.py            # figuras, relatorio e notebooks a partir dos resultados salvos
    python scripts/run_all.py --retrain  # retreina E reavalia no teste (artefatos sempre consistentes)

Nao recoleta dados: parte do dataset versionado em data/processed/.

Se o teste ainda nao foi avaliado, o script treina e avalia. Se ja foi, por
padrao ele NAO retreina: retreinar sobrescreveria modelos, quantis conformais e
treino_validacao.json, deixando test_final.json descrevendo artefatos que nao
existem mais. Com --retrain, treino e avaliacao sao refeitos juntos (a avaliacao
com --force), o que e uma reexecucao da mesma avaliacao, nao uma nova escolha.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "metrics" / "test_final.json"
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
    ap.add_argument("--retrain", action="store_true", help="retreina e reavalia no teste")
    ap.add_argument("--video-url", help="link do video, repassado ao relatorio")
    args = ap.parse_args()

    if not FINAL.exists():
        run("-m", "src.train")
        run("-m", "src.evaluate", "--final")
    elif args.retrain:
        run("-m", "src.train", "--force")
        run("-m", "src.evaluate", "--final", "--force")
    else:
        print("teste ja avaliado: usando os resultados salvos (use --retrain para refazer tudo)")
    run("-m", "src.figures")
    for nb in NOTEBOOKS:
        run("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=600", f"notebooks/{nb}.ipynb")
    run("scripts/apply_header.py")
    report = ["-m", "src.report"] + (["--video-url", args.video_url] if args.video_url else [])
    if sys.platform == "win32":
        report.append("--pdf")  # o PDF e o arquivo entregue; a exportacao usa o Word
    run(*report)
    if sys.platform != "win32":
        print("ATENCAO: relatorio_N2.pdf NAO foi regenerado (exige Word no Windows). Exporte o .docx manualmente.")
    run("-m", "pytest", "-q")


if __name__ == "__main__":
    main()
