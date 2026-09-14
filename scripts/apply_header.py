# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: scripts/apply_header.py
# SINTESE: Aplica o cabeçalho obrigatório nos arquivos .py e .ipynb.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-09-14 | grupo      | Versão inicial: coleta, features, alvos, testes e EDA.
# === FIM DO CABECALHO ===

"""Aplica o cabecalho obrigatorio em todos os .py e .ipynb.

    python scripts/apply_header.py
    python scripts/apply_header.py --check

Os dados vem de src/header.py. O bloco fica entre marcadores e e substituido a
cada execucao, entao rodar duas vezes nao duplica nada.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.header import DISCIPLINA, HISTORICO, INSTITUICAO, INTEGRANTES, PROJETO, SINTESES  # noqa: E402

# Montados por concatenacao: se o marcador aparecesse literal aqui, o script
# apagaria as proprias definicoes ao se processar.
BEGIN = "# === " + "CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ==="
END = "# === FIM DO " + "CABECALHO ==="

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".cache_nba", "data"}


def build_header(rel_path: str) -> str:
    linhas = [
        BEGIN,
        f"# {INSTITUICAO}",
        f"# Disciplina: {DISCIPLINA}",
        f"# Projeto: {PROJETO}",
        "#",
        "# INTEGRANTES DO GRUPO:",
    ]
    linhas += [f"#   - {m['nome']} - TIA {m['tia']}" for m in INTEGRANTES]
    linhas += [
        "#",
        f"# ARQUIVO: {rel_path}",
        f"# SINTESE: {SINTESES.get(rel_path, '(sem sintese cadastrada em src/header.py)')}",
        "#",
        "# HISTORICO DE ALTERACOES:",
        "#   data       | autor      | descricao",
        "#   -----------|------------|-------------------------------------------",
    ]
    linhas += [f"#   {data} | {autor:<10} | {desc}" for data, autor, desc in HISTORICO]
    linhas.append(END)
    return "\n".join(linhas)


def _strip_existing(text: str) -> str:
    # so remove se o bloco estiver no topo, para nao confundir com uma mencao
    # ao marcador no meio do codigo
    idx = text.find(BEGIN)
    if idx == -1 or text[:idx].count("\n") > 3:
        return text
    fim = text.find(END, idx)
    if fim == -1:
        return text
    return (text[:idx] + text[fim + len(END) :]).lstrip("\n")


def apply_py(path: Path, header: str, check: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = _strip_existing(original).split("\n")

    prefix: list[str] = []
    while lines and (lines[0].startswith("#!") or "coding:" in lines[0][:40]):
        prefix.append(lines.pop(0))

    new = "\n".join(prefix + [header, ""] + lines) if prefix else header + "\n\n" + "\n".join(lines)
    new = new.rstrip("\n") + "\n"

    if new == original:
        return False
    if not check:
        path.write_text(new, encoding="utf-8")
    return True


def apply_ipynb(path: Path, header: str, check: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    cells = [
        c for c in nb.get("cells", [])
        if not (c.get("cell_type") == "code" and "".join(c.get("source", [])).startswith(BEGIN))
    ]
    nb["cells"] = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [ln + "\n" for ln in header.split("\n")],
        },
        *cells,
    ]

    new = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if new == raw:
        return False
    if not check:
        path.write_text(new, encoding="utf-8")
    return True


def iter_targets() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*")
        if p.suffix in {".py", ".ipynb"}
        and not any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="nao escreve; sai com 1 se algo estiver desatualizado")
    args = ap.parse_args()

    if any("PREENCHER" in m["nome"] or "PREENCHER" in m["tia"] for m in INTEGRANTES):
        print("AVISO: ha integrantes nao preenchidos em src/header.py\n")

    changed = 0
    for path in iter_targets():
        rel = path.relative_to(ROOT).as_posix()
        fn = apply_ipynb if path.suffix == ".ipynb" else apply_py
        if fn(path, build_header(rel), args.check):
            changed += 1
            print(f"  {'desatualizado' if args.check else 'atualizado'}: {rel}")
        else:
            print(f"  ok: {rel}")

    if args.check and changed:
        print(f"\n{changed} arquivo(s) sem o cabecalho atualizado. Rode: python scripts/apply_header.py")
        return 1
    print(f"\n{changed} arquivo(s) alterado(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
