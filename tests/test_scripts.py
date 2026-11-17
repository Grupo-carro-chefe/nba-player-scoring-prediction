# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_scripts.py
# SINTESE: Cabeçalho obrigatório (apply_header) e ordem do pipeline em run_all.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: apply_header e run_all.
# === FIM DO CABECALHO ===

"""Scripts de apoio: cabecalho obrigatorio e pipeline run_all.

O cabecalho e exigencia do enunciado (identificacao dos integrantes, sintese e
historico em todo .py e .ipynb); o script que o aplica precisa ser idempotente e
nunca apagar o codigo do arquivo.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_header  # noqa: E402

from src import header  # noqa: E402


# ----------------------------------------------------------------------------
# apply_header
# ----------------------------------------------------------------------------

def test_cabecalho_tem_todos_os_campos_exigidos():
    texto = apply_header.build_header("src/predict.py")
    linhas = texto.splitlines()
    assert linhas[0] == apply_header.BEGIN and linhas[-1] == apply_header.END
    for membro in header.INTEGRANTES:
        assert f"{membro['nome']} - TIA {membro['tia']}" in texto
    assert "# ARQUIVO: src/predict.py" in texto
    assert f"# SINTESE: {header.SINTESES['src/predict.py']}" in texto
    assert "#   data       | autor      | descricao" in texto
    assert all(l.startswith("#") for l in linhas)
    for data, autor, desc in header.historico("src/predict.py"):
        assert f"{data} | {autor:<10} | {desc}" in texto


def test_arquivo_sem_sintese_cadastrada_e_sinalizado():
    assert "(sem sintese cadastrada" in apply_header.build_header("src/nao_existe.py")


def test_aplicar_duas_vezes_nao_muda_nada_e_preserva_o_codigo(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('"""Doc."""\nx = 1\n', encoding="utf-8")
    h = apply_header.build_header("src/exemplo.py")
    assert apply_header.apply_py(p, h, check=False) is True
    texto = p.read_text(encoding="utf-8")
    assert texto.startswith(apply_header.BEGIN) and texto.count(apply_header.BEGIN) == 1
    assert texto.endswith('"""Doc."""\nx = 1\n')
    assert apply_header.apply_py(p, h, check=False) is False
    assert p.read_text(encoding="utf-8") == texto


def test_modo_de_verificacao_nao_escreve(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n", encoding="utf-8")
    h = apply_header.build_header("src/exemplo.py")
    assert apply_header.apply_py(p, h, check=True) is True  # esta desatualizado...
    assert p.read_text(encoding="utf-8") == "x = 1\n"  # ...mas nada foi gravado


def test_cabecalho_antigo_e_substituido_sem_duplicar(tmp_path, monkeypatch):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n", encoding="utf-8")
    apply_header.apply_py(p, apply_header.build_header("src/exemplo.py"), check=False)
    monkeypatch.setattr(apply_header, "historico", lambda rel: [("2030-01-01", "Fulano", "alteracao nova")])
    assert apply_header.apply_py(p, apply_header.build_header("src/exemplo.py"), check=False) is True
    texto = p.read_text(encoding="utf-8")
    assert texto.count(apply_header.BEGIN) == 1 and "alteracao nova" in texto
    assert texto.endswith("x = 1\n")


def test_shebang_e_declaracao_de_codificacao_ficam_no_topo(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\nx = 1\n", encoding="utf-8")
    apply_header.apply_py(p, apply_header.build_header("src/exemplo.py"), check=False)
    linhas = p.read_text(encoding="utf-8").splitlines()
    assert linhas[0] == "#!/usr/bin/env python" and linhas[1] == "# -*- coding: utf-8 -*-"
    assert linhas[2] == apply_header.BEGIN


def test_marcador_citado_no_meio_do_codigo_nao_e_confundido_com_o_cabecalho(tmp_path):
    p = tmp_path / "m.py"
    corpo = "\n".join(["a = 1", "b = 2", "c = 3", "d = 4", f'MARCA = "{apply_header.BEGIN}"', "e = 5"]) + "\n"
    p.write_text(corpo, encoding="utf-8")
    apply_header.apply_py(p, apply_header.build_header("src/exemplo.py"), check=False)
    texto = p.read_text(encoding="utf-8")
    assert texto.count(apply_header.BEGIN) == 2 and texto.endswith(corpo)


def _caderno(celulas: list[dict]) -> dict:
    return {"cells": celulas, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_cabecalho_em_notebook_e_a_primeira_celula_e_idempotente(tmp_path):
    p = tmp_path / "n.ipynb"
    md = {"cell_type": "markdown", "metadata": {}, "source": ["# Titulo\n"]}
    p.write_text(json.dumps(_caderno([md])), encoding="utf-8")
    h = apply_header.build_header("notebooks/exemplo.ipynb")
    assert apply_header.apply_ipynb(p, h, check=False) is True
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["cells"][0]["cell_type"] == "code" and "".join(nb["cells"][0]["source"]).startswith(apply_header.BEGIN)
    assert nb["cells"][1] == md
    assert sum("".join(c["source"]).startswith(apply_header.BEGIN) for c in nb["cells"]) == 1
    antes = p.read_text(encoding="utf-8")
    assert apply_header.apply_ipynb(p, h, check=False) is False and p.read_text(encoding="utf-8") == antes


def test_verificacao_de_notebook_nao_escreve(tmp_path):
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(_caderno([])), encoding="utf-8")
    antes = p.read_text(encoding="utf-8")
    assert apply_header.apply_ipynb(p, apply_header.build_header("notebooks/x.ipynb"), check=True) is True
    assert p.read_text(encoding="utf-8") == antes


def test_varredura_inclui_o_projeto_e_ignora_ambientes_e_dados():
    rels = {p.relative_to(ROOT).as_posix() for p in apply_header.iter_targets()}
    assert {"src/predict.py", "scripts/run_all.py", "notebooks/01_eda.ipynb", "tests/test_scripts.py"} <= rels
    assert not [r for r in rels if set(Path(r).parts) & apply_header.SKIP_DIRS]
    assert not [r for r in rels if r.startswith(".venv/")]


# ----------------------------------------------------------------------------
# registro de cabecalhos em src/header.py
# ----------------------------------------------------------------------------

def test_toda_sintese_cadastrada_aponta_para_um_arquivo_que_existe():
    inexistentes = [rel for rel in header.SINTESES if not (ROOT / rel).exists()]
    assert not inexistentes, f"sinteses de arquivos que nao existem mais: {inexistentes}"


def test_historico_so_cita_arquivos_com_sintese_e_esta_em_ordem_cronologica():
    assert set(header.HISTORICO) <= set(header.SINTESES)
    for rel in header.SINTESES:
        entradas = header.historico(rel)
        assert entradas, f"{rel} sem nenhuma entrada de historico"
        datas = [datetime.date.fromisoformat(d) for d, _, _ in entradas]
        assert datas == sorted(datas), f"{rel}: historico fora de ordem"
        assert all(autor.strip() and desc.strip() for _, autor, desc in entradas)


def test_integrantes_do_cabecalho_estao_completos():
    assert len(header.INTEGRANTES) >= 2
    for m in header.INTEGRANTES:
        assert m["nome"].strip() and m["tia"].isdigit() and len(m["tia"]) == 8
        assert "PREENCHER" not in m["nome"] + m["tia"]


# ----------------------------------------------------------------------------
# run_all
# ----------------------------------------------------------------------------

@pytest.fixture()
def run_all(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_all_sob_teste", ROOT / "scripts" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    comandos: list[tuple[str, ...]] = []
    monkeypatch.setattr(mod, "run", lambda *a: comandos.append(a))

    def executar(*argv: str, avaliado: bool, plataforma: str = "linux") -> list[tuple[str, ...]]:
        monkeypatch.setattr(mod, "sys", SimpleNamespace(platform=plataforma, executable=sys.executable))
        flag = ROOT / "tests" / "__init__.py"  # qualquer arquivo existente: teste ja avaliado
        monkeypatch.setattr(mod, "FINAL", flag if avaliado else ROOT / "nao_existe.json")
        monkeypatch.setattr(sys, "argv", ["run_all", *argv])
        comandos.clear()
        mod.main()
        return list(comandos)

    executar.mod = mod
    return executar


def test_primeira_execucao_treina_e_avalia_antes_do_resto(run_all):
    cmds = run_all(avaliado=False)
    assert cmds[0] == ("-m", "src.train") and cmds[1] == ("-m", "src.evaluate", "--final")
    assert cmds[2] == ("-m", "src.figures")


def test_com_o_teste_ja_avaliado_nao_retreina_por_padrao(run_all, capsys):
    cmds = run_all(avaliado=True)
    assert not any(c[:2] in {("-m", "src.train"), ("-m", "src.evaluate")} for c in cmds)
    assert "teste ja avaliado" in capsys.readouterr().out
    assert cmds[0] == ("-m", "src.figures")


def test_retrain_refaz_treino_e_avaliacao_juntos_com_force(run_all):
    cmds = run_all("--retrain", avaliado=True)
    assert cmds[0] == ("-m", "src.train", "--force")
    assert cmds[1] == ("-m", "src.evaluate", "--final", "--force")


def test_pipeline_executa_notebooks_cabecalho_relatorio_e_testes_nesta_ordem(run_all):
    cmds = run_all(avaliado=True)
    nomes = [c[-1] for c in cmds if c[:4] == ("-m", "jupyter", "nbconvert", "--to")]
    assert nomes == [f"notebooks/{n}.ipynb" for n in run_all.mod.NOTEBOOKS]
    indice = {c[0] if c[0] != "-m" else c[1]: i for i, c in enumerate(cmds)}
    assert indice["src.figures"] < indice["jupyter"] < indice["scripts/apply_header.py"] < indice["src.report"] < indice["pytest"]
    assert cmds[-1] == ("-m", "pytest", "-q")


def test_link_do_video_e_repassado_ao_relatorio_e_o_pdf_so_no_windows(run_all):
    url = "https://youtu.be/abc123"
    linux = [c for c in run_all("--video-url", url, avaliado=True) if c[:2] == ("-m", "src.report")][0]
    assert linux == ("-m", "src.report", "--video-url", url)
    windows = [c for c in run_all("--video-url", url, avaliado=True, plataforma="win32") if c[:2] == ("-m", "src.report")][0]
    assert windows == ("-m", "src.report", "--video-url", url, "--pdf")


def test_avisa_que_o_pdf_nao_foi_regenerado_fora_do_windows(run_all, capsys):
    run_all(avaliado=True)
    assert "relatorio_N2.pdf NAO foi regenerado" in capsys.readouterr().out


def test_cadernos_do_pipeline_existem():
    spec = importlib.util.spec_from_file_location("run_all_nomes", ROOT / "scripts" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for nome in mod.NOTEBOOKS:
        assert (ROOT / "notebooks" / f"{nome}.ipynb").exists(), nome
