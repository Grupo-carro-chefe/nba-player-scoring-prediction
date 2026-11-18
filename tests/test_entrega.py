# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_entrega.py
# SINTESE: Checklist do enunciado: relatório, PDF, link do vídeo, links, notebooks e git.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: checklist do enunciado da N2.
#   2026-11-18 | Rodrigo    | Link do vídeo casado pelo ID de 11 caracteres, sem absorver o texto seguinte.
# === FIM DO CABECALHO ===

"""Checklist da entrega: o que o enunciado da N2 exige, conferido contra o repositorio.

Enunciado (Prof. Zerbinatti, 7J SI Noite): relatorio no template com 12 itens,
GitHub publico com relatorio, dataset com descricao, notebooks e fontes com
cabecalho, e o endereco do video no relatorio. Cada item que da para verificar
por programa esta aqui, para que um commit descuidado nao quebre a entrega sem
ninguem perceber.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src import header
from src.report import VIDEO_PLACEHOLDER, n_testes

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "relatorio" / "relatorio_N2.docx"
PDF = ROOT / "relatorio" / "relatorio_N2.pdf"
GRUPO = ROOT / "GRUPO.md"
GITHUB = "https://github.com/Grupo-carro-chefe/nba-player-scoring-prediction"

SECOES = [
    "Introdução", "Descrição do Problema", "Aspectos Éticos", "Conjunto de Dados",
    "Metodologia", "Resultados", "Conclusão", "Endereço GitHub", "Referências",
]
SUBSECOES = ["1.1. Contextualização", "1.2. Justificativa", "1.3. Objetivo", "1.4. Opção do projeto", "4.3. Análise exploratória"]
# O ID de um video do YouTube tem sempre 11 caracteres. Fixar o tamanho impede
# que, no texto sem espacos, o link absorva o que vem depois (ex.: "9. Referencias").
YOUTUBE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}")


def _paragrafos_docx(path: Path = DOCX) -> list[str]:
    docx = pytest.importorskip("docx")
    doc = docx.Document(str(path))
    partes = [p.text.strip() for p in doc.paragraphs]
    for t in doc.tables:
        partes += [c.text.strip() for row in t.rows for c in row.cells]
    return [p for p in partes if p]


def _texto_pdf() -> str:
    pypdf = pytest.importorskip("pypdf")
    return "\n".join(p.extract_text() for p in pypdf.PdfReader(str(PDF)).pages)


def _membros_do_grupo() -> list[tuple[str, str, str]]:
    linhas = re.findall(r"^\|\s*([^|]+?)\s*\|\s*(\d{8})\s*\|\s*([^|\s]+@[^|\s]+)\s*\|", GRUPO.read_text(encoding="utf-8"), flags=re.M)
    assert linhas, "tabela de integrantes nao encontrada em GRUPO.md"
    return linhas


# ----------------------------------------------------------------------------
# relatorio: os 12 itens do enunciado
# ----------------------------------------------------------------------------

def test_relatorio_tem_as_secoes_do_enunciado_numeradas_e_em_ordem():
    paras = _paragrafos_docx()
    posicoes = []
    for n, nome in enumerate(SECOES, start=1):
        idx = next((i for i, p in enumerate(paras) if p.startswith(f"{n}. {nome}")), None)
        assert idx is not None, f"secao '{n}. {nome}' ausente do relatorio"
        posicoes.append(idx)
    assert posicoes == sorted(posicoes), "secoes fora de ordem"
    for sub in SUBSECOES:
        assert any(p.startswith(sub) for p in paras), f"subsecao '{sub}' ausente"
    assert "Resumo" in paras and "Abstract" in paras


def test_relatorio_identifica_titulo_integrantes_professor_e_curso():
    texto = "\n".join(_paragrafos_docx())
    assert "Predição de Desempenho Individual de Atletas da NBA" in texto
    assert "Leandro Zerbinatti" in texto and "Mackenzie" in texto and "Inteligência Artificial" in texto
    for nome, ra, email in _membros_do_grupo():
        assert nome.split()[0] in texto and ra in texto and email in texto, f"{nome}: nome, RA ou e-mail ausente"


def test_integrantes_do_grupo_batem_com_o_cabecalho_dos_arquivos():
    grupo = {(nome, ra) for nome, ra, _ in _membros_do_grupo()}
    cabecalho = {(m["nome"], m["tia"]) for m in header.INTEGRANTES}
    assert grupo == cabecalho


def test_resumo_etica_e_referencias_tem_corpo():
    paras = _paragrafos_docx()

    def palavras_entre(inicio: str, fim: str) -> int:
        i = next(k for k, p in enumerate(paras) if p.startswith(inicio))
        j = next(k for k, p in enumerate(paras) if k > i and p.startswith(fim))
        return sum(len(p.split()) for p in paras[i + 1 : j])

    assert palavras_entre("Resumo", "Palavras-chave") >= 100
    assert palavras_entre("3. Aspectos Éticos", "4. Conjunto de Dados") >= 300
    i = next(k for k, p in enumerate(paras) if p.startswith("9. Referências"))
    assert len(paras[i + 1 :]) >= 8, "lista de referencias curta demais"


def test_relatorio_aponta_para_o_github_publico():
    texto = "".join(_paragrafos_docx()).replace(" ", "")
    assert GITHUB in texto
    assert GITHUB in GRUPO.read_text(encoding="utf-8")


def test_relatorio_nao_tem_marcas_de_rascunho():
    texto = "\n".join(_paragrafos_docx())
    for marca in ("PREENCHER", "TODO", "XXX", "lorem", "FIXME"):
        assert marca not in texto, marca


def test_relatorio_cita_a_quantidade_real_de_testes():
    """O relatorio diz quantos testes o projeto tem; a contagem sai de tests/, entao precisa ser regenerado ao mudar a suite."""
    texto = "\n".join(_paragrafos_docx())
    m = re.search(r"tem (\d+) testes automatizados", texto)
    assert m, "o relatorio nao cita o total de testes"
    assert int(m.group(1)) == n_testes(), f"relatorio diz {m.group(1)} testes, a suite tem {n_testes()}: rode python -m src.report --pdf"


# ----------------------------------------------------------------------------
# PDF entregue x DOCX x resultados
# ----------------------------------------------------------------------------

def test_pdf_entregue_e_o_mesmo_relatorio_do_docx_e_dos_resultados():
    from tests.test_resultados import _citados

    pdf = _texto_pdf()
    final = json.loads((ROOT / "results" / "metrics" / "test_final.json").read_text(encoding="utf-8"))
    ausentes = {k: v for k, v in _citados(final).items() if v not in pdf}
    assert not ausentes, f"o PDF nao cita os resultados atuais (PDF desatualizado em relacao ao JSON?): {ausentes}"
    for nome, ra, _ in _membros_do_grupo():
        assert ra in pdf
    m = re.search(r"tem (\d+)\s+testes", pdf)
    assert m and int(m.group(1)) == n_testes(), "PDF desatualizado em relacao a suite de testes"
    for n, nome in enumerate(SECOES, start=1):
        assert f"{n}. {nome}" in pdf, f"PDF sem a secao {n}. {nome}"


def test_pdf_tem_o_tamanho_de_um_relatorio_completo():
    pypdf = pytest.importorskip("pypdf")
    paginas = len(pypdf.PdfReader(str(PDF)).pages)
    assert 15 <= paginas <= 40, paginas


# ----------------------------------------------------------------------------
# link do video: o mesmo no relatorio, no PDF e no GRUPO.md
# ----------------------------------------------------------------------------

def test_link_do_video_e_consistente_entre_grupo_docx_e_pdf():
    fontes = {
        "GRUPO.md": GRUPO.read_text(encoding="utf-8"),
        "relatorio_N2.docx": "\n".join(_paragrafos_docx()),
    }
    try:
        fontes["relatorio_N2.pdf"] = _texto_pdf()
    except pytest.skip.Exception:
        pass
    links = {nome: set(YOUTUBE.findall(re.sub(r"\s+", "", texto))) for nome, texto in fontes.items()}
    todos = set().union(*links.values())
    assert len(todos) <= 1, f"links de video diferentes entre os arquivos: {links}"
    for nome, achados in links.items():
        if achados:
            continue
        # sem link ainda: tem de estar o aviso, e o aviso nao pode coexistir com link real
        aviso = "será adicionado" if nome == "GRUPO.md" else "será inserido"
        assert aviso in fontes[nome], f"{nome}: sem link do video e sem aviso de que ele sera inserido"
    if todos:
        faltam = [n for n, a in links.items() if not a]
        assert not faltam, f"o link do video esta em alguns arquivos mas nao em: {faltam}"
        assert VIDEO_PLACEHOLDER not in fontes["relatorio_N2.docx"]


# ----------------------------------------------------------------------------
# documentacao do repositorio
# ----------------------------------------------------------------------------

def test_links_relativos_dos_documentos_apontam_para_arquivos_que_existem():
    documentos = [ROOT / "README.md", GRUPO, ROOT / "relatorio" / "README.md", ROOT / "data" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    quebrados = []
    for md in documentos:
        for alvo in re.findall(r"\]\(([^)\s]+)\)", md.read_text(encoding="utf-8")):
            if alvo.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not (md.parent / alvo.split("#")[0]).resolve().exists():
                quebrados.append(f"{md.relative_to(ROOT)} -> {alvo}")
    assert not quebrados, f"links quebrados: {quebrados}"


def test_readme_menciona_todos_os_modulos_scripts_e_arquivos_de_teste():
    texto = (ROOT / "README.md").read_text(encoding="utf-8")
    esperados = [p.name for p in (ROOT / "src").glob("*.py") if p.name != "__init__.py"]
    esperados += [p.name for p in (ROOT / "scripts").glob("*.py")]
    esperados += [p.name for p in (ROOT / "tests").glob("test_*.py")]
    faltando = [n for n in esperados if n not in texto]
    assert not faltando, f"README nao menciona: {faltando}"


def test_readme_avisa_que_nao_e_recomendacao_de_aposta():
    texto = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "recomendação de aposta" in texto


def test_notebooks_sao_validos_executados_e_sem_erros():
    nbformat = pytest.importorskip("nbformat")
    cadernos = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [c.name[:2] for c in cadernos] == ["01", "02", "03", "04", "05"]
    for caderno in cadernos:
        nb = nbformat.read(caderno, as_version=4)
        nbformat.validate(nb)
        codigo = [c for c in nb.cells if c.cell_type == "code"]
        pos_cabecalho = codigo[1:]  # a primeira celula e o cabecalho (so comentario)
        assert codigo[0].source.startswith("# === CABECALHO"), caderno.name
        assert all(c.execution_count is not None for c in pos_cabecalho), f"{caderno.name}: celula nao executada"
        saidas = [o for c in codigo for o in c.get("outputs", [])]
        assert not [o for o in saidas if o.get("output_type") == "error"], f"{caderno.name}: celula com erro"
        bruto = caderno.read_text(encoding="utf-8")
        assert "Traceback (most recent call last)" not in bruto, caderno.name
        assert not re.search(r"[A-Za-z]:\\\\Users", bruto), f"{caderno.name}: caminho absoluto da maquina de quem executou"


def test_toda_dependencia_importada_esta_no_requirements():
    # IPython nao e instalado a parte: vem com jupyter/ipykernel, que estao no requirements
    padrao = {"sklearn": "scikit-learn", "docx": "python-docx", "IPython": "ipykernel"}
    opcionais = {"fastparquet"}  # importado dentro de try/except como alternativa ao pyarrow
    locais = {"src", "tests", "scripts", "apply_header"}
    declaradas = {
        re.split(r"[<>=!\s#]", linha.strip(), maxsplit=1)[0].lower().replace("_", "-")
        for linha in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.lstrip().startswith("#")
    }
    usados: set[str] = set()
    fontes = [*(ROOT / "src").glob("*.py"), *(ROOT / "scripts").glob("*.py"), *(ROOT / "tests").glob("*.py")]
    for arq in fontes:
        for no in ast.walk(ast.parse(arq.read_text(encoding="utf-8"))):
            if isinstance(no, ast.Import):
                usados |= {a.name.split(".")[0] for a in no.names}
            elif isinstance(no, ast.ImportFrom) and no.level == 0 and no.module:
                usados.add(no.module.split(".")[0])
    for arq in (ROOT / "notebooks").glob("*.ipynb"):
        for celula in json.loads(arq.read_text(encoding="utf-8"))["cells"]:
            if celula["cell_type"] == "code":
                usados |= set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", "".join(celula["source"]), flags=re.M))
    terceiros = {m for m in usados if m not in sys.stdlib_module_names and m not in locais and m not in opcionais}
    ausentes = sorted(m for m in terceiros if padrao.get(m, m).lower().replace("_", "-") not in declaradas)
    assert not ausentes, f"importados mas ausentes de requirements.txt: {ausentes}"


# ----------------------------------------------------------------------------
# escopo do projeto e organizacao do GitHub
# ----------------------------------------------------------------------------

def test_codigo_nao_tem_camada_de_apostas():
    """O README afirma que nao ha valor esperado, gestao de banca nem odds. Confere os nomes no codigo e no dataset."""
    proibidos = {"kelly", "bankroll", "stake", "odds", "sportsbook", "payout", "vig", "moneyline", "parlay", "ev"}
    achados = []
    for arq in [*(ROOT / "src").glob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        for no in ast.walk(ast.parse(arq.read_text(encoding="utf-8"))):
            nomes = []
            if isinstance(no, (ast.FunctionDef, ast.ClassDef)):
                nomes = [no.name]
            elif isinstance(no, ast.Name):
                nomes = [no.id]
            elif isinstance(no, ast.arg):
                nomes = [no.arg]
            elif isinstance(no, ast.Attribute):
                nomes = [no.attr]
            for nome in nomes:
                if set(nome.lower().split("_")) & proibidos or "expected_value" in nome.lower():
                    achados.append(f"{arq.name}: {nome}")
    assert not achados, f"nomes relacionados a apostas no codigo: {achados}"
    import pandas as pd

    colunas = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet").columns
    assert not [c for c in colunas if set(c.lower().split("_")) & proibidos]


def _arquivos_versionados() -> list[str]:
    try:
        r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git indisponivel")
    if r.returncode != 0:
        pytest.skip("nao e um repositorio git")
    return r.stdout.splitlines()


def test_entregaveis_estao_versionados_no_git():
    versionados = set(_arquivos_versionados())
    obrigatorios = [
        "README.md", "GRUPO.md", "requirements.txt", "relatorio/relatorio_N2.pdf", "relatorio/relatorio_N2.docx",
        "data/README.md", "data/processed/dataset.csv", "data/processed/dataset.parquet", "data/processed/dataset_manifest.json",
        "results/metrics/test_final.json", "results/metrics/treino_validacao.json",
        "results/models/regressao_final.joblib", "results/models/classificacao_final.joblib", "results/models/conformal.json",
        *(f"notebooks/{n}.ipynb" for n in ("01_eda", "02_baselines_e_regressao", "03_classificacao_e_calibracao", "04_incerteza_teto_e_erros", "05_demo_artefato")),
        *(f"src/{p.name}" for p in (ROOT / "src").glob("*.py")),
    ]
    faltando = [o for o in obrigatorios if o not in versionados]
    assert not faltando, f"entregaveis fora do git (ignorados pelo .gitignore?): {faltando}"


def test_git_nao_carrega_lixo_nem_arquivos_grandes_demais():
    versionados = _arquivos_versionados()
    lixo = [f for f in versionados if re.search(r"(^|/)(\.venv|__pycache__|\.pytest_cache|\.ipynb_checkpoints|\.cache_nba|node_modules)/|results/cache/|\.DS_Store|\.pyc$", f)]
    assert not lixo, f"arquivos que nao deveriam estar versionados: {lixo[:5]}"
    grandes = [(f, (ROOT / f).stat().st_size // 2**20) for f in versionados if (ROOT / f).exists() and (ROOT / f).stat().st_size > 50 * 2**20]
    assert not grandes, f"arquivos acima de 50 MB (limite do GitHub e 100 MB): {grandes}"
