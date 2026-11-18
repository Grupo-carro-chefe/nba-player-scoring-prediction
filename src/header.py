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
#   2026-11-15 | Rodrigo    | Histórico por arquivo e sínteses dos arquivos da N2.
#   2026-11-17 | Rodrigo    | Sínteses e histórico dos novos arquivos de teste.
# === FIM DO CABECALHO ===

"""Metadados do grupo usados pelo cabecalho obrigatorio.

Preencha INTEGRANTES, registre a sintese de cada arquivo em SINTESES e as
alteracoes em HISTORICO, e rode `python scripts/apply_header.py`.

O historico e POR ARQUIVO: cada entrada nova aparece so no cabecalho do arquivo
que mudou. A entrada da versao inicial (N1) vale para os arquivos que ja
existiam naquela entrega.
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
    "src/__init__.py": "Pacote Python do projeto (módulos de coleta, dados e modelagem).",
    "tests/__init__.py": "Pacote de testes automatizados (pytest).",
    "src/header.py": "Metadados do grupo para o cabeçalho obrigatório.",
    "src/collect.py": "Coleta de game logs da NBA com cache, retry e manifest.",
    "src/features.py": "Features pré-jogo (médias móveis shiftadas, calendário, adversário).",
    "src/target.py": "Alvos de regressão e classificação com linha sintética.",
    "src/build_dataset.py": "Pipeline de coleta bruta até dataset analítico com split temporal.",
    "src/io_utils.py": "Leitura e escrita de tabelas em parquet com fallback para CSV.",
    "src/modeling.py": "Dados, splits temporais, famílias de features, baselines e catálogo de modelos.",
    "src/metrics.py": "Métricas de regressão, classificação, calibração e bootstrap por jogador.",
    "src/uncertainty.py": "Intervalos conformais por estimativa e oráculos do teto preditivo.",
    "src/train.py": "Treino com validação cruzada temporal e escolhas feitas na validação.",
    "src/evaluate.py": "Avaliação final, única, no conjunto de teste (2025-26).",
    "src/figures.py": "Figuras do relatório a partir dos resultados salvos.",
    "src/predict.py": "Artefato de demonstração: pontos previstos, intervalo e probabilidade.",
    "src/report.py": "Gera o relatório da N2 (.docx) a partir dos resultados.",
    "scripts/apply_header.py": "Aplica o cabeçalho obrigatório nos arquivos .py e .ipynb.",
    "scripts/run_all.py": "Executa o pipeline da N2 de ponta a ponta (treino, teste, figuras, notebooks).",
    "tests/test_no_leakage.py": "Testes de vazamento temporal nas features.",
    "tests/test_documentacao.py": "Testes que mantêm a documentação coerente com o dataset gerado.",
    "tests/test_splits.py": "Testes da divisão temporal treino/validação/teste e das dobras do CV.",
    "tests/test_modelagem.py": "Testes da modelagem: features, pipeline, métricas, intervalos, sementes.",
    "tests/test_resultados.py": "Testes que prendem README e relatório aos números de test_final.json.",
    "tests/conftest.py": "Configuração compartilhada dos testes (marcador slow).",
    "tests/test_dados.py": "Invariantes do dataset versionado e coerência dos manifestos e do dicionário.",
    "tests/test_features_e_alvos.py": "Features, linha sintética e alvos com valores calculados à mão.",
    "tests/test_build_dataset.py": "Pipeline de construção do dataset de ponta a ponta em dados sintéticos.",
    "tests/test_collect_io.py": "Coleta (API simulada), cache, nova tentativa e leitura/escrita de tabelas.",
    "tests/test_unidades_modelagem.py": "Métricas, splits, catálogo de modelos, conformal e oráculos em casos pequenos.",
    "tests/test_predict.py": "Artefato de demonstração: consultas, recusas, formato e reprodução dos resultados.",
    "tests/test_artefatos.py": "Métricas do JSON recalculadas das previsões e travas do protocolo de teste único.",
    "tests/test_scripts.py": "Cabeçalho obrigatório (apply_header) e ordem do pipeline em run_all.",
    "tests/test_entrega.py": "Checklist do enunciado: relatório, PDF, link do vídeo, links, notebooks e git.",
    "tests/test_pipeline_fim_a_fim.py": "Treino, teste único e figuras com dados reduzidos; treino não lê o teste.",
    "tests/test_relatorio.py": "Relatório: formatação pt-BR, fatos verificados, geração do .docx e coerência com o versionado.",
    "notebooks/01_eda.ipynb": "Análise exploratória do dataset.",
    "notebooks/02_baselines_e_regressao.ipynb": "Baselines e modelos de regressão: validação temporal e teste.",
    "notebooks/03_classificacao_e_calibracao.ipynb": "Classificação contra o patamar recente e calibração.",
    "notebooks/04_incerteza_teto_e_erros.ipynb": "Intervalos, teto preditivo, ablação e análise de erros.",
    "notebooks/05_demo_artefato.ipynb": "Demonstração do artefato em jogos da temporada de teste.",
}

# Arquivos entregues na N1 recebem a entrada da versao inicial.
ARQUIVOS_N1 = {
    "src/__init__.py", "tests/__init__.py", "src/header.py", "src/collect.py", "src/features.py", "src/target.py",
    "src/build_dataset.py", "src/io_utils.py", "scripts/apply_header.py",
    "tests/test_no_leakage.py", "tests/test_documentacao.py", "notebooks/01_eda.ipynb",
}
HISTORICO_N1: tuple[str, str, str] = ("2026-09-14", "grupo", "Versão inicial: coleta, features, alvos, testes e EDA.")

_N2 = "2026-11-15"
_N3 = "2026-11-17"  # ampliacao da suite de testes
HISTORICO: dict[str, list[tuple[str, str, str]]] = {
    "src/collect.py": [(_N2, "Rodrigo", "Manifesto gravado com fim de linha LF.")],
    "src/build_dataset.py": [(_N2, "Rodrigo", "Manifesto gravado com fim de linha LF.")],
    "src/header.py": [
        (_N2, "Rodrigo", "Histórico por arquivo e sínteses dos arquivos da N2."),
        (_N3, "Rodrigo", "Sínteses e histórico dos novos arquivos de teste."),
    ],
    "scripts/apply_header.py": [
        (_N2, "Rodrigo", "Usa o histórico por arquivo de src/header.py."),
        (_N2, "Rodrigo", "Grava sempre com fim de linha LF."),
    ],
    "src/target.py": [(_N2, "Rodrigo", "Corrige a docstring: linha inteira admite empate (y_pts == linha).")],
    "tests/test_no_leakage.py": [(_N2, "Rodrigo", "Comparação sem caixa das colunas pós-jogo; teste do futuro para opp_*.")],
    "tests/test_documentacao.py": [
        (_N2, "Rodrigo", "Teste de que todo arquivo-fonte tem síntese e cabeçalho atualizado."),
        (_N2, "Rodrigo", "Reusa a lista de arquivos do apply_header."),
    ],
    "src/__init__.py": [(_N2, "Rodrigo", "Síntese cadastrada no cabeçalho.")],
    "tests/__init__.py": [(_N2, "Rodrigo", "Síntese cadastrada no cabeçalho.")],
    "src/modeling.py": [
        (_N2, "Rodrigo", "Versão inicial: splits, famílias, baselines e catálogo de modelos."),
        (_N2, "Rodrigo", "Docstring: a regressão linear sobre a exponencial não é a régua mais exigente em MAE."),
    ],
    "src/metrics.py": [(_N2, "Rodrigo", "Versão inicial: métricas, ECE e bootstrap pareado por jogador.")],
    "src/uncertainty.py": [
        (_N2, "Rodrigo", "Versão inicial: conformal normalizado e oráculos do teto."),
        (_N2, "Rodrigo", "Oráculos em mediana (régua justa para o MAE)."),
        (_N2, "Rodrigo", "Corrige o quantil conformal: k-ésima estatística de ordem."),
    ],
    "src/train.py": [
        (_N2, "Rodrigo", "Versão inicial: CV temporal, seleção, calibração e conformal."),
        (_N2, "Rodrigo", "Hash do código nos resultados; JSON com fim de linha LF."),
        (_N2, "Rodrigo", "Trava: não retreina com o teste já avaliado sem --force."),
    ],
    "src/evaluate.py": [
        (_N2, "Rodrigo", "Versão inicial: avaliação única no teste, ablação e teto."),
        (_N2, "Rodrigo", "Diagnósticos posteriores: regressão mediana e modelo com nível oráculo."),
        (_N2, "Rodrigo", "JSON com fim de linha LF."),
    ],
    "src/figures.py": [
        (_N2, "Rodrigo", "Versão inicial: figuras do relatório."),
        (_N2, "Rodrigo", "Figura da distribuição dos pontos (EDA) e baseline em mediana na escada."),
        (_N2, "Rodrigo", "Título e eixo calculados a partir dos dados."),
        (_N3, "Rodrigo", "Figura de dispersão previsto x observado com distribuições marginais."),
    ],
    "src/predict.py": [
        (_N2, "Rodrigo", "Versão inicial: demonstração com intervalo e aviso de escopo."),
        (_N2, "Rodrigo", "Saída em UTF-8 também quando redirecionada (Windows)."),
        (_N2, "Rodrigo", "--team sem data mostra só o jogo mais recente."),
        (_N2, "Rodrigo", "Recusa nome ambíguo e --sample fora do intervalo."),
        (_N3, "Rodrigo", "--sample 0 passa a ser recusado com a mensagem de intervalo (antes caía em 'informe --player')."),
    ],
    "src/report.py": [
        (_N2, "Rodrigo", "Versão inicial: relatório N2 gerado a partir dos resultados."),
        (_N2, "Rodrigo", "Correções da revisão crítica: teto, baselines, ablação, conformal, referências."),
        (_N2, "Rodrigo", "Fatos do texto verificados contra os resultados; números lidos dos arquivos."),
        (_N2, "Rodrigo", "Metadados próprios do .docx; texto do run_all --retrain."),
        ("2026-11-18", "Rodrigo", "Revisão de estilo do texto: menos negrito, frases mais diretas, mesmos números."),
    ],
    "scripts/run_all.py": [
        (_N2, "Rodrigo", "Versão inicial: pipeline da N2 de ponta a ponta."),
        (_N2, "Rodrigo", "Não retreina sem --retrain quando o teste já foi avaliado."),
        (_N2, "Rodrigo", "Passa --force ao treino e --pdf no Windows."),
    ],
    "tests/test_splits.py": [(_N2, "Rodrigo", "Versão inicial: não sobreposição e ordem temporal dos splits.")],
    "tests/test_modelagem.py": [
        (_N2, "Rodrigo", "Versão inicial: testes de modelagem."),
        (_N2, "Rodrigo", "Teste da estatística de ordem do conformal."),
    ],
    "tests/test_resultados.py": [
        (_N2, "Rodrigo", "Versão inicial: números citados batem com test_final.json."),
        (_N2, "Rodrigo", "Testes de proveniência (hash do código) e dos fatos do relatório."),
    ],
    "tests/conftest.py": [(_N3, "Rodrigo", "Versão inicial: registra o marcador slow.")],
    "tests/test_dados.py": [(_N3, "Rodrigo", "Versão inicial: invariantes do dataset versionado.")],
    "tests/test_features_e_alvos.py": [(_N3, "Rodrigo", "Versão inicial: features e alvos com valores conhecidos.")],
    "tests/test_build_dataset.py": [(_N3, "Rodrigo", "Versão inicial: pipeline do dataset em dados sintéticos.")],
    "tests/test_collect_io.py": [(_N3, "Rodrigo", "Versão inicial: coleta com API simulada e io_utils.")],
    "tests/test_unidades_modelagem.py": [(_N3, "Rodrigo", "Versão inicial: métricas, splits, modelos, conformal e oráculos.")],
    "tests/test_predict.py": [(_N3, "Rodrigo", "Versão inicial: artefato de demonstração.")],
    "tests/test_artefatos.py": [(_N3, "Rodrigo", "Versão inicial: métricas recalculadas e travas do protocolo.")],
    "tests/test_scripts.py": [(_N3, "Rodrigo", "Versão inicial: apply_header e run_all.")],
    "tests/test_entrega.py": [
        (_N3, "Rodrigo", "Versão inicial: checklist do enunciado da N2."),
        ("2026-11-18", "Rodrigo", "Link do vídeo casado pelo ID de 11 caracteres, sem absorver o texto seguinte."),
    ],
    "tests/test_pipeline_fim_a_fim.py": [(_N3, "Rodrigo", "Versão inicial: pipeline reduzido de ponta a ponta.")],
    "tests/test_relatorio.py": [(_N3, "Rodrigo", "Versão inicial: geração e coerência do relatório.")],
    "notebooks/02_baselines_e_regressao.ipynb": [(_N2, "Rodrigo", "Versão inicial.")],
    "notebooks/03_classificacao_e_calibracao.ipynb": [(_N2, "Rodrigo", "Versão inicial.")],
    "notebooks/04_incerteza_teto_e_erros.ipynb": [(_N2, "Rodrigo", "Versão inicial.")],
    "notebooks/05_demo_artefato.ipynb": [(_N2, "Rodrigo", "Versão inicial.")],
}


def historico(rel_path: str) -> list[tuple[str, str, str]]:
    """Entradas de historico de um arquivo, em ordem cronologica."""
    base = [HISTORICO_N1] if rel_path in ARQUIVOS_N1 else []
    return base + HISTORICO.get(rel_path, [])
