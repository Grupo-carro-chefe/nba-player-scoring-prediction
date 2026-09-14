# Previsão de desempenho de jogadores da NBA

Projeto da disciplina de **Inteligência Artificial**: Universidade Presbiteriana Mackenzie.

O objetivo é prever **quantos pontos um jogador da NBA marca em uma partida**,
usando apenas informação disponível antes do jogo começar. O enquadramento é de
previsão de desempenho esportivo: o repositório não contém cálculo de valor
esperado, gestão de banca nem qualquer camada relacionada a apostas.

- **1º bimestre (conteúdo atual):** coleta, dataset, análise exploratória e relatório.
- **2º bimestre:** modelagem, avaliação e calibração.

---

## Sumário

- [Como funciona](#como-funciona)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Dados gerados](#dados-gerados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Decisões metodológicas](#decisões-metodológicas)
- [Cabeçalho obrigatório](#cabeçalho-obrigatório)
- [Testes](#testes)
- [Documentação](#documentação)
- [Solução de problemas](#solução-de-problemas)

---

## Como funciona

O pipeline tem três etapas independentes:

```
  nba_api                src/collect.py            src/build_dataset.py
(stats.nba.com)  ──────►  data/raw/*.csv   ──────►  data/processed/dataset.csv
                          + manifest.json           + dataset_manifest.json
                                                    (features + alvos + split)
```

1. **Coleta** (`src/collect.py`): baixa os game logs da temporada, no grão de
   uma linha por jogador-jogo, com cache local e limite de requisições.
2. **Features e alvos** (`src/features.py`, `src/target.py`): transforma o
   histórico em variáveis pré-jogo e define o que será previsto.
3. **Dataset** (`src/build_dataset.py`): junta tudo, aplica os filtros e marca
   a divisão treino/validação/teste.

---

## Requisitos

- Python 3.11 ou superior
- Conexão com a internet na primeira coleta (depois o cache local resolve)

Bibliotecas (em `requirements.txt`): `nba_api`, `pandas`, `numpy`, `pyarrow`,
`matplotlib`, `seaborn`, `jupyter`, `pytest`.

---

## Instalação

```bash
git clone https://github.com/Grupo-carro-chefe/nba-player-scoring-prediction.git
cd nba-player-scoring-prediction

python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# Linux / macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Preencher a identificação do grupo

Edite a lista `INTEGRANTES` em [`src/header.py`](src/header.py):

```python
INTEGRANTES = [
    {"nome": "Fulano de Tal", "tia": "12345678"},
    {"nome": "Ciclana de Tal", "tia": "87654321"},
]
```

Depois propague o cabeçalho obrigatório para todos os arquivos:

```bash
python scripts/apply_header.py
```

### 2. Coletar os dados

```bash
python -m src.collect
```

Baixa 5 temporadas regulares (2021-22 a 2025-26): 10 requisições com pausa de
1,2 s entre elas, cerca de 131 mil linhas de jogador-jogo. As respostas ficam em
`.cache_nba/`, então rodar de novo não gera tráfego novo.

Opções:

```bash
python -m src.collect --seasons 2023-24 2024-25   # recorte menor (mais rápido)
python -m src.collect --include-playoffs          # inclui playoffs, marcados em IS_PLAYOFF
python -m src.collect --out caminho/alternativo
```

### 3. Montar o dataset

```bash
python -m src.build_dataset
```

Saída esperada:

```
dataset: 83,799 linhas x 37 colunas -> data/processed/dataset.*
split  : {'train': 50045, 'test': 17101, 'val': 16653}
classe positiva: 0.4831
```

Opções:

```bash
python -m src.build_dataset --min-minutes-prev 20   # recorte mais restrito de rotação
```

### 4. Rodar os testes

```bash
pytest -q
```

### 5. Abrir a análise exploratória

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Dados gerados

O dataset analítico e os manifests ficam versionados, por serem itens de
entrega. Os arquivos brutos não, por serem grandes e reproduzíveis pelos scripts.

| Arquivo | Conteúdo | No repositório |
|---|---|---|
| `data/processed/dataset.csv` | **Dataset analítico** (24 MB): features pré-jogo, alvos e split | sim |
| `data/processed/dataset.parquet` | Mesmo conteúdo em formato colunar (4,4 MB) | sim |
| `data/processed/dataset_manifest.json` | Metadados da construção: features, filtros, balanceamento | sim |
| `data/raw/manifest.json` | Metadados da coleta: data, endpoints, versão da lib, contagens | sim |
| `data/raw/player_game_logs.csv` | Bruto, uma linha por jogador-jogo (24 MB) | não |
| `data/raw/team_game_logs.csv` | Bruto por time, base da defesa do adversário (2 MB) | não |

A descrição do dataset (dimensões, colunas, divisão, balanceamento e como foi
construído) está em [`data/README.md`](data/README.md).

Os arquivos também são gravados em `.parquet` quando `pyarrow` está instalado.
O CSV é sempre gerado, então o projeto roda mesmo sem ele.

A descrição coluna a coluna está em
[`docs/DICIONARIO_DE_DADOS.md`](docs/DICIONARIO_DE_DADOS.md).

---

## Estrutura do repositório

```
├── src/
│   ├── header.py           # identificação do grupo (fonte do cabeçalho)
│   ├── collect.py          # coleta via nba_api
│   ├── features.py         # features pré-jogo
│   ├── target.py           # alvos e linha sintética
│   ├── build_dataset.py    # pipeline completo
│   └── io_utils.py         # parquet com fallback para CSV
├── scripts/
│   └── apply_header.py     # aplica o cabeçalho obrigatório
├── tests/
│   └── test_no_leakage.py  # testes de vazamento temporal
├── notebooks/
│   └── 01_eda.ipynb        # análise exploratória
├── docs/
│   ├── DICIONARIO_DE_DADOS.md
│   ├── AUDITORIA_TECNICA.md
│   └── LIMITACOES_E_VIESES.md
├── data/
│   ├── raw/                # dados brutos (não versionado)
│   └── processed/          # dataset analítico (não versionado)
├── GRUPO.md
├── requirements.txt
└── README.md
```

---

## Decisões metodológicas

### Recorte temporal

Cinco temporadas regulares: **2021-22 a 2025-26**. As temporadas 2019-20 e
2020-21 foram excluídas por serem atípicas (pandemia, bolha em Orlando,
calendário encurtado, ginásios sem público), o que distorceria médias móveis e o
efeito de mando de quadra.

### Features sem vazamento

Toda feature é uma estatística dos jogos **anteriores**. Na prática:

1. Ordena por jogador e data.
2. Agrupa por **(jogador, temporada)**.
3. Calcula a estatística móvel.
4. Aplica `shift(1)` dentro do grupo.

O agrupamento por temporada é o que impede a janela de arrastar jogos do ano
anterior (inclusive playoffs) para dentro da média da temporada corrente.
Existem testes automatizados cobrindo isso.

Grupos de features: médias móveis de 3, 5 e 10 jogos, média exponencial
(fator 0,9), minutos, volume de arremesso, dias de descanso, back-to-back,
mando de quadra, proxy de titularidade e força defensiva do adversário.

### Alvos

| Coluna | Tipo | Descrição |
|---|---|---|
| `y_pts` | regressão | pontos marcados no jogo |
| `y_over` | classificação | 1 se `y_pts` superou `line_synthetic` |
| `line_synthetic` | referência | mediana dos 10 jogos anteriores, arredondada para `.5` |

Não existe fonte gratuita, completa e reproduzível de linha histórica de
mercado, então usamos uma linha sintética construída apenas com dados
pré-jogo. O arredondamento para `.5` elimina empates, já que pontos são
inteiros. A justificativa e o trade-off estão em
[`docs/AUDITORIA_TECNICA.md`](docs/AUDITORIA_TECNICA.md).

### Divisão dos dados

Split **temporal**, nunca aleatório:

| Split | Temporadas |
|---|---|
| `train` | 2021-22, 2022-23, 2023-24 |
| `val` | 2024-25 |
| `test` | 2025-26 |

Um split aleatório colocaria jogos de dias vizinhos em conjuntos diferentes.
Como as features descrevem o estado recente do jogador, o modelo acabaria
treinando com informação do futuro daquele mesmo atleta.

---

## Cabeçalho obrigatório

Todo arquivo `.py` e `.ipynb` carrega, em comentário, a identificação dos
integrantes, a síntese do arquivo e o histórico de alterações. O bloco é
**gerado** a partir de `src/header.py`:

```bash
python scripts/apply_header.py          # aplica ou atualiza
python scripts/apply_header.py --check  # apenas verifica (sai com 1 se desatualizado)
```

Não edite o bloco manualmente nos arquivos: ele é sobrescrito na próxima
execução. Para registrar uma alteração, adicione uma linha em `HISTORICO`
(em `src/header.py`) e rode o script de novo.

---

## Testes

```bash
pytest -q
```

Seis testes cobrem vazamento temporal. O principal altera o resultado de um jogo
futuro e verifica que nenhuma feature das linhas anteriores muda, se mudasse,
haveria informação do futuro vazando para o passado. Os outros verificam o
cálculo das médias móveis, a linha sintética, o reset entre temporadas, a
ausência de colunas pós-jogo na lista de features e a inexistência de
correlação quase perfeita com o alvo.

---

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/DICIONARIO_DE_DADOS.md`](docs/DICIONARIO_DE_DADOS.md) | Todas as colunas: tipo, descrição, fonte e se é pré ou pós-jogo |
| [`docs/AUDITORIA_TECNICA.md`](docs/AUDITORIA_TECNICA.md) | Auditoria do estimador de probabilidade anterior: amostra pequena, contaminação entre temporadas, efeito do decaimento e ausência de regressão à média |
| [`docs/LIMITACOES_E_VIESES.md`](docs/LIMITACOES_E_VIESES.md) | Vieses de amostragem, lacunas de cobertura e limites metodológicos |
| [`data/README.md`](data/README.md) | Descrição do dataset entregue: dimensões, colunas, divisão e construção |
| [`GRUPO.md`](GRUPO.md) | Identificação dos integrantes |

---

## Solução de problemas

**`ModuleNotFoundError: No module named 'nba_api'`**
A instalação das dependências foi interrompida antes do fim (o `jupyter` é
grande e demora). Confira o que falta e complete:

```bash
python -c "import nba_api, matplotlib, seaborn"
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'src'`**
Rode os comandos a partir da raiz do repositório, usando `python -m src.collect`
(e não `python src/collect.py`).

**`Unable to find a usable engine` ao ler parquet**
Falta o `pyarrow`. Instale com `pip install pyarrow` ou ignore: os arquivos
`.csv` são gerados de qualquer forma e o pipeline funciona com eles.

**A coleta falha ou fica lenta**
A API pública da NBA limita requisições em rajada. O script já espera 1,2 s
entre chamadas e tenta novamente com espera crescente. Se persistir, aguarde
alguns minutos, o que já foi baixado fica em `.cache_nba/` e não será
requisitado outra vez.

**`FileNotFoundError` ao montar o dataset**
Rode `python -m src.collect` antes de `python -m src.build_dataset`.
