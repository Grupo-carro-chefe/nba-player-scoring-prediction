# Dataset

Descrição do conjunto de dados utilizado no trabalho. A definição coluna a
coluna está em [`../docs/DICIONARIO_DE_DADOS.md`](../docs/DICIONARIO_DE_DADOS.md).

## Arquivo principal

`processed/dataset.csv` (24 MB) e `processed/dataset.parquet` (4,4 MB), mesmo
conteúdo em dois formatos.

| Propriedade | Valor |
|---|---|
| Grão | uma linha por jogador por partida |
| Chave | (`PLAYER_ID`, `GAME_ID`) |
| Linhas | 83.799 |
| Colunas | 37 |
| Período | 07/11/2021 a 12/04/2026 (5 temporadas regulares) |
| Jogadores | 753 |
| Partidas | 5.430 |
| Fonte | `stats.nba.com`, via biblioteca `nba_api` |

## O que cada grupo de colunas representa

- **Identificação (9 colunas):** jogador, time, adversário, jogo, data, temporada.
- **Features pré-jogo (24 colunas):** estatísticas calculadas **apenas com jogos
  anteriores**. Médias móveis de 3, 5 e 10 partidas, média exponencial com fator
  0,9, minutos, volume de arremesso, dias de descanso, jogo em sequência
  (*back-to-back*), mando de quadra, proxy de titularidade e força defensiva do
  adversário.
- **Alvos (3 colunas):** `y_pts` (pontos marcados, regressão), `line_synthetic`
  (mediana dos 10 jogos anteriores arredondada para 0,5) e `y_over` (1 se o
  jogador superou essa linha, classificação).
- **Controle (1 coluna):** `split`, com a divisão temporal treino/validação/teste.

## Divisão dos dados

| Split | Temporadas | Linhas |
|---|---|---|
| `train` | 2021-22, 2022-23, 2023-24 | 50.045 |
| `val` | 2024-25 | 16.653 |
| `test` | 2025-26 | 17.101 |

A divisão é temporal, nunca aleatória: o modelo treina no passado e é avaliado
no futuro, que é o cenário real de uso.

## Balanceamento do alvo

A classe positiva (jogador superou a linha) responde por **48,31%** das linhas.
O conjunto é equilibrado por construção, já que a linha é a mediana recente do
próprio jogador. Um classificador que chute sempre a classe majoritária acerta
cerca de 51,7%, que é a linha de base a ser superada.

## Como o dataset foi construído

A partir de **131.292** registros brutos de jogador por partida:

1. Remoção das linhas sem histórico suficiente para calcular a linha sintética,
   que exige 10 jogos anteriores na mesma temporada (2.814 linhas).
2. Recorte para jogadores de rotação, com média de ao menos 15 minutos nos
   5 jogos anteriores (6.839 linhas). O filtro usa informação anterior à
   partida, e não os minutos do jogo em questão.

Restam as 83.799 linhas do arquivo final.

## Arquivos nesta pasta

| Arquivo | Descrição | Versionado |
|---|---|---|
| `processed/dataset.csv` | Dataset analítico | sim |
| `processed/dataset.parquet` | Mesmo conteúdo, formato colunar | sim |
| `processed/dataset_manifest.json` | Metadados da construção: filtros, features, balanceamento | sim |
| `raw/manifest.json` | Metadados da coleta: data, endpoints, temporadas, contagens | sim |
| `raw/player_game_logs.csv` | Registros brutos por jogador (24 MB) | não |
| `raw/team_game_logs.csv` | Registros brutos por time (2 MB) | não |

Os arquivos brutos não são versionados por serem grandes e integralmente
reprodutíveis com `python -m src.collect`. O `raw/manifest.json` registra
exatamente qual coleta gerou o dataset publicado.

## Reprodução

```
python -m src.collect
python -m src.build_dataset
```

## Limitações

As restrições conhecidas do conjunto, incluindo a ausência de dados de lesão e
escalação, estão descritas em
[`../docs/LIMITACOES_E_VIESES.md`](../docs/LIMITACOES_E_VIESES.md).
