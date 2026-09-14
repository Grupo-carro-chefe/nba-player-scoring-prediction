# Dicionário de dados

Dataset analítico: `data/processed/dataset.csv` (e `.parquet` quando `pyarrow` está instalado).

**Grão:** uma linha por **jogador-jogo**.
**Chave:** (`PLAYER_ID`, `GAME_ID`).

A coluna **Momento** é a mais importante para avaliar o trabalho:

- **pré-jogo**: informação disponível **antes da bola subir**. Pode ser usada como feature (`X`).
- **pós-jogo**: só existe depois que o jogo acabou. É **alvo** (`y`) e nunca pode entrar em `X`.
- **contexto**: identificação/metadado. Não é feature nem alvo (mas `is_home` é exceção: é conhecido antes e vira feature).

---

## Identificação e contexto

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `PLAYER_ID` | int | Identificador do jogador na base da NBA | nba_api / LeagueGameLog | contexto |
| `PLAYER_NAME` | str | Nome do jogador | nba_api / LeagueGameLog | contexto |
| `TEAM_ABBREVIATION` | str | Sigla do time do jogador (ex.: `LAL`) | nba_api / LeagueGameLog | contexto |
| `OPPONENT` | str | Sigla do adversário, extraída de `MATCHUP` | derivada (`src/collect.py`) | pré-jogo |
| `GAME_ID` | str | Identificador do jogo | nba_api / LeagueGameLog | contexto |
| `GAME_DATE` | datetime | Data do jogo | nba_api / LeagueGameLog | pré-jogo |
| `SEASON` | str | Temporada no formato `2023-24` | parâmetro da coleta | contexto |
| `SEASON_TYPE` | str | `Regular Season` ou `Playoffs` | parâmetro da coleta | contexto |
| `IS_PLAYOFF` | bool | `True` se o jogo é de playoffs | derivada | contexto |
| `split` | str | `train` / `val` / `test` (split **temporal**) | derivada (`src/build_dataset.py`) | contexto |

---

## Features pré-jogo: produção recente do jogador

Todas calculadas por (`PLAYER_ID`, `SEASON`) e **shiftadas em 1 jogo**: a linha do jogo *t* só enxerga jogos < *t*, e a janela **zera a cada temporada**.

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `pts_ma3_prev` | float | Média de pontos dos 3 jogos anteriores | derivada de `PTS` | pré-jogo |
| `pts_ma5_prev` | float | Média de pontos dos 5 jogos anteriores | derivada de `PTS` | pré-jogo |
| `pts_ma10_prev` | float | Média de pontos dos 10 jogos anteriores | derivada de `PTS` | pré-jogo |
| `pts_ewm_prev` | float | Média **exponencial** de pontos (peso `0.9^k`, meia-vida ≈ 6,6 jogos) | derivada de `PTS` | pré-jogo |
| `pts_std5_prev` | float | Desvio-padrão dos pontos nos 5 jogos anteriores (volatilidade) | derivada de `PTS` | pré-jogo |
| `pts_season_mean_prev` | float | Média de pontos acumulada na temporada até o jogo anterior | derivada de `PTS` | pré-jogo |

## Features pré-jogo: oportunidade (minutos e volume)

Minutos e volume de arremesso são os preditores mais fortes de pontos: quem não joga, não pontua.

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `min_ma3_prev` | float | Média de minutos dos 3 jogos anteriores | derivada de `MIN` | pré-jogo |
| `min_ma5_prev` | float | Média de minutos dos 5 jogos anteriores | derivada de `MIN` | pré-jogo |
| `min_ma10_prev` | float | Média de minutos dos 10 jogos anteriores | derivada de `MIN` | pré-jogo |
| `min_ewm_prev` | float | Média exponencial de minutos (peso `0.9^k`) | derivada de `MIN` | pré-jogo |
| `min_season_mean_prev` | float | Média de minutos acumulada na temporada | derivada de `MIN` | pré-jogo |
| `fga_ma5_prev` | float | Média de tentativas de arremesso de quadra (5 jogos) | derivada de `FGA` | pré-jogo |
| `fg3a_ma5_prev` | float | Média de tentativas de 3 pontos (5 jogos) | derivada de `FG3A` | pré-jogo |
| `fta_ma5_prev` | float | Média de tentativas de lance livre (5 jogos) | derivada de `FTA` | pré-jogo |

## Features pré-jogo: contexto do jogador

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `reb_ma5_prev` | float | Média de rebotes (5 jogos): proxy de papel/posição | derivada de `REB` | pré-jogo |
| `ast_ma5_prev` | float | Média de assistências (5 jogos): proxy de papel de criação | derivada de `AST` | pré-jogo |
| `games_played_prev` | int | Jogos já disputados pelo jogador **naquela temporada** (0 no primeiro) | derivada | pré-jogo |
| `starter_proxy_prev` | Int64 (0/1) | **PROXY** de titularidade: `1` se `min_ma5_prev ≥ 28`. Ver ressalva abaixo | derivada de `MIN` | pré-jogo |

> ⚠️ **`starter_proxy_prev` não é a escalação oficial.** O endpoint de game log não expõe `START_POSITION`; obter a escalação real exigiria uma requisição de box score **por jogo** (dezenas de milhares), inviável sob rate limit. Usamos minutos recentes como proxy declarado. Consequência: um titular recém-promovido aparece como reserva por alguns jogos, e um titular voltando de lesão idem. Está listado em `docs/LIMITACOES_E_VIESES.md`.

## Features pré-jogo: calendário e mando

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `rest_days` | float | Dias desde o jogo anterior do jogador (NaN no 1º da temporada) | derivada de `GAME_DATE` | pré-jogo |
| `is_b2b` | Int64 (0/1) | `1` se é *back-to-back* (`rest_days == 1`) | derivada | pré-jogo |
| `is_home` | Int64 (0/1) | `1` se o jogador joga em casa | derivada de `MATCHUP` | pré-jogo |

## Features pré-jogo: força do adversário

Calculadas a partir do log **por time** e shiftadas: na linha do jogo *t*, o valor é a média do adversário **até o jogo anterior dele**.

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `opp_def_pts_allowed_prev` | float | Média de pontos que o adversário cedeu por jogo, na temporada, até antes deste jogo | derivada de `team_game_logs` | pré-jogo |
| `opp_pace_proxy_prev` | float | Proxy de ritmo: média do total de pontos dos jogos do adversário | derivada de `team_game_logs` | pré-jogo |
| `opp_games_played_prev` | int | Jogos já disputados pelo adversário na temporada | derivada | pré-jogo |

> Usamos pontos cedidos calculados dos próprios jogos em vez de um *defensive rating* de temporada fechada **de propósito**: o rating publicado resume a temporada inteira, incluindo jogos que ainda não aconteceram na data da linha, o que é vazamento
clássico.

---

## Linha de referência e alvos

| Coluna | Tipo | Descrição | Fonte | Momento |
|---|---|---|---|---|
| `line_synthetic` | float | **Linha sintética**: mediana dos pontos dos 10 jogos anteriores, arredondada para `.5` | derivada (`src/target.py`) | pré-jogo |
| `y_pts` | int | **Alvo de regressão**: pontos marcados no jogo | nba_api / `PTS` | **pós-jogo** |
| `y_over` | int (0/1) | **Alvo de classificação**: `1` se `y_pts > line_synthetic` | derivada | **pós-jogo** |

Notas sobre os alvos:

- A linha é arredondada para `.5` e os pontos são inteiros, então **nunca há empate** (*push*): o alvo binário é sempre bem definido.
- `y_over` é `NaN` (e a linha é descartada) enquanto o jogador não tem 10 jogos na temporada. Não viram zero.
- A justificativa completa da linha sintética (e por que não usamos linha de mercado) está em `src/target.py` e em `docs/AUDITORIA_TECNICA.md`.

---

## Colunas do bruto que **não** entram no dataset

Ficam apenas em `data/raw/player_game_logs.*`, como material de auditoria. São todas **pós-jogo** e não podem virar feature:

`PTS` (vira `y_pts`), `MIN`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `PLUS_MINUS`, `WL`.

A lista viva está em `src/features.py::POST_GAME_COLUMNS` e o teste `tests/test_no_leakage.py::test_no_post_game_column_in_feature_list` falha se alguma dessas vazar para as features.
