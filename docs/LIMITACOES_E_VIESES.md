# Limitações e vieses conhecidos do dataset

Documento de honestidade metodológica. Cada item traz **o que é**, **o impacto
esperado** e **como mitigamos ou por que aceitamos**.

---

## A. Vieses de amostragem — quem está no dataset

### A1. Seleção por minutos (o mais relevante)
Aplicamos `min_ma5_prev ≥ 15` para recortar jogadores de rotação.

- **Impacto:** o dataset **não representa a liga inteira**. Fim de banco,
  atletas de contrato de 10 dias e chamados da G-League ficam de fora. Um modelo
  treinado aqui não deve ser aplicado a esses perfis.
- **Mitigação:** o filtro usa **informação pré-jogo** (média de minutos anterior),
  não os minutos do jogo em questão — então é recorte de população, não vazamento.
  O parâmetro é ajustável (`--min-minutes-prev`) e o número de linhas descartadas
  fica registrado em `dataset_manifest.json`.

### A2. Sobre-representação de quem joga muito
Cada jogador contribui com uma linha por jogo. Quem disputa 78 jogos aparece 78
vezes; quem se lesiona no meio, 30.

- **Impacto:** estrelas duráveis dominam o conjunto; métricas agregadas refletem
  mais esse perfil do que o jogador mediano.
- **Mitigação:** no 2º bimestre, reportar métricas **também por faixa de minutos**
  e considerar validação agrupada por jogador.

### A3. Recorte temporal exclui as temporadas de COVID
2019-20 e 2020-21 foram removidas de propósito (bolha em Orlando, calendário
encurtado, ginásios vazios).

- **Impacto:** menos dados e ausência de um regime atípico que poderia ensinar
  algo sobre o efeito do público no mando de campo.
- **Mitigação:** decisão consciente e documentada; são outliers de contexto que
  distorceriam médias móveis e o efeito casa/fora.

---

## B. Limitações de cobertura — o que falta no dado

### B1. Sem informação de lesão e escalação  ⟵ maior fonte de erro irredutível
Não sabemos quem está fora, quem é dúvida, nem quem foi promovido a titular.

- **Impacto:** o caso mais informativo do basquete — o astro fica fora e o reserva
  recebe 34 minutos — aparece como uma "surpresa" que o modelo não tinha como
  prever. Isso coloca um **teto na acurácia** alcançável.
- **Mitigação parcial:** as médias móveis de minutos capturam a mudança **depois**
  que ela acontece (com 1 a 3 jogos de atraso). No relatório, tratar isso como
  limite estrutural, não como falha de modelagem.

### B2. `starter_proxy_prev` é proxy, não escalação
Derivado de `min_ma5_prev ≥ 28`, porque o game log não expõe `START_POSITION` e
o box score exigiria uma requisição por jogo.

- **Impacto:** erra sistematicamente em janelas de transição (promoção a titular,
  retorno de lesão, mudança de treinador).
- **Mitigação:** está declarado como proxy no dicionário de dados e no código.

### B3. Troca de time no meio da temporada
A janela agrupa por (`PLAYER_ID`, `SEASON`) — **não** reseta quando o jogador
muda de time.

- **Impacto:** logo após uma troca, as médias móveis descrevem o papel do jogador
  no time **antigo**, que pode ser bem diferente.
- **Mitigação:** conhecida e não corrigida nesta entrega. Correção natural no 2º
  bimestre: agrupar por (`PLAYER_ID`, `SEASON`, `TEAM_ABBREVIATION`), ao custo de
  zerar o histórico e perder linhas.

### B4. Contexto físico ausente
Não temos distância de viagem, fuso horário, altitude (Denver) nem sequência de
jogos fora de casa. Só `rest_days`, `is_b2b` e `is_home`.

### B5. *Garbage time* não identificado
Jogos decididos cedo alteram minutos e pontos de forma não modelada — em ambas as
direções (titular poupado, reserva inflado).

---

## C. Limitações metodológicas

### C1. O alvo não é a linha do mercado
`line_synthetic` é a mediana móvel do próprio jogador, não a linha de uma casa.

- **Impacto:** o problema resolvido é "o jogador supera o próprio patamar
  recente?", que é **previsão de desempenho** — não valor de aposta. Nenhum
  resultado deste trabalho permite concluir lucratividade em mercado real.
- **Justificativa:** não existe fonte gratuita, completa e reproduzível de linhas
  históricas de props (detalhes em `docs/AUDITORIA_TECNICA.md`). Preferimos um
  alvo reproduzível e honesto a um alvo de origem duvidosa.

### C2. Alvo auto-referencial
Como a linha nasce do histórico do próprio jogador, ela **persegue** o desempenho:
quem melhora vê a própria linha subir alguns jogos depois.

- **Impacto:** o alvo fica próximo de 50/50 por construção (medimos ~48%), o que é
  bom para balanceamento, mas significa que boa parte da variância é ruído
  genuíno — o teto de acurácia é modesto.
- **Mitigação:** entregamos também `y_pts` (regressão), que não sofre desse efeito
  e é o alvo primário mais defensável.

### C3. Observações não são independentes
O mesmo jogador aparece centenas de vezes; companheiros de equipe compartilham o
mesmo jogo. A suposição i.i.d. é violada.

- **Impacto:** intervalos de confiança ingênuos ficam **otimistas demais**.
- **Mitigação:** split temporal (já feito) e, no 2º bimestre, considerar validação
  agrupada por jogador além da cronológica.

### C4. Perda das primeiras partidas de cada temporada
A linha sintética exige 10 jogos de histórico; as anteriores saem do dataset.

- **Impacto:** perdemos justamente o início de temporada, período de maior
  incerteza e de definição de papéis.
- **Mitigação:** aceito. A alternativa (imputar com a temporada anterior) traria
  de volta a contaminação entre temporadas que a auditoria condenou.

### C5. Deriva entre temporadas (*drift*)
A liga muda: ritmo, volume de bolas de 3, regras de falta. Treinamos em 2021-24 e
testamos em 2025-26.

- **Impacto:** parte do erro no teste é deriva, não falha do modelo.
- **Mitigação:** é **proposital** — reproduz o cenário real de uso (prever o
  futuro com o passado). Reportar a métrica por temporada para separar os efeitos.

---

## D. Reprodutibilidade

### D1. Fonte única
Tudo vem de `stats.nba.com` via `nba_api`. Não há validação cruzada com uma
segunda fonte, e mudanças no endpoint quebram a coleta.

- **Mitigação:** `data/raw/manifest.json` registra data da coleta, endpoints,
  versão da biblioteca, filtros e contagem de linhas; o cache local
  (`.cache_nba/`) permite reconstruir o dataset sem novas requisições.

### D2. Dado vivo
A API reflete o estado atual da base. Correções retroativas de estatística ou uma
temporada ainda em andamento fazem duas coletas em datas diferentes divergirem.

- **Mitigação:** sempre citar o `coletado_em_utc` do manifest ao reportar números.

---

## Resumo para o relatório

O dataset é **adequado para prever o desempenho de jogadores de rotação da NBA em
temporada regular**, com um teto de acurácia imposto principalmente pela ausência
de dados de lesão e escalação (B1). Ele **não** serve para: avaliar jogadores fora
da rotação (A1), analisar playoffs (A3/C5) ou sustentar qualquer conclusão sobre
mercados de aposta (C1).
