# Auditoria técnica do modelo de probabilidade herdado

Este documento resume a auditoria feita sobre o estimador de probabilidade do
projeto de origem (um sistema de análise de desempenho de jogadores da NBA) e
explica **como cada achado moldou as decisões deste dataset**.

O estimador auditado funcionava assim: para uma linha `L` e uma estatística
(pontos), calculava-se a **frequência empírica ponderada** de o jogador superar
`L` nos últimos *n* jogos, com pesos que decaíam exponencialmente:

```
P(over) = Σ wᵢ · 1[xᵢ > L] / Σ wᵢ ,  com wᵢ = 0.9^k  (k = distância do jogo mais recente)
```

Quatro problemas, do mais grave ao mais sutil.

---

## 1. Amostra pequena: o intervalo de confiança engole o sinal

Estimar uma probabilidade binária com 10–20 observações produz uma incerteza que,
na prática, é maior que o efeito que se quer medir.

Para uma proporção, o erro-padrão no pior caso (`p = 0,5`) é `0,5/√n`:

| Janela nominal | Erro-padrão | IC 95% |
|---|---|---|
| 10 jogos | 15,8 pp | ± 31,0 pp |
| 20 jogos | 11,2 pp | ± 21,9 pp |
| 100 jogos | 5,0 pp | ± 9,8 pp |

Com 20 jogos, uma estimativa de "65%" tem intervalo de confiança que vai de ~43%
a ~87%. Qualquer decisão baseada na diferença entre 62% e 58% é ruído.

**O que fizemos:** abandonamos a frequência empírica como estimador. No 2º
bimestre, a probabilidade sai de um **modelo treinado sobre milhares de linhas**,
não da contagem dos últimos jogos de um único atleta, e será avaliada por
**Brier score** e **curva de calibração**, não só por acurácia.

---

## 2. Contaminação entre temporada regular e playoffs

Este foi o achado mais concreto. A janela deslizante era de 100 jogos **por
jogador**, e atravessava temporadas. Consequências:

- Durante a temporada regular, jogos de **playoffs da temporada anterior** ainda
  estavam dentro da janela.
- O sistema marcava o contexto como "playoffs" indevidamente e priorizava esses
  jogos no *lookback*.
- Resultado: a estimativa da temporada regular corrente ficava dominada por
  partidas de playoff antigas, com outro nível de defesa, outra rotação e outro
  ritmo.

Playoffs não são uma amostra a mais da mesma distribuição: a defesa aperta, a
rotação encurta, o ritmo cai. Misturar os dois contextos é misturar populações.

**O que fizemos:**

- Toda janela é agrupada por (`PLAYER_ID`, **`SEASON`**) e **zera a cada
  temporada**: ver `src/features.py::GROUP`.
- O recorte padrão é **só temporada regular**. Playoffs entram apenas se pedidos
  explicitamente (`--include-playoffs`) e vêm marcados em `IS_PLAYOFF`.
- Há um teste automatizado que falha se a janela vazar entre temporadas:
  `tests/test_no_leakage.py::test_window_resets_between_seasons`.

---

## 3. Decaimento exponencial: reduz o viés, mas destrói a amostra

O decaimento `0.9^k` existe por um motivo legítimo: dar mais peso à forma recente.
O problema é que ele **reduz o tamanho efetivo da amostra**, e isso não estava
sendo contabilizado.

Usando o tamanho efetivo de Kish, `n_eff = (Σwᵢ)² / Σwᵢ²`:

| Janela nominal | `n_eff` sem decaimento | `n_eff` com decaimento 0.9 | IC 95% (com decaimento) |
|---|---|---|---|
| 10 jogos | 10,0 | 9,2 | ± 32,4 pp |
| 20 jogos | 20,0 | **14,9** | ± 25,4 pp |
| 100 jogos | 100,0 | **19,0** | ± 22,5 pp |

O achado que mais importa está na última linha: **com decaimento 0.9, aumentar a
janela de 20 para 100 jogos leva o tamanho efetivo de 14,9 para apenas 19,0.**
A janela ficou 5× maior e a informação praticamente não mudou, porque o peso do
100º jogo mais antigo é `0.9^99 ≈ 0,00003`. Na prática, ele não existe.

Dois números que ajudam a entender o regime:

- **Meia-vida:** 6,6 jogos. Um jogo de 7 partidas atrás pesa metade do último.
- Os **5 jogos mais recentes concentram 46,6%** de todo o peso.

Ou seja, o estimador era, na prática, uma média de ~5 a 7 jogos com verniz de
"20 jogos".

**O que fizemos:** mantivemos o decaimento (`pts_ewm_prev`, mesma constante 0.9,
por continuidade com o projeto de origem), mas **como uma feature entre outras**,
lado a lado com médias simples de 3, 5 e 10 jogos. O modelo decide o peso de cada
uma; nós não fixamos o horizonte por decreto. E o `n_eff` está documentado aqui
para que o relatório não repita a ilusão de amostra grande.

---

## 4. Ausência de regressão à média

A frequência empírica é o estimador de máxima verossimilhança e **não regride à
média**. Um jogador que superou a linha em 8 dos 10 últimos jogos recebia 80%.
Mas, com `n = 10`, boa parte desse 80% é sorte: o valor verdadeiro está quase
certamente mais perto da média da população.

O efeito é sistemático: infla as extremidades da distribuição de probabilidade
e produz exatamente os casos que *parecem* as melhores oportunidades. É o
mecanismo clássico pelo qual um sistema desses fica confiante justamente onde
está mais errado.

A correção padrão é **encolhimento** (*shrinkage*) em direção à média da
população, por exemplo com um prior Beta-Binomial:

```
p̂ = (k + α) / (n + α + β)
```

onde `α, β` vêm da distribuição da liga. Com `n` pequeno, `p̂` fica perto da média
da liga; conforme `n` cresce, converge para a frequência observada.

**O que fizemos:** o dataset não tem estimador de probabilidade embutido; isso
passa a ser trabalho do modelo. No 2º bimestre, a regressão à média é tratada de
duas formas: (a) modelos regularizados, que naturalmente encolhem coeficientes; e
(b) **calibração explícita** das probabilidades previstas, comparando previsto vs.
observado por faixa.

---

## Consequência transversal: nada de linha de mercado

A auditoria também tentou reconstruir o histórico de linhas para validar o
estimador. Não há fonte **gratuita, completa e reproduzível** de linhas
históricas de props:

- APIs comerciais têm o dado, mas o histórico de props fica atrás de plano pago.
- Agregadores não publicam histórico; raspagem viola termos de uso e não é
  reproduzível por quem for corrigir o trabalho.
- Datasets públicos avulsos cobrem poucos meses, sem metodologia declarada e sem
  informar se a linha é de abertura ou de fechamento.

Por isso este projeto adota uma **linha sintética** (mediana móvel pré-jogo,
arredondada para `.5`) e assume o trade-off por escrito. Ver `src/target.py` e a
seção correspondente em `docs/LIMITACOES_E_VIESES.md`.

---

## Resumo: do achado à decisão

| Achado | Decisão neste dataset |
|---|---|
| Amostra pequena (IC ± 22–31 pp) | Modelo treinado em milhares de linhas; avaliação por Brier + calibração |
| Contaminação regular × playoffs | Janela agrupada por temporada; playoffs opt-in e marcados; teste automatizado |
| Decaimento reduz `n_eff` (100 → 19) | Decaimento vira **uma** feature entre médias de 3/5/10; `n_eff` documentado |
| Sem regressão à média | Regularização + calibração explícita no 2º bimestre |
| Sem linha de mercado confiável | Linha sintética documentada + alvo de regressão como principal |
