# Roteiro do vídeo (máximo 5 minutos)

Alvo: **4:40**, com folga para não passar de 5:00. Os dois integrantes falam:
a proposta de divisão está em cada bloco (R = Rodrigo, A = André).

Antes de gravar, deixe abertos:

1. este roteiro;
2. o PDF `relatorio/relatorio_N2.pdf`, nas figuras 1, 4 e 5;
3. um terminal na raiz do repositório, com o venv ativo;
4. o repositório no navegador: https://github.com/Grupo-carro-chefe/nba-player-scoring-prediction.

---

## 0:00 a 0:20 · Abertura (R)

**Tela:** primeira página do relatório (título e logo da FCI).

> "Olá. Este é o projeto *Predição de Desempenho Individual de Atletas da NBA
> por Meio de Aprendizado de Máquina Supervisionado*, de André Ihsan Ward e
> Rodrigo Oliveira, para a disciplina de Inteligência Artificial do professor
> Leandro Zerbinatti, curso de Sistemas de Informação da Universidade
> Presbiteriana Mackenzie."

## 0:20 a 0:55 · Problema e ética (A)

**Tela:** seção 2 do relatório.

> "O problema é estimar quantos pontos um jogador vai marcar num jogo que ainda
> não aconteceu, usando só o que se sabe antes da partida. Na prática, isso se
> faz com a média dos últimos jogos. Queríamos saber se um modelo supervisionado
> faz melhor, e quanto da pontuação é de fato previsível.
> Um cuidado ético guiou o projeto: isto não é ferramenta de apostas. Não há
> odds nem cálculo de valor esperado, e toda estimativa sai com a sua incerteza."

## 0:55 a 1:35 · Dados e preparação (A)

**Tela:** `data/README.md` no GitHub e a Tabela 1 do relatório.

> "Coletamos os dados nós mesmos, nas interfaces públicas da NBA: 83.799
> registros de jogador por partida, em cinco temporadas. Toda variável usa só
> jogos anteriores: médias móveis, média exponencial, minutos, descanso, mando e
> força do adversário. Testes automatizados alteram um jogo futuro e garantem
> que nada do passado muda, ou seja, não há vazamento de dados. A divisão é
> temporal: treino até 2023-24, validação 2024-25 e teste 2025-26."

## 1:35 a 2:35 · Fundamentos de IA usados (R)

**Tela:** Seção 5 do relatório (Tabela 2) ou `src/modeling.py`.

> "Usamos aprendizado supervisionado com scikit-learn. Começamos por baselines,
> as médias móveis, e comparamos regressão linear regularizada (Ridge e Lasso,
> por causa da multicolinearidade entre as médias), um modelo linear
> generalizado de Poisson, floresta aleatória e gradient boosting, que são
> comitês de árvores.
> Os hiperparâmetros foram escolhidos por validação cruzada temporal: o modelo
> sempre treina no passado e é validado na temporada seguinte. Isso evita
> overfitting e vazamento.
> O teste de 2025-26 foi usado uma única vez, no fim.
> Para a incerteza, usamos predição conformal, que dá um intervalo em volta de
> cada previsão. Na classificação, avaliamos se as probabilidades são
> calibradas."

## 2:35 a 3:35 · Demonstração do artefato (A)

**Tela:** terminal. Comandos já testados:

```bash
python -m src.predict --player "Jalen Brunson" --date 2026-01-11
python -m src.predict --sample 3
python -m src.predict --player "ninguem" --date 2026-01-11
```

> "Este é o artefato. Dado um jogador e um jogo, ele mostra os pontos previstos,
> os intervalos de 80% e 90%, a média dos últimos 5 jogos para comparação e a
> probabilidade de o jogador superar a mediana dos seus últimos dez jogos. Como
> são jogos da temporada de teste, mostramos também o que aconteceu: o Brunson
> fez 26 pontos para uma previsão de 26,8, dentro do intervalo. E quando a
> consulta está fora do que o modelo sabe prever, ele recusa em vez de inventar
> um número."

Alternativa: abrir `notebooks/05_demo_artefato.ipynb` e mostrar o gráfico de
previsão com faixa ao longo da temporada.

## 3:35 a 4:20 · Resultados (R)

**Tela:** Figura 1 (escada de MAE), depois Figuras 4 e 5.

> "No teste, o modelo final, gradient boosting com perda absoluta, errou em
> média 4,92 pontos por jogo, contra 5,23 da média de 5 jogos e 4,98 da média
> exponencial. A melhoria é modesta, mas estatisticamente consistente.
> Os intervalos de 80% e 90% cobriram 80,2% e 90,2% dos jogos, como prometido.
> As probabilidades são calibradas, com Brier abaixo do estimador ingênuo.
> E o achado mais interessante: mesmo dando ao modelo o nível real do jogador
> na temporada, com informação do futuro, o erro só cairia mais 0,07 ponto. A
> maior parte do erro que sobra é variação de jogo a jogo: minutos, acerto de
> arremessos, contusões."

## 4:20 a 4:40 · Conclusão e GitHub (A)

**Tela:** página inicial do repositório; role pelas pastas `src`, `notebooks`,
`results` e `relatorio`.

> "Três dos cinco resultados esperados foram alcançados e dois, parcialmente: o
> contexto do jogo ajuda pouco, e o teto preditivo foi estimado só para a
> informação sobre o nível do jogador. Tudo está neste repositório público: dados,
> código com testes, notebooks, resultados e o relatório. Obrigado."

---

## Publicação no YouTube

1. Exportar o vídeo em 1080p e conferir a duração (**menos de 5:00**).
2. No YouTube Studio, enviar com o título *"Predição de Desempenho de Atletas
   da NBA com Aprendizado de Máquina | IA Mackenzie 2026"*.
3. Visibilidade: **Público** (não "não listado").
4. Abrir o link numa janela anônima para confirmar que o acesso é público.

## Inserir o link e gerar o PDF final

```bash
python -m src.report --video-url https://youtu.be/SEU_LINK --pdf
```

Depois, coloque o link em `GRUPO.md`, no lugar de *"será adicionado após a
publicação"*. Rode `pytest -q`, faça o commit e o push, e envie o
`relatorio/relatorio_N2.pdf` no Ambiente Virtual.
