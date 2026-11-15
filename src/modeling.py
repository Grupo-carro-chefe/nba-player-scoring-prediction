# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/modeling.py
# SINTESE: Dados, splits temporais, famílias de features, baselines e catálogo de modelos.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: splits, famílias, baselines e catálogo de modelos.
# === FIM DO CABECALHO ===

"""Dados, splits, familias de features, baselines e catalogo de modelos.

Este modulo concentra tudo que define O QUE o modelo ve e COMO ele e treinado:

- `load_dataset` le o dataset versionado (nunca recoleta: a API da NBA e dado
  vivo e uma nova coleta poderia mudar as linhas).
- `season_folds` gera a validacao cruzada temporal por temporada (forward
  chaining): treina no passado, valida na temporada seguinte. Nunca aleatoria.
- `FEATURE_FAMILIES` particiona as 24 features pre-jogo em familias, usadas na
  ablacao e na importancia agrupada (16 pares de features tem |corr| > 0,9, o
  que torna enganosa a importancia coluna a coluna).
- `add_clf_features` cria as features relativas a linha sintetica, sem as quais
  um classificador nao tem como saber qual e o limiar a superar.
- `regression_zoo` / `classification_zoo` devolvem Pipelines do scikit-learn: o
  StandardScaler fica DENTRO do pipeline, entao e reajustado em cada dobra
  apenas com dados de treino.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features import POST_GAME_COLUMNS, PRE_GAME_FEATURES

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset.parquet"

TARGET_REG = "y_pts"
TARGET_CLF = "y_over"
LINE = "line_synthetic"

# Particao exata das 24 features pre-jogo (um teste garante que nada sobra nem
# se repete).
FEATURE_FAMILIES: dict[str, list[str]] = {
    "historico_pontos": [
        "pts_ma3_prev", "pts_ma5_prev", "pts_ma10_prev",
        "pts_ewm_prev", "pts_std5_prev", "pts_season_mean_prev",
    ],
    "minutos_e_papel": [
        "min_ma3_prev", "min_ma5_prev", "min_ma10_prev",
        "min_ewm_prev", "min_season_mean_prev", "starter_proxy_prev",
    ],
    "volume_de_jogo": ["fga_ma5_prev", "fg3a_ma5_prev", "fta_ma5_prev", "reb_ma5_prev", "ast_ma5_prev"],
    "descanso_e_mando": ["rest_days", "is_b2b", "is_home"],
    "adversario": ["opp_def_pts_allowed_prev", "opp_pace_proxy_prev", "opp_games_played_prev"],
    "calendario_do_jogador": ["games_played_prev"],
}

# Features extras da classificacao: todas sao funcao de colunas pre-jogo da
# propria linha, entao nao trazem informacao do jogo que esta sendo previsto.
CLF_EXTRA_FEATURES: list[str] = [
    LINE,
    "diff_ewm_linha",
    "diff_ma5_linha",
    "diff_media_temporada_linha",
    "z_ewm_linha",
    "linha_inteira",
]

# Faixas de minutos pela media dos 5 jogos ANTERIORES (os minutos do proprio
# jogo sao pos-jogo e nao podem estratificar uma metrica de uso real).
MINUTE_BANDS = [15, 20, 25, 30, 35, np.inf]
MINUTE_BAND_LABELS = ["15-20", "20-25", "25-30", "30-35", "35+"]

REST_DAYS_CAP = 10  # acima disso o valor reflete ausencia longa, nao descanso


# ----------------------------------------------------------------------------
# Dados e splits
# ----------------------------------------------------------------------------

def load_dataset(path: Path = DATASET) -> pd.DataFrame:
    """Le o dataset analitico e prepara os tipos para o scikit-learn.

    Colunas `Int64` (nullable) viram float64 porque o sklearn nao aceita pd.NA.
    O dataset nao tem nulos nas features; se tiver, algo mudou na construcao e
    o erro deve aparecer aqui, nao no meio de um treino.
    """
    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    for col in PRE_GAME_FEATURES + [LINE, TARGET_REG, TARGET_CLF]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    nulos = int(df[PRE_GAME_FEATURES + [LINE, TARGET_REG, TARGET_CLF]].isna().sum().sum())
    if nulos:
        raise ValueError(f"dataset com {nulos} valores nulos em features/alvos")
    return df.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)


def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in {"train", "val", "test"}:
        raise ValueError(f"split desconhecido: {name}")
    return df[df["split"] == name].reset_index(drop=True)


def assert_disjoint_temporal(df: pd.DataFrame) -> None:
    """Falha se os splits se sobrepoem em jogos ou fora de ordem no tempo."""
    order = ["train", "val", "test"]
    seen: set = set()
    last_end = None
    for name in order:
        part = df[df["split"] == name]
        if part.empty:
            raise AssertionError(f"split {name} vazio")
        games = set(part["GAME_ID"])
        if games & seen:
            raise AssertionError(f"jogos de {name} aparecem em split anterior")
        seen |= games
        if last_end is not None and part["GAME_DATE"].min() <= last_end:
            raise AssertionError(f"{name} comeca antes do fim do split anterior")
        last_end = part["GAME_DATE"].max()
    extras = set(df["split"].unique()) - set(order)
    if extras:
        raise AssertionError(f"linhas fora dos splits: {extras}")


def season_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dobras forward chaining por temporada, dentro do treino.

    Com 3 temporadas no treino: (21-22 -> 22-23) e (21-22 + 22-23 -> 23-24).
    Cada dobra treina so no passado e valida na temporada inteira seguinte, que
    e o cenario real de uso. Um TimeSeriesSplit por numero de linhas cortaria
    temporadas ao meio.
    """
    seasons = sorted(train["SEASON"].unique())
    pos = np.arange(len(train))
    folds = []
    for k in range(1, len(seasons)):
        tr = pos[train["SEASON"].isin(seasons[:k]).to_numpy()]
        va = pos[(train["SEASON"] == seasons[k]).to_numpy()]
        folds.append((tr, va))
    return folds


def minute_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["min_ma5_prev"], bins=MINUTE_BANDS, labels=MINUTE_BAND_LABELS, right=False)


# ----------------------------------------------------------------------------
# Features da classificacao
# ----------------------------------------------------------------------------

def add_clf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as features relativas a linha sintetica.

    A linha e a mediana dos 10 jogos anteriores, logo e pre-jogo. A pergunta da
    classificacao e relativa a ela ("supera o proprio patamar recente?"), entao
    o modelo precisa receber a linha e a distancia entre as medias recentes e a
    linha. Arvores nao constroem diferencas entre colunas sozinhas.

    `linha_inteira` marca as linhas sem o .5: nelas existe empate possivel
    (y_pts == linha), que o rotulo trata como "nao superou". Ela entra para que
    a analise possa medir quanto do sinal vem desse artefato.
    """
    out = df.copy()
    line = out[LINE]
    out["diff_ewm_linha"] = out["pts_ewm_prev"] - line
    out["diff_ma5_linha"] = out["pts_ma5_prev"] - line
    out["diff_media_temporada_linha"] = out["pts_season_mean_prev"] - line
    out["z_ewm_linha"] = (out["pts_ewm_prev"] - line) / (out["pts_std5_prev"] + 1.0)
    out["linha_inteira"] = (line == np.floor(line)).astype("float64")
    return out


def clf_feature_set(variant: str = "completo") -> list[str]:
    """Conjuntos de features da classificacao usados nas ablacoes."""
    if variant == "completo":
        return PRE_GAME_FEATURES + CLF_EXTRA_FEATURES
    if variant == "sem_linha_inteira":
        return PRE_GAME_FEATURES + [c for c in CLF_EXTRA_FEATURES if c != "linha_inteira"]
    if variant == "so_pre_jogo":
        return list(PRE_GAME_FEATURES)
    raise ValueError(variant)


def assert_no_post_game(features: list[str]) -> None:
    """Nenhuma feature pode ser alvo ou coluna pos-jogo (comparacao sem caixa)."""
    proibidas = {c.lower() for c in POST_GAME_COLUMNS} | {TARGET_REG, TARGET_CLF, "min", "pts"}
    vazadas = [f for f in features if f.lower() in proibidas]
    if vazadas:
        raise AssertionError(f"features pos-jogo no modelo: {vazadas}")


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

# Previsores sem treino: a propria media recente do jogador.
REG_BASELINES: dict[str, str] = {
    "media_3_jogos": "pts_ma3_prev",
    "media_5_jogos": "pts_ma5_prev",  # baseline prometida no relatorio N1
    "media_10_jogos": "pts_ma10_prev",
    "media_exponencial": "pts_ewm_prev",
    "media_da_temporada": "pts_season_mean_prev",
    "mediana_10_jogos": LINE,
}


def predict_baselines(df: pd.DataFrame, train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Previsoes de todas as baselines para as linhas de `df`.

    Inclui as constantes do treino (media e mediana) e uma regressao linear de
    uma variavel sobre a media exponencial. Esta ultima e a baseline mais
    exigente: mostra se o ganho de um modelo vem de usar mais informacao ou so
    de reescalar a media recente.
    """
    preds = {nome: df[col].to_numpy(dtype=float) for nome, col in REG_BASELINES.items()}
    preds["constante_media_treino"] = np.full(len(df), train[TARGET_REG].mean())
    preds["constante_mediana_treino"] = np.full(len(df), train[TARGET_REG].median())
    lin = LinearRegression().fit(train[["pts_ewm_prev"]], train[TARGET_REG])
    preds["linear_sobre_exponencial"] = lin.predict(df[["pts_ewm_prev"]])
    return preds


# ----------------------------------------------------------------------------
# Catalogo de modelos
# ----------------------------------------------------------------------------

def cap_rest_days(X: pd.DataFrame) -> pd.DataFrame:
    """Limita `rest_days` (ausencias de 100+ dias distorcem modelos lineares).

    Sem estado: o limite e fixo, nao aprendido, entao nao ha o que vazar.
    """
    if isinstance(X, pd.DataFrame) and "rest_days" in X.columns:
        X = X.copy()
        X["rest_days"] = X["rest_days"].clip(upper=REST_DAYS_CAP)
    return X


def _linear(estimator) -> Pipeline:
    return Pipeline([
        ("cap", FunctionTransformer(cap_rest_days)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _tree(estimator) -> Pipeline:
    # Arvores nao precisam de padronizacao; o pipeline fica igual por simetria.
    return Pipeline([("cap", FunctionTransformer(cap_rest_days)), ("model", estimator)])


def regression_zoo(seed: int = SEED) -> dict[str, tuple[Pipeline, dict]]:
    """Modelos de regressao em ordem crescente de complexidade, com suas grades.

    As grades sao pequenas de proposito: cada combinacao e treinada em 2 dobras
    temporais, e o objetivo e comparar familias, nao espremer o ultimo decimo.
    """
    hgb_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_iter": [200, 400],
        "model__max_leaf_nodes": [15, 31],
        "model__min_samples_leaf": [50, 200],
        "model__l2_regularization": [0.0, 1.0],
    }

    def hgb(loss: str) -> Pipeline:
        # early_stopping=False: o early stopping do HGB separa uma validacao
        # ALEATORIA interna, o que quebraria a disciplina temporal.
        return _tree(HistGradientBoostingRegressor(loss=loss, early_stopping=False, random_state=seed))

    return {
        "ridge": (_linear(Ridge()), {"model__alpha": list(np.logspace(-2, 3, 11))}),
        "lasso": (_linear(Lasso(max_iter=10000)), {"model__alpha": list(np.logspace(-4, 0, 9))}),
        "poisson_glm": (
            _linear(PoissonRegressor(max_iter=1000)),
            {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
        ),
        "random_forest": (
            _tree(RandomForestRegressor(
                n_estimators=200, max_samples=0.5, n_jobs=-1, random_state=seed,
            )),
            {"model__max_features": [0.33, 0.6], "model__min_samples_leaf": [20, 50]},
        ),
        "hgb_quadratico": (hgb("squared_error"), hgb_grid),
        "hgb_poisson": (hgb("poisson"), hgb_grid),
        "hgb_absoluto": (hgb("absolute_error"), hgb_grid),
    }


def classification_zoo(seed: int = SEED) -> dict[str, tuple[Pipeline, dict]]:
    """Classificadores equivalentes aos algoritmos da regressao."""
    return {
        "logistica_l2": (
            _linear(LogisticRegression(l1_ratio=0.0, max_iter=2000)),
            {"model__C": list(np.logspace(-3, 1, 9))},
        ),
        "logistica_l1": (
            _linear(LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=3000, random_state=seed)),
            {"model__C": list(np.logspace(-3, 1, 9))},
        ),
        "random_forest": (
            _tree(RandomForestClassifier(
                n_estimators=200, max_samples=0.5, n_jobs=-1, random_state=seed,
            )),
            {"model__max_features": [0.33], "model__min_samples_leaf": [50, 200]},
        ),
        "hgb": (
            _tree(HistGradientBoostingClassifier(early_stopping=False, random_state=seed)),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_iter": [200],
                "model__max_leaf_nodes": [8, 15],
                "model__min_samples_leaf": [200, 500],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    }
