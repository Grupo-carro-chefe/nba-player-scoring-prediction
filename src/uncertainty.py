# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: src/uncertainty.py
# SINTESE: Intervalos conformais por estimativa e oráculos do teto preditivo.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-15 | Rodrigo    | Versão inicial: conformal normalizado e oráculos do teto.
# === FIM DO CABECALHO ===

"""Incerteza por estimativa (conformal) e estimativa do teto preditivo.

Intervalo conformal dividido, normalizado
-----------------------------------------
O relatorio N1 se compromete a acompanhar cada estimativa de uma medida de
dispersao. Usamos predicao conformal dividida (split conformal): o modelo e
treinado no treino, e a validacao (2024-25) serve para medir o tamanho tipico
do erro. O escore de nao conformidade e normalizado pela raiz da previsao,
|y - mu| / sqrt(mu), porque a variancia de uma contagem cresce com a media:
quem marca 25 pontos oscila mais que quem marca 6. O intervalo resultante e
mu +- q * sqrt(mu), com q o quantil conformal do escore.

Sob permutabilidade entre validacao e teste a cobertura nominal (80%, 90%) e
garantida; como ha deriva entre temporadas, a cobertura no teste e MEDIDA e
reportada, nao presumida.

Teto preditivo (oraculos)
-------------------------
Para estimar o quanto do erro e irredutivel com informacao pre-jogo, comparamos
o modelo com dois previsores que trapaceiam de proposito, usando o futuro:

- oraculo_temporada: media do jogador na temporada INTEIRA, sem o proprio jogo
  (leave-one-out). Sabe o nivel real do atleta naquele ano.
- oraculo_local: media dos 5 jogos anteriores e dos 5 posteriores, sem o
  proprio. Sabe o nivel do atleta naquele trecho da temporada.

Cada um existe em versao media e mediana (`oracle_median`). Como o MAE e
minimizado pela mediana e os pontos sao assimetricos, a versao em mediana e a
regua justa para o MAE.

Nenhum deles e utilizavel na pratica; eles servem de regua. A distancia entre o
modelo e o oraculo e o que ainda poderia ser ganho conhecendo o nivel do
jogador; o erro do proprio oraculo e, em boa parte, variancia de jogo a jogo
(minutos, aproveitamento, contusoes) que nenhuma informacao de nivel elimina.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MIN_SCALE = 1.0  # evita raiz de previsoes ~0 explodir o escore


def _scale(mu) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(mu, dtype=float), MIN_SCALE))


@dataclass
class ConformalBand:
    """Intervalo mu +- q * sqrt(mu), com q calibrado na validacao."""

    alpha: float
    q: float
    n_calibracao: int

    @classmethod
    def fit(cls, y_cal, mu_cal, alpha: float) -> "ConformalBand":
        y_cal = np.asarray(y_cal, dtype=float)
        scores = np.abs(y_cal - np.asarray(mu_cal, dtype=float)) / _scale(mu_cal)
        n = len(scores)
        # quantil com correcao de amostra finita: ceil((n+1)(1-alpha))/n
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(scores, level, method="higher"))
        return cls(alpha=alpha, q=q, n_calibracao=n)

    def predict(self, mu) -> tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(mu, dtype=float)
        half = self.q * _scale(mu)
        return np.maximum(mu - half, 0.0), mu + half

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalBand":
        return cls(**d)


def coverage_report(y, lo, hi) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    dentro = (y >= lo) & (y <= hi)
    return {
        "n": int(len(y)),
        "cobertura": float(dentro.mean()),
        "largura_media": float(np.mean(hi - lo)),
        "largura_mediana": float(np.median(hi - lo)),
    }


def coverage_by_group(y, lo, hi, groups: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": np.asarray(y, float), "lo": lo, "hi": hi, "g": groups.to_numpy()})
    rows = []
    for g, part in tmp.groupby("g", observed=True, sort=True):
        r = coverage_report(part["y"], part["lo"], part["hi"])
        r["grupo"] = str(g)
        rows.append(r)
    return pd.DataFrame(rows).set_index("grupo")


# ----------------------------------------------------------------------------
# Oraculos (teto preditivo) - usam o futuro de proposito, so como regua
# ----------------------------------------------------------------------------

def oracle_season_loo(df: pd.DataFrame, y_col: str = "y_pts") -> np.ndarray:
    """Media do jogador na temporada, sem o proprio jogo (usa jogos futuros)."""
    g = df.groupby(["PLAYER_ID", "SEASON"])[y_col]
    soma = g.transform("sum")
    n = g.transform("count")
    loo = (soma - df[y_col]) / (n - 1)
    # jogador com um unico jogo no recorte: sem vizinhos, cai na propria media
    return loo.where(n > 1, df[y_col]).to_numpy(dtype=float)


def oracle_local(df: pd.DataFrame, half_window: int = 5, y_col: str = "y_pts") -> np.ndarray:
    """Media dos `half_window` jogos antes e depois, sem o proprio jogo."""
    d = df[["PLAYER_ID", "SEASON", "GAME_DATE", y_col]].sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"])
    w = 2 * half_window + 1
    g = d.groupby(["PLAYER_ID", "SEASON"], sort=False)[y_col]
    soma = g.transform(lambda s: s.rolling(w, center=True, min_periods=1).sum())
    cont = g.transform(lambda s: s.rolling(w, center=True, min_periods=1).count())
    viz = (soma - d[y_col]) / (cont - 1)
    viz = viz.where(cont > 1, d[y_col])
    return viz.reindex(df.index).to_numpy(dtype=float)  # volta a ordem original de df


def _loo_median_group(v: np.ndarray, half_window: int | None) -> np.ndarray:
    """Mediana dos demais jogos do grupo (todos, ou so a vizinhanca)."""
    out = np.empty(len(v), dtype=float)
    for i in range(len(v)):
        lo, hi = (0, len(v)) if half_window is None else (max(0, i - half_window), min(len(v), i + half_window + 1))
        viz = np.concatenate([v[lo:i], v[i + 1:hi]])
        out[i] = np.median(viz) if len(viz) else v[i]
    return out


def oracle_median(df: pd.DataFrame, half_window: int | None = None, y_col: str = "y_pts") -> np.ndarray:
    """Versao em MEDIANA dos oraculos (temporada inteira ou vizinhanca).

    O MAE e minimizado pela mediana, nao pela media; como a distribuicao de
    pontos e assimetrica a direita, um oraculo em media fica artificialmente
    pior em MAE. Para servir de regua de MAE, o oraculo precisa prever a mediana.
    """
    d = df[["PLAYER_ID", "SEASON", "GAME_DATE", y_col]].sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"])
    res = pd.Series(np.nan, index=d.index)
    for _, part in d.groupby(["PLAYER_ID", "SEASON"], sort=False):
        res.loc[part.index] = _loo_median_group(part[y_col].to_numpy(dtype=float), half_window)
    return res.reindex(df.index).to_numpy(dtype=float)
