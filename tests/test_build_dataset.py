# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_build_dataset.py
# SINTESE: Pipeline de construção do dataset de ponta a ponta em dados sintéticos.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: pipeline do dataset em dados sintéticos.
# === FIM DO CABECALHO ===

"""Pipeline de construcao do dataset, de ponta a ponta, em dados sinteticos.

O arquivo entregue (data/processed) so pode ser conferido de fora. Aqui o
pipeline roda do zero numa pasta temporaria, com dados cuja saida foi calculada
a mao, e prova que filtros, aritmetica, split e arquivos gerados se comportam
como o README e o relatorio afirmam.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import build_dataset
from src.build_dataset import ID_COLUMNS, SPLIT_BY_SEASON, build
from src.features import POST_GAME_COLUMNS, PRE_GAME_FEATURES
from src.io_utils import read_table, write_table
from src.modeling import assert_disjoint_temporal

N_JOGOS = 14  # por jogador e temporada: 4 deles (indices 10 a 13) tem 10 jogos de historico
INICIO = {"2020-21": "2020-12-25", "2021-22": "2021-10-20", "2024-25": "2024-10-25", "2025-26": "2025-10-24"}


def _raw(tmp_path, com_times: bool = True):
    """3 jogadores por temporada: dois de rotacao (A e B) e um reserva (C, 8 min)."""
    jogadores, times = [], []
    for si, (temporada, inicio) in enumerate(INICIO.items()):
        datas = pd.date_range(inicio, periods=N_JOGOS, freq="2D")
        for i, d in enumerate(datas):
            gid = f"00{si}{i:05d}"
            for pid, nome, time, adv, casa, minutos, pts in (
                (1, "Ala Um", "AAA", "BBB", True, 32.0, 20 + i % 5),
                (2, "Pivo Dois", "BBB", "AAA", False, 30.0, 15 + i % 4),
                (3, "Reserva Tres", "AAA", "BBB", True, 40.0 if i == N_JOGOS - 1 else 8.0, 2 + i % 3),
            ):
                jogadores.append({
                    "PLAYER_ID": pid, "PLAYER_NAME": nome, "TEAM_ABBREVIATION": time, "OPPONENT": adv,
                    "GAME_ID": gid, "GAME_DATE": d, "SEASON": temporada, "SEASON_TYPE": "Regular Season",
                    "IS_PLAYOFF": False, "IS_HOME": casa, "PTS": pts, "MIN": minutos,
                    "FGA": 10.0, "FG3A": 3.0, "FTA": 4.0, "REB": 5.0, "AST": 4.0,
                })
            times.append({"GAME_ID": gid, "GAME_DATE": d, "SEASON": temporada, "TEAM_ID": 1, "TEAM_ABBREVIATION": "AAA", "PTS": 100 + i})
            times.append({"GAME_ID": gid, "GAME_DATE": d, "SEASON": temporada, "TEAM_ID": 2, "TEAM_ABBREVIATION": "BBB", "PTS": 95 + i})
    raw = tmp_path / "raw"
    write_table(pd.DataFrame(jogadores), raw / "player_game_logs")
    if com_times:
        write_table(pd.DataFrame(times), raw / "team_game_logs")
    return raw, len(jogadores)


@pytest.fixture()
def construido(tmp_path):
    raw, n_brutas = _raw(tmp_path)
    out_dir = tmp_path / "processed"
    df = build(raw, out_dir)
    return df, out_dir, n_brutas


def test_aritmetica_dos_filtros_calculada_a_mao(construido):
    df, out_dir, n_brutas = construido
    m = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    f = m["filtros"]
    assert n_brutas == 168  # 3 jogadores x 14 jogos x 4 temporadas
    assert f["linhas_brutas"] == 168
    # reserva (56 linhas) + primeiro jogo dos dois titulares em 4 temporadas (8) = 64
    assert f["descartadas_por_minutos"] == 64
    # jogos 1 a 9 dos titulares, sem 10 jogos de historico, em 4 temporadas: 9 x 2 x 4 = 72
    assert f["descartadas_sem_alvo"] == 72
    assert f["linhas_brutas"] - f["descartadas_por_minutos"] - f["descartadas_sem_alvo"] == len(df) == 32
    assert f["conferencia"] == "168 - 64 - 72 = 32"


def test_corte_de_minutos_usa_a_media_anterior_e_nao_o_jogo_atual(construido):
    df, _, _ = construido
    # no ultimo jogo o reserva jogou 40 minutos, mas ate ali sua media era 8
    assert 3 not in set(df["PLAYER_ID"])
    assert (df["min_ma5_prev"] >= build_dataset.MIN_MINUTES_PREV).all()


def test_parametro_de_minutos_e_respeitado(tmp_path):
    raw, _ = _raw(tmp_path)
    df = build(raw, tmp_path / "out", min_minutes_prev=0.0)
    assert 3 in set(df["PLAYER_ID"])
    assert len(df) == 48  # 3 jogadores x 4 jogos x 4 temporadas


def test_so_ha_linhas_com_historico_suficiente(construido):
    df, _, _ = construido
    assert (df["games_played_prev"] >= 10).all()
    assert df.groupby(["PLAYER_ID", "SEASON"]).size().eq(4).all()
    assert df["y_over"].notna().all() and df["line_synthetic"].notna().all()


def test_split_pelas_temporadas_conhecidas_e_unassigned_para_as_demais(construido):
    df, out_dir, _ = construido
    por_temporada = df.groupby("SEASON")["split"].first().to_dict()
    assert por_temporada == {"2020-21": "unassigned", "2021-22": "train", "2024-25": "val", "2025-26": "test"}
    m = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert m["distribuicao_split"] == {"train": 8, "val": 8, "test": 8, "unassigned": 8}
    assert m["split_temporal"] == SPLIT_BY_SEASON
    # e a trava de modelagem recusa um dataset com temporada fora dos splits
    with pytest.raises(AssertionError, match="fora dos splits"):
        assert_disjoint_temporal(df)


def test_colunas_de_saida_sao_identificacao_features_e_alvos(construido):
    df, _, _ = construido
    esperadas = ID_COLUMNS + PRE_GAME_FEATURES + ["line_synthetic", "y_pts", "y_over", "split"]
    assert list(df.columns) == esperadas
    assert not set(df.columns) & set(POST_GAME_COLUMNS), "coluna pos-jogo no dataset"


def test_features_do_adversario_vem_preenchidas(construido):
    df, _, _ = construido
    for col in ("opp_def_pts_allowed_prev", "opp_pace_proxy_prev", "opp_games_played_prev"):
        assert df[col].notna().all(), col


def test_saida_ordenada_por_data_e_jogador(construido):
    df, _, _ = construido
    chaves = list(zip(df["GAME_DATE"], df["PLAYER_ID"]))
    assert chaves == sorted(chaves)
    assert df.index.tolist() == list(range(len(df)))


def test_arquivos_gravados_e_relidos_sem_perda(construido):
    df, out_dir, _ = construido
    for nome in ("dataset.csv", "dataset.parquet", "dataset_manifest.json"):
        assert (out_dir / nome).exists(), nome
    lido = read_table(out_dir / "dataset")
    assert len(lido) == len(df)
    assert lido["GAME_DATE"].equals(df["GAME_DATE"])
    assert np.allclose(lido["y_pts"], df["y_pts"])


def test_manifesto_registra_o_que_foi_feito(construido):
    df, out_dir, _ = construido
    m = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert m["linhas"] == len(df) and m["colunas"] == df.shape[1]
    assert m["features_pre_jogo"] == PRE_GAME_FEATURES
    assert m["colunas_pos_jogo_removidas"] == POST_GAME_COLUMNS
    assert m["alvos"] == {"regressao": "y_pts", "classificacao": "y_over", "linha": "line_synthetic"}
    assert m["alvo"]["linhas_com_alvo"] == len(df)
    assert "\r" not in (out_dir / "dataset_manifest.json").read_text(encoding="utf-8")


def test_sem_log_de_times_o_dataset_sai_sem_features_do_adversario(tmp_path):
    raw, _ = _raw(tmp_path, com_times=False)
    df = build(raw, tmp_path / "out")
    assert "opp_def_pts_allowed_prev" not in df.columns
    assert len(df) == 32


def test_dados_brutos_ausentes_dao_mensagem_clara(tmp_path):
    with pytest.raises(SystemExit, match="python -m src.collect"):
        build(tmp_path / "nao_existe", tmp_path / "out")


def test_linha_de_comando_encaminha_os_argumentos(tmp_path, monkeypatch):
    raw, _ = _raw(tmp_path)
    out = tmp_path / "saida"
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--raw", str(raw), "--out", str(out), "--min-minutes-prev", "0"])
    build_dataset.main()
    m = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert m["filtros"]["min_minutes_prev"] == 0.0
    assert m["linhas"] == 48
