# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_collect_io.py
# SINTESE: Coleta (API simulada), cache, nova tentativa e leitura/escrita de tabelas.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: coleta com API simulada e io_utils.
# === FIM DO CABECALHO ===

"""Coleta e leitura/escrita de tabelas, sem rede.

A API da NBA e substituida por um duble: o que se testa aqui e a logica do
projeto em volta dela (cache, nova tentativa com espera crescente, normalizacao
do MATCHUP, manifesto), nao a disponibilidade de stats.nba.com.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from src import collect, io_utils
from src.io_utils import parquet_available, read_table, table_exists, write_table


def _tabela() -> pd.DataFrame:
    return pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
        "PLAYER_ID": [1, 2, 3],
        "PTS": [10.5, 20.0, 0.0],
        "TIME": ["AAA", "BBB", "CCC"],
    })


# ----------------------------------------------------------------------------
# io_utils
# ----------------------------------------------------------------------------

def test_escrita_e_leitura_sem_perda(tmp_path):
    arquivos = write_table(_tabela(), tmp_path / "sub" / "t")  # cria a pasta
    assert [p.name for p in arquivos][0] == "t.csv"
    lido = read_table(tmp_path / "sub" / "t")
    pd.testing.assert_frame_equal(lido, _tabela(), check_dtype=False)
    assert pd.api.types.is_datetime64_any_dtype(lido["GAME_DATE"])


def test_grava_parquet_quando_ha_engine(tmp_path):
    if not parquet_available():
        pytest.skip("sem engine de parquet")
    arquivos = write_table(_tabela(), tmp_path / "t")
    assert sorted(p.suffix for p in arquivos) == [".csv", ".parquet"]


def test_sem_pyarrow_grava_so_csv_e_avisa(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(io_utils, "parquet_available", lambda: False)
    arquivos = write_table(_tabela(), tmp_path / "t")
    assert [p.suffix for p in arquivos] == [".csv"]
    assert not (tmp_path / "t.parquet").exists()
    assert "sem pyarrow" in capsys.readouterr().out


def test_sem_engine_le_o_csv_mesmo_com_parquet_presente(tmp_path, monkeypatch):
    write_table(_tabela(), tmp_path / "t")
    monkeypatch.setattr(io_utils, "parquet_available", lambda: False)
    lido = read_table(tmp_path / "t")
    assert len(lido) == 3
    assert pd.api.types.is_datetime64_any_dtype(lido["GAME_DATE"])  # a data e reinterpretada no csv


def test_leitura_de_tabela_inexistente_diz_o_que_faltou(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"t\.parquet nem t\.csv"):
        read_table(tmp_path / "t")


def test_table_exists(tmp_path):
    assert not table_exists(tmp_path / "t")
    (tmp_path / "t.csv").write_text("a\n1\n", encoding="utf-8")
    assert table_exists(tmp_path / "t")


def test_data_invalida_vira_nat(tmp_path):
    (tmp_path / "t.csv").write_text("GAME_DATE,x\n2024-01-01,1\nnao-e-data,2\n", encoding="utf-8")
    lido = read_table(tmp_path / "t")
    assert pd.isna(lido.loc[1, "GAME_DATE"]) and lido.loc[0, "GAME_DATE"] == pd.Timestamp("2024-01-01")


def test_parquet_available_devolve_booleano():
    assert isinstance(parquet_available(), bool)


# ----------------------------------------------------------------------------
# normalizacao
# ----------------------------------------------------------------------------

def test_normalize_extrai_mando_e_adversario_do_matchup():
    bruto = pd.DataFrame({
        "game_date": ["2024-01-01", "2024-01-03", "2024-01-05"],
        "matchup": ["LAL vs. PHX", "LAL @ PHX", "NYK vs. BKN"],
        "MIN": ["30", "abc", "12.5"],
        "PTS": ["10", "11", "12"],
    })
    out = collect._normalize(bruto, "2023-24", "Regular Season")
    assert out["IS_HOME"].tolist() == [True, False, True]
    assert out["OPPONENT"].tolist() == ["PHX", "PHX", "BKN"]
    assert out["SEASON"].eq("2023-24").all() and out["SEASON_TYPE"].eq("Regular Season").all()
    assert not out["IS_PLAYOFF"].any()
    assert pd.api.types.is_datetime64_any_dtype(out["GAME_DATE"])
    assert out["PTS"].tolist() == [10, 11, 12]
    assert out["MIN"].iloc[[0, 2]].tolist() == [30.0, 12.5] and pd.isna(out["MIN"].iloc[1])  # texto vira NaN


def test_normalize_marca_playoffs():
    bruto = pd.DataFrame({"GAME_DATE": ["2024-05-01"], "MATCHUP": ["BOS vs. MIA"], "PTS": [1]})
    assert collect._normalize(bruto, "2023-24", "Playoffs")["IS_PLAYOFF"].all()


def test_normalize_nao_altera_a_entrada():
    bruto = pd.DataFrame({"game_date": ["2024-01-01"], "matchup": ["LAL @ PHX"], "PTS": [1]})
    antes = bruto.copy()
    collect._normalize(bruto, "2023-24", "Regular Season")
    pd.testing.assert_frame_equal(bruto, antes)


def test_nome_do_cache_nao_tem_espaco():
    nome = collect._cache_path("player", "2023-24", "Regular Season").name
    assert nome == "player_2023-24_regular-season"


# ----------------------------------------------------------------------------
# busca na API (duble), cache e nova tentativa
# ----------------------------------------------------------------------------

def _api_falsa(monkeypatch, falhas: int, df: pd.DataFrame | None = None) -> dict:
    pytest.importorskip("nba_api")
    from nba_api.stats.endpoints import leaguegamelog

    estado = {"chamadas": 0, "kwargs": []}
    resposta = df if df is not None else pd.DataFrame({"PLAYER_ID": [1], "GAME_DATE": ["2024-01-01"], "MATCHUP": ["A vs. B"]})

    class Falsa:
        def __init__(self, **kwargs):
            estado["chamadas"] += 1
            estado["kwargs"].append(kwargs)
            if estado["chamadas"] <= falhas:
                raise ConnectionError("rede fora")

        def get_data_frames(self):
            return [resposta]

    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", Falsa)
    return estado


@pytest.fixture()
def sem_espera(monkeypatch, tmp_path):
    esperas: list[float] = []
    monkeypatch.setattr(collect.time, "sleep", esperas.append)
    monkeypatch.setattr(collect, "CACHE_DIR", tmp_path / "cache")
    return esperas


def _relatorio() -> collect.CollectionReport:
    return collect.CollectionReport(seasons=["2023-24"], season_types=["Regular Season"])


def test_cache_evita_a_chamada_a_api(sem_espera, monkeypatch):
    write_table(_tabela(), collect._cache_path("player", "2023-24", "Regular Season"))
    estado = _api_falsa(monkeypatch, falhas=0)
    rep = _relatorio()
    df = collect._fetch_league_game_log("2023-24", "Regular Season", "P", rep)
    assert len(df) == 3
    assert rep.cache_hits == 1 and rep.requests_made == 0 and estado["chamadas"] == 0
    assert sem_espera == []


def test_nova_tentativa_com_espera_crescente_e_pausa_apos_o_sucesso(sem_espera, monkeypatch):
    estado = _api_falsa(monkeypatch, falhas=2)
    rep = _relatorio()
    df = collect._fetch_league_game_log("2023-24", "Regular Season", "T", rep)
    assert len(df) == 1 and estado["chamadas"] == 3
    assert sem_espera == [2.0, 4.0, collect.SLEEP_BETWEEN]
    assert rep.requests_made == 1 and rep.endpoints == ["leaguegamelog.LeagueGameLog(player_or_team_abbreviation='T')"]
    ultima = estado["kwargs"][-1]
    assert ultima["season"] == "2023-24" and ultima["season_type_all_star"] == "Regular Season"
    assert ultima["player_or_team_abbreviation"] == "T" and ultima["timeout"] == collect.TIMEOUT
    # o resultado foi para o cache: a segunda chamada nao toca a API
    collect._fetch_league_game_log("2023-24", "Regular Season", "T", rep)
    assert estado["chamadas"] == 3 and rep.cache_hits == 1


def test_desiste_depois_do_maximo_de_tentativas(sem_espera, monkeypatch):
    estado = _api_falsa(monkeypatch, falhas=10**6)
    with pytest.raises(RuntimeError, match="falha ao coletar player 2023-24 Regular Season"):
        collect._fetch_league_game_log("2023-24", "Regular Season", "P", _relatorio())
    assert estado["chamadas"] == collect.MAX_RETRIES
    assert sem_espera == [collect.BACKOFF_BASE**k for k in range(1, collect.MAX_RETRIES + 1)]


# ----------------------------------------------------------------------------
# collect() de ponta a ponta com a busca substituida
# ----------------------------------------------------------------------------

def _busca_falsa(season, season_type, player_or_team, report):
    report.requests_made += 1
    if player_or_team == "P":
        return pd.DataFrame({
            "PLAYER_ID": [2, 1], "GAME_ID": ["g2", "g1"], "GAME_DATE": ["2024-01-03", "2024-01-01"],
            "MATCHUP": ["AAA vs. BBB", "AAA @ BBB"], "MIN": [30.0, 28.0], "PTS": [11, 22],
        })
    return pd.DataFrame({
        "TEAM_ID": [10, 10], "GAME_ID": ["g1", "g2"], "GAME_DATE": ["2024-01-01", "2024-01-03"],
        "MATCHUP": ["AAA @ BBB", "AAA vs. BBB"], "PTS": [100, 110],
    })


def test_collect_grava_tabelas_e_manifesto(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_fetch_league_game_log", _busca_falsa)
    saida = collect.collect(seasons=["2023-24"], out_dir=tmp_path)
    assert set(saida) == {"players", "teams"}
    assert saida["players"]["PLAYER_ID"].tolist() == [1, 2]  # ordenado por jogador e data
    assert saida["players"]["IS_HOME"].tolist() == [False, True]
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["temporadas"] == ["2023-24"] and m["season_types"] == ["Regular Season"]
    assert m["linhas"] == {"player_game_logs": 2, "team_game_logs": 2}
    assert m["requests_realizados"] == 2 and m["cache_hits"] == 0
    assert m["filtros"]["playoffs_incluidos"] is False
    assert m["fonte"].startswith("stats.nba.com")
    assert {"player_game_logs.csv", "team_game_logs.csv"} <= set(m["arquivos"])
    assert read_table(tmp_path / "player_game_logs")["SEASON"].eq("2023-24").all()


def test_collect_com_playoffs_dobra_as_buscas(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_fetch_league_game_log", _busca_falsa)
    saida = collect.collect(seasons=["2023-24"], include_playoffs=True, out_dir=tmp_path)
    assert len(saida["players"]) == 4
    assert saida["players"]["IS_PLAYOFF"].sum() == 2
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["season_types"] == ["Regular Season", "Playoffs"] and m["filtros"]["playoffs_incluidos"] is True


def test_temporadas_padrao_excluem_as_atipicas():
    assert collect.DEFAULT_SEASONS == ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    assert not {"2019-20", "2020-21"} & set(collect.DEFAULT_SEASONS)


def test_linha_de_comando_da_coleta(tmp_path, monkeypatch):
    capturado = {}
    monkeypatch.setattr(collect, "collect", lambda **kw: capturado.update(kw))
    monkeypatch.setattr(sys, "argv", ["collect", "--seasons", "2023-24", "2024-25", "--include-playoffs", "--out", str(tmp_path)])
    collect.main()
    assert capturado["seasons"] == ["2023-24", "2024-25"]
    assert capturado["include_playoffs"] is True and str(capturado["out_dir"]) == str(tmp_path)
