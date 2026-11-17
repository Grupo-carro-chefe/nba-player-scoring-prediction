# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_features_e_alvos.py
# SINTESE: Features, linha sintética e alvos com valores calculados à mão.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: features e alvos com valores conhecidos.
# === FIM DO CABECALHO ===

"""Valores conhecidos para features, linha sintetica e alvos.

test_no_leakage.py garante que o futuro nao muda o passado. Estes testes
garantem que as contas, alem de nao vazarem, estao CERTAS: cada numero esperado
aqui foi calculado a mao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import DECAY, STARTER_MIN_THRESHOLD, build_features, build_opponent_strength
from src.target import LINE_WINDOW, add_targets, make_synthetic_line, round_to_half, target_report
from tests.test_no_leakage import _fake_players


# ----------------------------------------------------------------------------
# features do jogador
# ----------------------------------------------------------------------------

def test_primeira_linha_da_temporada_nao_tem_historico():
    df = build_features(_fake_players(n=6))
    assert df.loc[0, "games_played_prev"] == 0
    for col in ("pts_ma3_prev", "pts_ma5_prev", "pts_ma10_prev", "pts_ewm_prev", "pts_season_mean_prev", "min_ma5_prev"):
        assert pd.isna(df.loc[0, col]), col
    assert pd.isna(df.loc[0, "rest_days"]) and pd.isna(df.loc[0, "is_b2b"]) and pd.isna(df.loc[0, "starter_proxy_prev"])


def test_medias_moveis_de_valor_conhecido():
    df = build_features(_fake_players(n=15, start_pts=10))  # PTS = 10, 11, 12, ...
    # linha 6: os 5 jogos anteriores sao 11..15
    assert df.loc[6, "pts_ma5_prev"] == pytest.approx(13.0)
    # linha 6: os 3 anteriores sao 13, 14, 15
    assert df.loc[6, "pts_ma3_prev"] == pytest.approx(14.0)
    # linha 6: os 10 anteriores nao existem; usa os 6 que ha (10..15 => 12,5)
    assert df.loc[6, "pts_ma10_prev"] == pytest.approx(12.5)
    # com um unico jogo anterior a janela usa esse jogo (min_periods=1)
    assert df.loc[1, "pts_ma5_prev"] == pytest.approx(10.0)


def test_media_da_temporada_e_acumulada():
    df = build_features(_fake_players(n=8, start_pts=10))
    assert df.loc[4, "pts_season_mean_prev"] == pytest.approx(np.mean([10, 11, 12, 13]))
    assert df.loc[7, "pts_season_mean_prev"] == pytest.approx(np.mean(range(10, 17)))


def test_media_exponencial_bate_com_a_formula():
    df = build_features(_fake_players(n=8, start_pts=10))
    # linha 4: anteriores 10, 11, 12, 13; o mais recente pesa 1, o anterior 0,9...
    pesos = DECAY ** np.arange(3, -1, -1)
    esperado = float((pesos * np.array([10, 11, 12, 13])).sum() / pesos.sum())
    assert df.loc[4, "pts_ewm_prev"] == pytest.approx(esperado)
    # o ultimo jogo pesa mais que a media simples: a ewm fica acima dela numa serie crescente
    assert df.loc[4, "pts_ewm_prev"] > df.loc[4, "pts_season_mean_prev"]


def test_desvio_padrao_usa_ddof_1_e_exige_dois_jogos():
    df = build_features(_fake_players(n=8, start_pts=10))
    assert pd.isna(df.loc[1, "pts_std5_prev"])  # um unico jogo anterior
    assert df.loc[6, "pts_std5_prev"] == pytest.approx(np.std([11, 12, 13, 14, 15], ddof=1))


def test_jogos_jogados_conta_os_anteriores():
    df = build_features(_fake_players(n=9))
    assert df["games_played_prev"].tolist() == list(range(9))


def test_descanso_e_back_to_back():
    p = _fake_players(n=5)  # jogos a cada 2 dias
    p.loc[3, "GAME_DATE"] = p.loc[2, "GAME_DATE"] + pd.Timedelta(days=1)
    df = build_features(p)
    assert df.loc[1, "rest_days"] == 2 and df.loc[1, "is_b2b"] == 0
    assert df.loc[3, "rest_days"] == 1 and df.loc[3, "is_b2b"] == 1


def test_mando_de_quadra_vem_do_log():
    p = _fake_players(n=6)
    df = build_features(p)
    assert df["is_home"].astype(bool).tolist() == p["IS_HOME"].tolist()


def test_titular_estimado_usa_o_limiar_de_minutos():
    alto = build_features(_fake_players(n=4))
    assert alto["starter_proxy_prev"].dropna().eq(1).all()  # 30 min >= 28
    baixo_in = _fake_players(n=4)
    baixo_in["MIN"] = STARTER_MIN_THRESHOLD - 8
    assert build_features(baixo_in)["starter_proxy_prev"].dropna().eq(0).all()


def test_janelas_nao_atravessam_jogadores():
    a = _fake_players(n=6, player_id=1, start_pts=10)
    b = _fake_players(n=6, player_id=2, start_pts=40)
    df = build_features(pd.concat([a, b], ignore_index=True))
    primeira_do_segundo = df[df["PLAYER_ID"] == 2].iloc[0]
    assert pd.isna(primeira_do_segundo["pts_ma5_prev"])
    assert primeira_do_segundo["games_played_prev"] == 0
    segunda_do_segundo = df[df["PLAYER_ID"] == 2].iloc[1]
    assert segunda_do_segundo["pts_ma5_prev"] == pytest.approx(40.0)  # so o jogo 1 do jogador 2


def test_features_nao_alteram_o_dataframe_de_entrada():
    p = _fake_players(n=6)
    antes = p.copy()
    build_features(p)
    pd.testing.assert_frame_equal(p, antes)


# ----------------------------------------------------------------------------
# forca do adversario
# ----------------------------------------------------------------------------

def _teams() -> pd.DataFrame:
    datas = pd.date_range("2024-01-01", periods=3, freq="2D")
    pts_a, pts_b = [100, 110, 120], [90, 95, 105]
    linhas = []
    for i, d in enumerate(datas):
        gid = f"001{i:04d}"  # mesmos ids de _fake_players(player_id=1)
        linhas.append((gid, d, "2023-24", 1, "AAA", pts_a[i]))
        linhas.append((gid, d, "2023-24", 2, "BBB", pts_b[i]))
    return pd.DataFrame(linhas, columns=["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"])


def test_forca_do_adversario_usa_so_jogos_anteriores_dele():
    s = build_opponent_strength(_teams()).set_index(["GAME_ID", "OPPONENT"])
    # AAA, jogo 3: antes dele AAA sofreu 90 e 95 (pontos de BBB) e jogou partidas de 190 e 205 pontos
    r = s.loc[("0010002", "AAA")]
    assert r["opp_def_pts_allowed_prev"] == pytest.approx(92.5)
    assert r["opp_pace_proxy_prev"] == pytest.approx(197.5)
    assert r["opp_games_played_prev"] == 2
    # primeiro jogo: nada antes
    primeiro = s.loc[("0010000", "AAA")]
    assert pd.isna(primeiro["opp_def_pts_allowed_prev"]) and primeiro["opp_games_played_prev"] == 0


def test_adversario_do_jogador_e_juntado_pelo_oponente():
    df = build_features(_fake_players(n=3), _teams())  # jogador de AAA contra BBB
    # no jogo 3 o adversario BBB sofreu 100 e 110 (pontos de AAA) antes dele
    assert df.loc[2, "opp_def_pts_allowed_prev"] == pytest.approx(105.0)
    assert df.loc[2, "opp_pace_proxy_prev"] == pytest.approx(197.5)
    assert df.loc[2, "opp_games_played_prev"] == 2
    assert pd.isna(df.loc[0, "opp_def_pts_allowed_prev"])
    assert len(df) == 3  # o merge nao duplica linhas


# ----------------------------------------------------------------------------
# alvos e linha sintetica
# ----------------------------------------------------------------------------

def test_round_to_half():
    s = pd.Series([10.2, 10.3, 10.74, 10.76, 11.0])
    assert round_to_half(s).tolist() == [10.0, 10.5, 10.5, 11.0, 11.0]


def test_linha_sintetica_so_existe_com_dez_jogos_de_historico():
    p = _fake_players(n=12, start_pts=10)
    linha = make_synthetic_line(p)
    assert linha.iloc[:LINE_WINDOW].isna().all()
    assert linha.iloc[10] == pytest.approx(14.5)  # mediana de 10..19
    assert linha.iloc[11] == pytest.approx(15.5)  # mediana de 11..20


def test_linha_sintetica_reinicia_a_cada_temporada():
    a = _fake_players(n=12, season="2022-23")
    b = _fake_players(n=12, season="2023-24")
    b["GAME_DATE"] = b["GAME_DATE"] + pd.Timedelta(days=400)
    b["GAME_ID"] = ["x" + g for g in b["GAME_ID"]]
    linha = make_synthetic_line(pd.concat([a, b], ignore_index=True))
    assert linha.iloc[12:22].isna().all()  # os 10 primeiros jogos da 2a temporada
    assert not np.isnan(linha.iloc[22])


def test_linha_sintetica_preserva_o_indice_original():
    p = _fake_players(n=12).sample(frac=1, random_state=3)
    linha = make_synthetic_line(p)
    assert linha.index.equals(p.index)
    # o jogo com PTS=20 (11o jogo) e o unico cuja linha existe com 10 anteriores 10..19
    assert linha[p.index[p["PTS"] == 20][0]] == pytest.approx(14.5)


def test_empate_com_a_linha_conta_como_nao_superou():
    p = _fake_players(n=11)
    p["PTS"] = [12] * 11
    out = add_targets(p)
    assert out.loc[10, "line_synthetic"] == 12.0
    assert out.loc[10, "y_over"] == 0.0  # 12 > 12 e falso

    p["PTS"] = [12] * 10 + [13]
    assert add_targets(p).loc[10, "y_over"] == 1.0


def test_sem_historico_o_alvo_e_nan_e_nao_zero():
    out = add_targets(_fake_players(n=12))
    assert out["y_over"].iloc[:10].isna().all()
    assert out["y_over"].iloc[10:].notna().all()
    assert (out["y_pts"] == out["PTS"]).all()


def test_alvo_de_classificacao_so_assume_zero_ou_um():
    out = add_targets(_fake_players(n=30))
    assert set(out["y_over"].dropna().unique()) <= {0.0, 1.0}


def test_relatorio_do_alvo_conta_linhas():
    df = pd.DataFrame({"y_over": [np.nan, np.nan, 1.0, 0.0, 1.0]})
    r = target_report(df)
    assert r["linhas_totais"] == 5
    assert r["linhas_com_alvo"] == 3
    assert r["linhas_sem_historico"] == 2
    assert r["proporcao_classe_positiva"] == pytest.approx(0.6667)
    assert r["janela_da_linha"] == LINE_WINDOW


def test_relatorio_do_alvo_sem_linhas_validas():
    r = target_report(pd.DataFrame({"y_over": [np.nan, np.nan]}))
    assert r["linhas_com_alvo"] == 0
    assert np.isnan(r["proporcao_classe_positiva"])
