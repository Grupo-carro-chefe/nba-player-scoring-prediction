# === CABECALHO ACADEMICO (gerado por scripts/apply_header.py) ===
# Universidade Presbiteriana Mackenzie
# Disciplina: Inteligência Artificial
# Projeto: Previsão de desempenho de jogadores da NBA (pontos por jogo)
#
# INTEGRANTES DO GRUPO:
#   - Rodrigo Lucas Mascarenhas Leite Oliveira - TIA 10427925
#   - André Ihsan Ward - TIA 10425684
#
# ARQUIVO: tests/test_predict.py
# SINTESE: Artefato de demonstração: consultas, recusas, formato e reprodução dos resultados.
#
# HISTORICO DE ALTERACOES:
#   data       | autor      | descricao
#   -----------|------------|-------------------------------------------
#   2026-11-17 | Rodrigo    | Versão inicial: artefato de demonstração.
# === FIM DO CABECALHO ===

"""O artefato de demonstracao (src/predict.py), que e o que o video mostra.

Cobre as consultas aceitas, as recusas, o formato da saida e, principalmente,
que o modelo salvo em disco reproduz as previsoes registradas em
results/predictions_test.parquet: a prova de que o artefato demonstrado e o
mesmo que gerou os resultados do relatorio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import predict
from src.modeling import load_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset()


@pytest.fixture(scope="module")
def artefatos():
    return predict.load_artifacts()


@pytest.fixture()
def cli(monkeypatch, capsys, df, artefatos):
    """Roda main() com os argumentos dados, sem reler o dataset e os modelos a cada teste."""
    monkeypatch.setattr(predict, "load_dataset", lambda: df)
    monkeypatch.setattr(predict, "load_artifacts", lambda: artefatos)

    def executar(*args: str) -> str:
        monkeypatch.setattr(sys, "argv", ["predict", *args])
        predict.main()
        return capsys.readouterr().out

    return executar


# ----------------------------------------------------------------------------
# funcoes auxiliares
# ----------------------------------------------------------------------------

def test_normalizacao_de_nomes_remove_acentos_e_caixa():
    assert predict._norm("  José Álvarez ") == "jose alvarez"
    assert predict._norm("Luka Dončić") == "luka doncic"
    assert predict._norm("Nikola Jokić") == "nikola jokic"


def test_nomes_com_acento_do_dataset_sao_encontraveis_sem_acento(df):
    nomes = df["PLAYER_NAME"].drop_duplicates()
    com_acento = [n for n in nomes if predict._norm(n) != n.lower()]
    assert com_acento, "o dataset deveria ter jogadores com acento no nome"
    for nome in com_acento[:10]:
        assert predict._norm(nome) in nomes.map(predict._norm).tolist()


def test_artefatos_carregam_com_os_dois_niveis_de_intervalo(artefatos):
    reg, clf, conf = artefatos
    assert set(conf) == {"80", "90"}
    assert 0 < conf["80"].q < conf["90"].q  # mais confianca, intervalo mais largo
    assert conf["80"].alpha == pytest.approx(0.2) and conf["90"].alpha == pytest.approx(0.1)
    assert hasattr(reg, "predict") and hasattr(clf, "predict_proba")


def _linha(**mudancas) -> pd.Series:
    base = {
        "PLAYER_NAME": "Jalen Brunson", "TEAM_ABBREVIATION": "NYK", "OPPONENT": "POR",
        "GAME_DATE": pd.Timestamp("2026-01-11"), "split": "test",
        "previsto": 26.8, "ic80_inf": 14.8, "ic80_sup": 38.9, "ic90_inf": 11.6, "ic90_sup": 42.1,
        "pts_ma5_prev": 26.6, "line_synthetic": 28.5, "prob_superar_patamar": 0.43,
        "min_ma5_prev": 34.0, "y_pts": 26.0,
    }
    return pd.Series({**base, **mudancas})


def test_saida_formatada_com_virgula_decimal():
    texto = predict.render(_linha())
    assert "Jalen Brunson (NYK) x POR  |  11/01/2026  |  split: test" in texto
    assert "Pontos previstos ............ 26,8" in texto
    assert "Intervalo de 80% ............ 14,8 a 38,9" in texto
    assert "Intervalo de 90% ............ 11,6 a 42,1" in texto
    assert "P(superar o patamar) ........ 43%" in texto
    assert "Real: 26 pontos  |  erro do modelo: -0,8  |  erro da baseline: -0,6" in texto
    assert "Dentro do intervalo de 80%? sim  |  superou o patamar? não" in texto


def test_saida_acusa_quando_o_real_ficou_fora_e_acima_da_linha():
    texto = predict.render(_linha(y_pts=40.0, ic80_sup=38.9))
    assert "Dentro do intervalo de 80%? não  |  superou o patamar? sim" in texto


def test_limites_do_intervalo_sao_inclusivos():
    assert "Dentro do intervalo de 80%? sim" in predict.render(_linha(y_pts=38.9))
    assert "Dentro do intervalo de 80%? sim" in predict.render(_linha(y_pts=14.8))


# ----------------------------------------------------------------------------
# previsao em lote
# ----------------------------------------------------------------------------

def test_previsoes_tem_intervalos_aninhados_e_probabilidades_validas(df, artefatos):
    rows = df[df["split"] == "test"].sample(300, random_state=1).reset_index(drop=True)
    colunas_antes = list(rows.columns)
    out = predict.predict_rows(rows, *artefatos)
    assert list(rows.columns) == colunas_antes, "predict_rows nao pode alterar a entrada"
    assert len(out) == 300 and np.isfinite(out["previsto"]).all()
    assert (out["ic80_inf"] >= 0).all() and (out["ic90_inf"] >= 0).all()
    assert (out["ic90_inf"] <= out["ic80_inf"]).all() and (out["ic80_sup"] <= out["ic90_sup"]).all()
    assert (out["ic80_inf"] <= out["ic80_sup"]).all()
    assert (out["ic80_inf"] <= out["previsto"].clip(lower=0)).all() and (out["previsto"] <= out["ic80_sup"]).all()
    assert out["prob_superar_patamar"].between(0, 1).all()
    pd.testing.assert_frame_equal(out, predict.predict_rows(rows, *artefatos))  # deterministico


def test_modelo_salvo_reproduz_as_previsoes_registradas_do_teste(df, artefatos):
    registradas = pd.read_parquet(ROOT / "results" / "predictions_test.parquet")
    teste = df[df["split"] == "test"].sample(400, random_state=2).reset_index(drop=True)
    out = predict.predict_rows(teste, *artefatos)
    out[["PLAYER_ID", "GAME_ID"]] = teste[["PLAYER_ID", "GAME_ID"]]
    junto = out.merge(registradas, on=["PLAYER_ID", "GAME_ID"], validate="one_to_one")
    assert len(junto) == 400
    assert np.allclose(junto["previsto"], junto["reg__hgb_absoluto"], rtol=1e-6, atol=1e-6)
    assert np.allclose(junto["ic80_inf"], junto["intervalo80_inf"], atol=1e-6)
    assert np.allclose(junto["ic90_sup"], junto["intervalo90_sup"], atol=1e-6)
    assert np.allclose(junto["prob_superar_patamar"], junto["clf__final_calibrado"], atol=1e-6)


# ----------------------------------------------------------------------------
# linha de comando
# ----------------------------------------------------------------------------

def test_consulta_do_roteiro_do_video(cli):
    out = cli("--player", "Jalen Brunson", "--date", "2026-01-11")
    assert "Jalen Brunson (NYK) x POR" in out
    assert "Pontos previstos ............ 26,8" in out and "Real: 26 pontos" in out
    assert out.rstrip().endswith(predict.AVISO)


def test_aviso_de_que_nao_e_recomendacao_de_aposta(cli):
    out = cli("--sample", "1")
    assert "Não é recomendação de aposta" in out
    assert "odds" not in out.lower()


def test_busca_por_nome_sem_acento(cli):
    assert "Nikola Jokić" in cli("--player", "jokic")  # sem data: os jogos mais recentes


def test_jogador_sem_data_mostra_os_cinco_jogos_mais_recentes(cli):
    out = cli("--player", "Brunson")
    assert out.count("Pontos previstos") == 5
    assert "Resumo: 5 jogos" in out
    datas = [l.split("|")[1].strip() for l in out.splitlines() if "split:" in l]
    assert datas == sorted(datas, key=lambda d: d.split("/")[::-1])  # cronologico


def test_rotacao_de_um_time_num_jogo(cli):
    out = cli("--team", "BOS", "--date", "2026-01-10")
    cabecalhos = [l for l in out.splitlines() if "split:" in l]
    assert len(cabecalhos) >= 5 and all("(BOS)" in l for l in cabecalhos)


def test_time_sem_data_mostra_so_o_jogo_mais_recente(cli):
    out = cli("--team", "BOS")
    datas = {l.split("|")[1].strip() for l in out.splitlines() if "split:" in l}
    assert len(datas) == 1


def test_sorteio_de_jogos_e_reprodutivel_e_so_do_teste(cli):
    a = cli("--sample", "3", "--seed", "7")
    assert a == cli("--sample", "3", "--seed", "7")
    assert a.count("split: test") == 3 and "split: train" not in a and "split: val" not in a
    assert "Resumo: 3 jogos" in a


CONSULTAS_RECUSADAS = [
    (("--player", "ninguem", "--date", "2026-01-11"), "fora do escopo"),
    (("--player", "Brunson", "--date", "2025-07-01"), "fora do escopo"),  # entressafra
    (("--player", "Brunson", "--date", "2026-05-01"), "fora do escopo"),  # pos-temporada
    (("--player", "james"), "mais de um jogador"),
    (("--sample", "0"), "--sample deve estar entre 1 e"),
    (("--sample", "99999999"), "--sample deve estar entre 1 e"),
    ((), "informe --player, --team ou --sample"),
    (("--date", "2026-01-11"), "informe --player, --team ou --sample"),
]


def test_consultas_fora_do_dominio_sao_recusadas_com_motivo(cli):
    """Em vez de inventar um numero, o artefato recusa (laco, nao parametrize: o relatorio conta funcoes de teste)."""
    for args, trecho in CONSULTAS_RECUSADAS:
        with pytest.raises(SystemExit) as erro:
            cli(*args)
        assert trecho in str(erro.value), f"consulta {args}: {erro.value}"
