"""Gatilhos preventivos, dois alvos e cache do simulador."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.credito.chaves import GATILHOS, PARES, coerente, vetor
from src.credito.dataset import COLUNAS, FEATURES, PROIBIDAS, gerar_base
from src.credito.simulador import chances

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data" / "credito" / "clientes_credito_50k.csv"
RELATORIO = RAIZ / "data" / "credito" / "relatorio.json"
RELATO = RAIZ / "src" / "credito" / "relato.py"


class _Falso:
    def __init__(self, probabilidade: float) -> None:
        self.probabilidade = probabilidade

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.probabilidade
        return np.array([[1.0 - p, p]])


def test_reincidencia_liga_o_gatilho_base():
    respostas = coerente({"protesto_3_ou_mais": True, "mais_de_um_cheque_sem_fundo_12m": True})
    assert respostas["protesto_cartorio"]
    assert respostas["cheque_sem_fundo_12m"]
    assert not respostas["renegociacao_ativa"]
    linha = vetor(respostas)
    assert linha["protesto_3_ou_mais"] == 1
    assert linha["protesto_cartorio"] == 1
    assert set(linha) == set(FEATURES)


def test_chance_de_90_dias_nao_passa_da_inadimplencia():
    modelos = {
        "inadimplente": {"logistica": _Falso(0.22)},
        "inadimplente_90d": {"logistica": _Falso(0.81)},
    }
    p_inad, p_90 = chances(modelos, "logistica", {coluna: 0 for coluna in FEATURES})
    assert p_inad == 0.22
    assert p_90 == 0.22


def test_base_curta_respeita_hierarquia_e_exclusoes():
    df, _parametros = gerar_base(3000, 7, prefixo_id="T")
    assert list(df.columns) == COLUNAS
    assert not (PROIBIDAS & set(df.columns))
    assert int((df["inadimplente_90d"] > df["inadimplente"]).sum()) == 0
    for filho, pai in PARES:
        assert int(((df[filho] == 1) & (df[pai] == 0)).sum()) == 0
    assert set(df[FEATURES].to_numpy().ravel()).issubset({0, 1})


def test_tela_deixa_a_mesa_recolhida_e_o_simulador_aberto():
    texto = RELATO.read_text(encoding="utf-8")
    assert 'st.expander("O que a mesa avalia", expanded=False)' in texto
    assert "@st.fragment" in texto
    simulador = texto.split("@st.fragment", 1)[1]
    assert "st.expander" not in simulador
    assert "st.slider" not in texto
    assert "st.number_input" not in texto
    assert len(GATILHOS) == 23
    estudo = texto.split("def render_estudo", 1)[1].split("def render_simulador", 1)[0]
    assert "_simulador(" not in estudo
    app = (RAIZ / "app.py").read_text(encoding="utf-8")
    assert "Machine Learning Aplicado à Inadimplência" in app
    assert "Simulador de Inadimplência" in app
    assert "Simula Brasileirão" in app
    assert '"ml"' in app and '"simulador"' in app


def test_livro_e_pesos_quando_o_pipeline_ja_rodou():
    if not CSV.exists() or not RELATORIO.exists():
        return
    import json

    df = pd.read_csv(CSV)
    assert len(df) == 50_000
    assert list(df.columns) == COLUNAS
    assert not (PROIBIDAS & set(df.columns))
    assert int((df["inadimplente_90d"] > df["inadimplente"]).sum()) == 0
    for filho, pai in PARES:
        assert int(((df[filho] == 1) & (df[pai] == 0)).sum()) == 0
    assert 0.11 <= float(df["inadimplente"].mean()) <= 0.13
    assert 0.05 <= float(df["inadimplente_90d"].mean()) <= 0.07
    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    for slug in ("inadimplente", "inadimplente_90d"):
        for _modelo, itens in relatorio["alvos"][slug]["importancia"]["itens"].items():
            pcts = [item["pct"] for item in itens]
            assert abs(sum(pcts) - 100.0) < 0.05
            assert pcts == sorted(pcts, reverse=True)
