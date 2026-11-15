"""Gatilhos preventivos, dois alvos e cache do simulador."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.credito.chaves import GATILHOS, PARES, PONTOS, codificar, coerente, pontuar, vetor
from src.credito.politica import semaforo
from src.credito.combinacoes import medir_escada
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
    assert "O que a mesa avalia" in texto
    assert "Clique para abrir a lista completa" in texto
    assert "st.expander(ROTULO_MESA, expanded=False)" in texto
    for frase in (
        "Verdadeiro Negativo (Acerto — Bom pagador)",
        "Verdadeiro Positivo (Acerto — Inadimplente capturado)",
        "Falso Positivo (Alarme falso)",
        "Falso Negativo (Risco que escapou)",
        "Ficou em dia e o modelo aprovou.",
        "Atrasou e o modelo alertou antes.",
        "Ficou em dia, mas o modelo marcou risco.",
        "Ficou inadimplente, mas passou sem alerta do modelo.",
    ):
        assert frase in texto
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
    assert 0.11 <= float(df["inadimplente"].mean()) <= 0.15
    assert 0.05 <= float(df["inadimplente_90d"].mean()) <= 0.08
    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    for slug in ("inadimplente", "inadimplente_90d"):
        for _modelo, itens in relatorio["alvos"][slug]["importancia"]["itens"].items():
            pcts = [item["pct"] for item in itens]
            assert abs(sum(pcts) - 100.0) < 0.05
            assert pcts == sorted(pcts, reverse=True)
        blocos = relatorio["alvos"][slug]["importancia"].get("blocos") or {}
        for _modelo, itens_bloco in blocos.items():
            fatias = [item["pct"] for item in itens_bloco]
            assert abs(sum(fatias) - 100.0) < 0.05
            assert fatias == sorted(fatias, reverse=True)


def test_reincidencia_substitui_e_nao_soma():
    esperado = {
        "protesto_3_ou_mais": 30,
        "mais_de_um_cheque_sem_fundo_12m": 25,
        "uso_cheque_especial_mais_3m_em_6m": 12,
        "rotativo_mais_2x_6m": 18,
        "cartao_acima_90_mais_3x_6m": 12,
    }
    for filho, pai in PARES:
        linha = vetor({filho: True})
        assert linha[filho] == 1 and linha[pai] == 1
        exclusivo = codificar(linha)
        assert exclusivo[filho] == 1 and exclusivo[pai] == 0
        assert pontuar(linha) == esperado[filho]
        assert pontuar(linha) != PONTOS[pai] + PONTOS[filho]
    simples = vetor({"protesto_cartorio": True})
    assert simples["protesto_3_ou_mais"] == 0
    assert pontuar(simples) == 18
    faixa, _postura, trava = semaforo(pontuar(simples), False)
    assert faixa == "amarelo" and not trava
    faixa, _postura, trava = semaforo(0, False)
    assert faixa == "verde"
    faixa, _postura, trava = semaforo(26, False)
    assert faixa == "vermelho"
    _faixa, _postura, trava = semaforo(35, True)
    assert trava


def test_a_tela_nao_mostra_a_pontuacao_interna():
    texto = RELATO.read_text(encoding="utf-8").lower()
    assert " pts" not in texto
    assert "pontos" not in texto
    assert "semáforo do comitê" in texto
    assert "hard stop" in texto
    assert "baixo risco" in texto


def test_gatilho_isolado_nos_quatro_modelos():
    joblib = RAIZ / "data" / "credito" / "modelos.joblib"
    if not joblib.exists():
        return
    from src.credito.simulador import carregar_modelos

    modelos = carregar_modelos()
    slugs = ("logistica", "floresta", "boosting", "mlp")

    def probs(respostas: dict) -> list[float]:
        linha = vetor(respostas)
        return [chances(modelos, slug, linha)[0] for slug in slugs]

    def faixa_de(respostas: dict) -> str:
        linha = vetor(respostas)
        faixa, _postura, _trava = semaforo(
            pontuar(linha), bool(linha["restricao_outras_instituicoes"])
        )
        return faixa

    zeros = probs({})
    assert faixa_de({}) == "verde"
    assert all(p < 0.05 for p in zeros), zeros

    protesto = probs({"protesto_cartorio": True})
    assert faixa_de({"protesto_cartorio": True}) == "amarelo"
    assert all(0.18 <= p <= 0.40 for p in protesto), protesto

    grave = probs({"protesto_3_ou_mais": True})
    assert faixa_de({"protesto_3_ou_mais": True}) == "vermelho"
    assert all(p > 0.50 for p in grave), grave

    trava = probs({"restricao_outras_instituicoes": True})
    assert faixa_de({"restricao_outras_instituicoes": True}) == "vermelho"
    assert all(p > 0.65 for p in trava), trava


def test_escada_sobe_quando_os_gatilhos_se_acumulam():
    sinais = np.array([0, 0, 2, 4, 6, 8, 11, 12])
    inad = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    grave = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    degraus = medir_escada(sinais, inad, grave)
    assert [item["rotulo"] for item in degraus] == ["0", "1–2", "3–4", "5–6", "7–8", "9–10", "11+"]
    assert degraus[0]["taxa_inad"] == 0
    assert degraus[-1]["taxa_inad"] == 1
    assert degraus[-1]["taxa_90"] > degraus[0]["taxa_90"]
    assert degraus[-1]["n"] == 2
