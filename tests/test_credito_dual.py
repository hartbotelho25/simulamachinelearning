"""Gatilhos preventivos, dois alvos e cache do simulador."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.credito.chaves import GATILHOS, PARES, coerente, vetor
from src.credito.politica import nome_faixa
from src.credito.combinacoes import medir_escada
from src.credito.dataset import COLUNAS, FEATURES, PROIBIDAS, gerar_base
from src.credito.simulador import chances, linha_para_frame

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


def test_a_chance_e_o_predict_proba_sem_ajuste():
    modelos = {
        "inadimplente": {"logistica": _Falso(0.22)},
        "inadimplente_90d": {"logistica": _Falso(0.81)},
    }
    linha = {coluna: 0 for coluna in FEATURES}
    p_inad, p_90 = chances(modelos, "logistica", linha)
    assert p_inad == 0.22
    assert p_90 == 0.81
    assert p_inad == modelos["inadimplente"]["logistica"].predict_proba(linha_para_frame(linha))[0, 1]
    assert p_90 == modelos["inadimplente_90d"]["logistica"].predict_proba(linha_para_frame(linha))[0, 1]


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
    assert "O código" in texto
    assert "Quando os sinais se juntam" not in texto
    assert "Primeira versão" not in texto
    assert "linha-tempo" not in texto
    assert "text/csv" not in texto
    assert "st.expander(titulo, expanded=False)" in texto
    assert "download_button" not in texto
    for caminho in list((RAIZ / "src").rglob("*.py")) + [RAIZ / "app.py"]:
        assert "download_button" not in caminho.read_text(encoding="utf-8"), caminho
    for nome in (
        "Regressão logística",
        "Random Forest",
        "Gradient Boosting (XGBoost)",
        "Rede neural MLP",
    ):
        assert nome in texto
    url = (
        "https://raw.githubusercontent.com/hartbotelho25/simulamachinelearning/"
        "main/data/credito/clientes_credito_50k.csv"
    )
    for slug in ("regressao_logistica", "random_forest", "gradient_boosting", "rede_neural"):
        corpo = (RAIZ / "notebooks" / f"{slug}.py").read_text(encoding="utf-8")
        assert url in corpo
        assert "roc_auc_score" in corpo
        assert "confusion_matrix" in corpo
        assert "train_test_split" in corpo
        if slug == "rede_neural":
            assert "IsotonicRegression" in corpo
        else:
            assert 'method="isotonic"' in corpo
    rede = (RAIZ / "notebooks" / "rede_neural.py").read_text(encoding="utf-8")
    assert 'activation="relu"' in rede
    assert 'activation="sigmoid"' in rede
    assert "epochs=" in rede
    assert "binary_crossentropy" in rede
    assert "EarlyStopping" in rede
    assert "restore_best_weights=True" in rede
    assert 'class_weight="balanced"' in rede
    treino = (RAIZ / "src" / "credito" / "treino.py").read_text(encoding="utf-8")
    assert "early_stopping=True" in treino
    assert "compute_sample_weight" in treino
    assert 'class_weight="balanced"' in treino
    assert "CalibratedClassifierCV" in treino
    assert 'method="isotonic"' in treino
    assert "@st.fragment" in texto
    simulador = texto.split("@st.fragment", 1)[1]
    assert "st.expander" not in simulador
    assert "st.slider" not in texto
    assert "st.number_input" not in texto
    assert len(GATILHOS) == 23
    assert len(FEATURES) == 23
    assert FEATURES == [item["id"] for item in GATILHOS]
    estudo = texto.split("def render_estudo", 1)[1].split("def render_simulador", 1)[0]
    assert "_simulador(" not in estudo
    app = (RAIZ / "app.py").read_text(encoding="utf-8")
    assert "Machine Learning Aplicado à Inadimplência" in app
    assert "Simulador de Inadimplência" in app
    assert "Simulador do Brasileirão" in app
    assert "Por onde você entra?" not in app
    assert "PDF" not in app
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


def test_o_modelo_recebe_o_vetor_binario_intacto():
    linha = vetor({"protesto_3_ou_mais": True, "salto_gastos_cartao": True})
    assert linha["protesto_3_ou_mais"] == 1
    assert linha["protesto_cartorio"] == 1
    frame = linha_para_frame(linha)
    assert list(frame.columns) == list(FEATURES)
    assert set(frame.to_numpy().ravel()).issubset({0, 1})
    assert int(frame.iloc[0]["protesto_cartorio"]) == 1
    assert int(frame.iloc[0]["protesto_3_ou_mais"]) == 1
    assert int(frame.iloc[0]["salto_gastos_cartao"]) == 1


def test_a_faixa_sai_da_probabilidade():
    assert nome_faixa(0.19) == ("verde", "Aprovar")
    assert nome_faixa(0.20) == ("amarelo", "Atenção")
    assert nome_faixa(0.49) == ("amarelo", "Atenção")
    assert nome_faixa(0.50) == ("vermelho", "Recusar")


def test_nao_ha_scorecard_manual():
    texto = RELATO.read_text(encoding="utf-8").lower()
    assert " pts" not in texto
    assert "pontos" not in texto
    assert "scorecard" not in texto
    assert "hard stop" not in texto
    assert "regra de negócio" not in texto
    assert not (RAIZ / "src" / "credito" / "scorecard.py").exists()
    fontes = "\n".join(
        caminho.read_text(encoding="utf-8").lower()
        for caminho in (RAIZ / "src" / "credito").glob("*.py")
    )
    assert "scorecard" not in fontes
    assert "pontos" not in fontes
    assert "from src.credito.scorecard" not in fontes
    assert "import scorecard" not in fontes


def test_gatilho_isolado_nos_quatro_modelos():
    arquivo = RAIZ / "data" / "credito" / "modelos.joblib"
    if not arquivo.exists():
        return
    from src.credito.simulador import carregar_modelos

    modelos = carregar_modelos()
    slugs = ("logistica", "floresta", "boosting", "mlp")

    def probs(respostas: dict) -> list[float]:
        linha = vetor(respostas)
        saida = []
        for slug in slugs:
            prevista, _outra = chances(modelos, slug, linha)
            bruta = float(modelos["inadimplente"][slug].predict_proba(linha_para_frame(linha))[0, 1])
            assert abs(prevista - bruta) < 1e-12
            saida.append(prevista)
        return saida

    zeros = probs({})
    assert all(p < 0.08 for p in zeros), zeros
    assert max(zeros) - min(zeros) < 0.06, zeros
    assert all(nome_faixa(p)[0] == "verde" for p in zeros)

    dois = probs({"salto_gastos_cartao": True, "novo_emprestimo_6m": True})
    assert all(p < 0.30 for p in dois), dois

    grave = probs({"restricao_outras_instituicoes": True})
    assert all(0.10 <= p <= 0.40 for p in grave), grave

    protesto = probs({"protesto_cartorio": True})
    assert all(p < 0.25 for p in protesto), protesto

    vários = probs(
        {
            "restricao_outras_instituicoes": True,
            "protesto_3_ou_mais": True,
            "renegociacao_ativa": True,
        }
    )
    assert all(0.35 <= p <= 0.80 for p in vários), vários
    assert max(vários) - min(vários) < 0.25, vários


def test_joblib_abre_sem_codigo_do_projeto():
    import subprocess
    import sys

    arquivo = RAIZ / "data" / "credito" / "modelos.joblib"
    if not arquivo.exists():
        return
    comando = (
        "import joblib\n"
        "from joblib import numpy_pickle as npickle\n"
        "original = npickle.NumpyUnpickler.find_class\n"
        "def find_class(self, module, name):\n"
        "    if module == '__main__' or module.startswith('src.'):\n"
        "        raise SystemExit(f'referencia proibida: {module}.{name}')\n"
        "    return original(self, module, name)\n"
        "npickle.NumpyUnpickler.find_class = find_class\n"
        "modelos = joblib.load('data/credito/modelos.joblib')\n"
        "for bloco in modelos.values():\n"
        "    for modelo in bloco.values():\n"
        "        passos = getattr(modelo, 'named_steps', None)\n"
        "        nomes = [type(modelo).__name__]\n"
        "        if passos:\n"
        "            nomes.extend(type(passo).__name__ for passo in passos.values())\n"
        "        if 'FunctionTransformer' in nomes:\n"
        "            raise SystemExit('FunctionTransformer ainda esta no joblib')\n"
        "print('ok')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", comando],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr or resultado.stdout
    assert "ok" in resultado.stdout


def test_joblib_quebrado_treina_a_partir_do_csv(monkeypatch):
    def estoura(*_args, **_kwargs):
        raise AttributeError("find_class")

    monkeypatch.setattr("src.credito.simulador.joblib.load", estoura)
    from src.credito.simulador import carregar_modelos, chances, resolver_modelos

    modelos = resolver_modelos(carregar_modelos, lambda: None, carregar_modelos)
    slugs = ("logistica", "floresta", "boosting", "mlp")
    zeros = {coluna: 0 for coluna in FEATURES}
    for slug in slugs:
        p_inad, p_90 = chances(modelos, slug, zeros)
        assert 0.0 <= p_90 <= p_inad <= 1.0
        restricao = vetor({"restricao_outras_instituicoes": True})
        alta = [chances(modelos, slug, restricao)[0] for slug in slugs]
        assert all(0.0 < p < 1.0 for p in alta), alta


def test_joblib_calibrado_abre_pesos_e_o_simulador():
    import joblib
    from streamlit.testing.v1 import AppTest

    from src.credito.importancia import pesos_do_modelo, ranking_itens
    from src.credito.simulador import pesos_do_slug

    arquivo = RAIZ / "data" / "credito" / "modelos.joblib"
    modelos = joblib.load(arquivo)
    slugs = ("logistica", "floresta", "boosting", "mlp")
    for alvo in ("inadimplente", "inadimplente_90d"):
        for slug in slugs:
            assert type(modelos[alvo][slug]).__name__ == "CalibratedClassifierCV"
            itens = ranking_itens(pesos_do_modelo(modelos[alvo][slug]))
            pcts = [item["pct"] for item in itens]
            assert pcts, (alvo, slug)
            assert abs(sum(pcts) - 100.0) < 0.05, (alvo, slug, sum(pcts))
            assert pcts == sorted(pcts, reverse=True), (alvo, slug, pcts)
            fatia = pesos_do_slug(modelos, slug)
            assert abs(sum(fatia.values()) - 100.0) < 0.05, (slug, sum(fatia.values()))

    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=90)
    app.query_params["estudo"] = "simulador"
    app.run()
    assert not app.exception, app.exception
    for slug in slugs:
        app.selectbox[0].set_value(slug).run()
        assert not app.exception, app.exception
    for indice in range(4):
        app.toggle[indice].set_value(True).run()
        assert not app.exception, app.exception
    cartao = " ".join(getattr(bloco, "value", "") for bloco in app.markdown)
    assert "Predição do modelo" in cartao
    assert "Vai ficar inadimplente?" in cartao
    assert "Vai passar de 90 dias?" in cartao

    estudo = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=90)
    estudo.query_params["estudo"] = "ml"
    estudo.run()
    assert not estudo.exception, estudo.exception
    titulos = [getattr(bloco, "value", "") for bloco in estudo.subheader]
    assert "O código" in titulos


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
