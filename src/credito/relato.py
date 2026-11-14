"""Relatório executivo estático e simulador ao lado.

Os números e as figuras vêm do pipeline. O simulador é o único trecho ao vivo.
"""

from __future__ import annotations

import streamlit as st

from .dataset import BLOCO_NOME, BLOCOS, CATEGORICAS, FEATURES, LIMITES, ROTULOS
from .politica import CORTE_MATRIZ, nome_faixa
from .simulador import carregar_modelos, chance, cliente_novo, figura, ler_perfil, ler_relatorio, pronto

ROTULO_CAT = {
    "clt": "CLT",
    "servidor": "Servidor",
    "autonomo": "Autônomo",
    "empresario": "Empresário",
    "aposentado": "Aposentado",
    "informal": "Informal",
    "fundamental": "Fundamental",
    "medio": "Médio",
    "superior": "Superior",
    "pos": "Pós-graduação",
    "integral": "Integral",
    "parcial": "Parcial",
    "minimo": "Mínimo",
}
FLAGS = {
    "possui_imovel_proprio",
    "flag_pagamento_cartao_com_cheque_especial",
    "flag_historico_fpd",
    "possui_renegociacao_ativa",
}
COR_FAIXA = {"verde": "#3dd68c", "amarelo": "#E8B923", "vermelho": "#ff8a65"}
BLOCO_CURTO = {
    "cadastral": "Cadastral",
    "fluxo": "Fluxo de caixa",
    "cheque": "Cheque especial",
    "cartao": "Cartão",
    "endividamento": "Empréstimos",
    "regua": "Régua de atrasos",
    "bureau": "Bureau",
}


@st.cache_resource
def _modelos_gravados():
    return carregar_modelos()


def _pct(valor: float | None) -> str:
    if valor is None:
        return "—"
    return f"{100 * valor:.1f}%".replace(".", ",")


def _num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _reais(valor: float) -> str:
    texto = f"{valor:,.0f}"
    return "R$ " + texto.replace(",", "X").replace(".", ",").replace("X", ".")


def _alinhar(coluna: str, valor) -> float | int | str:
    if coluna in CATEGORICAS:
        return str(valor)
    baixo, alto, passo = LIMITES[coluna]
    if coluna in FLAGS:
        return int(valor) == 1
    numero = float(valor)
    numero = min(max(numero, float(baixo)), float(alto))
    degraus = round((numero - float(baixo)) / float(passo))
    ajustado = float(baixo) + degraus * float(passo)
    ajustado = min(max(ajustado, float(baixo)), float(alto))
    if float(passo).is_integer() and float(baixo).is_integer():
        return int(round(ajustado))
    return round(ajustado, 4)


def _valor_tipico(perfil: dict, coluna: str):
    info = perfil[coluna]
    if info["tipo"] == "categoria":
        return info["moda"]
    return _alinhar(coluna, info["mediana"])


def _imagem(relatorio: dict, chave: str, legenda: str) -> None:
    nome = relatorio["figuras"][chave]
    caminho = figura(nome)
    st.image(str(caminho), width="stretch")
    st.caption(legenda)


def _mil(valor: float) -> str:
    return f"{int(round(valor)):,}".replace(",", ".")


def _kpis_base(base: dict) -> None:
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi" style="--accent:#E8B923"><div class="lbl">Clientes</div>
            <div class="val">{_mil(base['n'])}</div><div class="sub">Livro sintético</div></div>
            <div class="kpi" style="--accent:#ff8a65"><div class="lbl">Inadimplência 90 dias</div>
            <div class="val">{_pct(base['taxa'])}</div><div class="sub">{_mil(base['positivos'])} positivos · taxa realizada</div></div>
            <div class="kpi" style="--accent:#6ea8fe"><div class="lbl">Teste</div>
            <div class="val">{_mil(base['n_teste'])}</div><div class="sub">{_pct(base['taxa_teste'])} inadimplentes no teste</div></div>
            <div class="kpi" style="--accent:#3dd68c"><div class="lbl">Renda mediana</div>
            <div class="val" style="font-size:1.35rem">{_reais(base['renda_mediana'])}</div>
            <div class="sub">Idade mediana {base['idade_mediana']:.0f} · score mediano {base['score_mediano']:.0f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tabela(relatorio: dict) -> None:
    import pandas as pd

    linhas = []
    for modelo in relatorio["modelos"]:
        m = modelo["matriz"]
        linhas.append(
            {
                "Modelo": modelo["nome"],
                "ROC AUC": _num(modelo["roc_auc"]),
                "Precisão (%)": _num(m["precisao"] * 100, 1),
                "Recall (%)": _num(m["recall"] * 100, 1),
                "F1 (%)": _num(m["f1"] * 100, 1),
                "Acurácia (%)": _num(m["acuracia"] * 100, 1),
                "FP (alarme)": m["fp"],
                "FN (perdidos)": m["fn"],
            }
        )
    st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)


def _controles(perfil: dict) -> dict:
    linha = {}
    for bloco, colunas in BLOCOS.items():
        with st.expander(BLOCO_CURTO[bloco], expanded=False):
            for coluna in colunas:
                chave = f"sim_{coluna}"
                if chave not in st.session_state:
                    st.session_state[chave] = _valor_tipico(perfil, coluna)
                rotulo = ROTULOS[coluna]
                if coluna in CATEGORICAS:
                    opcoes = perfil[coluna]["opcoes"]
                    escolha = st.selectbox(rotulo, opcoes, key=chave, format_func=lambda v: ROTULO_CAT.get(v, v))
                    linha[coluna] = escolha
                elif coluna in FLAGS:
                    linha[coluna] = int(st.toggle(rotulo, key=chave))
                else:
                    baixo, alto, passo = LIMITES[coluna]
                    if float(passo).is_integer() and float(baixo).is_integer():
                        linha[coluna] = int(st.slider(rotulo, int(baixo), int(alto), step=int(passo), key=chave))
                    else:
                        linha[coluna] = float(st.slider(rotulo, float(baixo), float(alto), step=float(passo), key=chave))
    return linha


def _simulador(relatorio: dict, ir) -> str:
    with st.sidebar:
        if st.button("Voltar ao menu", key="voltar_credito", width="stretch"):
            ir("home")
        st.markdown("**Comitê de Crédito**")
        st.markdown("**Simulador**")
        st.caption("Os blocos abaixo mudam a chance ao vivo. O cliente novo confere o acerto no corte 0,50.")
        try:
            modelos = _modelos_gravados()
        except Exception:
            st.warning(
                "Os modelos gravados não abriram nesta instalação. O relatório continua nesta página. "
                "requirements.txt fixa scikit-learn e xgboost na versão em que eles foram salvos."
            )
            return ""
        nomes = {item["slug"]: item["nome"] for item in relatorio["modelos"]}
        slug = st.selectbox(
            "Modelo",
            list(nomes),
            index=list(nomes).index(relatorio["referencia"]),
            format_func=lambda s: nomes[s],
            key="sim_modelo",
        )
        perfil = ler_perfil()
        for coluna in FEATURES:
            chave = f"sim_{coluna}"
            if chave not in st.session_state:
                st.session_state[chave] = _valor_tipico(perfil, coluna)
        linha = {}
        for coluna in FEATURES:
            valor = st.session_state[f"sim_{coluna}"]
            linha[coluna] = int(bool(valor)) if coluna in FLAGS else valor
        probabilidade = chance(modelos[slug], linha)
        faixa, acao = nome_faixa(probabilidade)
        cor = COR_FAIXA[faixa]
        st.markdown(
            f"""
            <div class="kpi" style="--accent:{cor};margin-top:0.6rem">
                <div class="lbl">Chance ao vivo</div>
                <div class="val">{_pct(probabilidade)}</div>
                <div class="sub">{faixa} · {acao}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Modelo gravado: {nomes[slug]}. A chance usa os blocos como estão agora.")
        st.markdown("**Cliente novo**")
        if st.button("Criar cliente novo", type="primary", width="stretch", key="sim_criar"):
            semente = int(st.session_state.get("sim_semente", 1000)) + 1
            novo = cliente_novo(semente)
            st.session_state["sim_caso"] = novo
            st.session_state["sim_semente"] = semente
            for coluna in FEATURES:
                st.session_state[f"sim_{coluna}"] = _alinhar(coluna, novo[coluna])
            st.rerun()
        caso = st.session_state.get("sim_caso")
        if caso:
            p_caso = chance(modelos[slug], caso)
            previsto = int(p_caso >= CORTE_MATRIZ)
            real = int(caso["inadimplente_90d"])
            acertou = previsto == real
            faixa_c, acao_c = nome_faixa(p_caso)
            situacao = "inadimplente" if real == 1 else "em dia"
            leitura = "Acertou" if acertou else "Errou"
            cor_l = "#3dd68c" if acertou else "#ff8a65"
            st.markdown(
                f"""
                <div class="diag">
                    <div class="diag-title">{caso['id_cliente']}</div>
                    Chance {_pct(p_caso)} · faixa {faixa_c} · {acao_c}.<br>
                    Alvo sintético: <strong>{situacao}</strong>.
                    No corte 0,50 o modelo <strong style="color:{cor_l}">{leitura.lower()}</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("O alvo deste cliente foi sorteado no mesmo processo da base. Num cliente real esse futuro ainda não existe.")
        else:
            st.caption("Ainda sem cliente novo. O botão sorteia um e compara a previsão com o alvo gerado.")
        st.divider()
        st.markdown("**Blocos**")
        _controles(perfil)
    return slug


def render_relato(ir) -> None:
    if not pronto():
        st.error("O dossiê ainda não foi gerado. Na raiz do repositório, rode python -m src.credito.pipeline.")
        st.stop()
    relatorio = ler_relatorio()
    base = relatorio["base"]
    _simulador(relatorio, ir)

    st.markdown(
        """
        <style>
            .sintetico {
                border: 1px solid rgba(232,185,35,0.45);
                background: rgba(232,185,35,0.10);
                color: #F4F1E8;
                border-radius: 12px;
                padding: 0.75rem 0.95rem;
                margin-bottom: 0.85rem;
                line-height: 1.45;
            }
            .sintetico strong { color: #E8B923; }
        </style>
        <div class="sintetico">
            <strong>Base sintética corporativa.</strong>
            50.000 clientes, 38 colunas, processo aberto. Não é carteira de banco.
            Os números desta página foram calculados no teste desta base.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="credit-top">Desenvolvido por Hart Botelho</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Base sintética · 50.000 clientes · 38 colunas</div>
            <h1>Comitê de Crédito</h1>
            <p>
                Estima se o cliente, nas condições em que se apresenta, ficaria
                inadimplente. O evento é regulatório: mais de 90 dias de atraso
                nos 12 meses seguintes. O relato se lê de cima para baixo.
                O simulador, na barra ao lado, responde na hora.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### A pergunta do comitê")
    st.markdown(
        """
        <div class="treat">
            <h3>Da cobrança para o alerta</h3>
            <p>
                O dossiê não espera o atraso aparecer. Ele estima a chance e devolve uma de três
                saídas: aprovar, condicionar garantia ou recusar. O corte 0,50 compara os modelos.
                As faixas verde, amarela e vermelha são a proposta de política desta tela.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### O livro sintético")
    _kpis_base(base)
    st.caption(
        f"Treino {_mil(base['n_treino'])} · teste {_mil(base['n_teste'])} · sorteio estratificado · semente {relatorio['seed']}. "
        "O identificador do cliente fica fora do treino."
    )
    # The replace in _kpis_base already turned thousands separators into dots. Caption uses python format with comma.
    # Fix caption to Brazilian thousands without being confusing. I'll format explicitly.
    blocos = []
    for chave, colunas in BLOCOS.items():
        blocos.append(
            f'<div class="cell"><div class="k">Bloco</div>'
            f'<div class="n" style="font-size:1rem">{BLOCO_NOME[chave]}</div>'
            f'<div class="s">{len(colunas)} colunas</div></div>'
        )
    blocos.append(
        '<div class="cell"><div class="k">Alvo</div>'
        '<div class="n" style="font-size:1rem">inadimplente_90d</div>'
        '<div class="s">Mais de 90 dias nos 12 meses seguintes</div></div>'
    )
    st.markdown(
        f"""
        <div class="treat">
            <h3>Oito blocos, 38 colunas</h3>
            <p>
                Sete blocos entram no modelo. O oitavo é o alvo. Renda, fluxo, saldo, limite e fatura
                nascem juntos. Um fator escondido organiza essas correlações e não entra na conta do alvo:
                o calote nasce dos campos visíveis, com ruído.
            </p>
            <div class="treat-grid">{''.join(blocos)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Idade mediana {base['idade_mediana']:.0f} anos. "
        f"Renda mediana {_reais(base['renda_mediana'])}. "
        f"Fluxo mediano de entradas {_reais(base['fluxo_mediano'])}. "
        f"Score mediano de bureau {base['score_mediano']:.0f}."
    )

    st.markdown("### O que os quatro modelos verificaram")
    st.markdown(f'<div class="diag">{relatorio["textos"]["auc"]} {relatorio["textos"]["corte"]}</div>', unsafe_allow_html=True)
    _tabela(relatorio)
    maioria = 1.0 - base["taxa_teste"]
    st.caption(
        f"Precisão, recall, F1, acurácia, FP e FN estão no corte 0,50. ROC AUC não depende desse corte. "
        f"Prever sempre em dia acertaria {_pct(maioria)} deste teste, porque a inadimplência realizada é {_pct(base['taxa_teste'])}. "
        "A acurácia perto desse piso quase não separa os modelos. O recall no corte 0,50 deixa passar a maior parte dos calotes; as faixas abaixo é que antecipam o alerta."
    )
    _imagem(
        relatorio,
        "metricas",
        "Figura gravada pelo script: os cinco índices no mesmo eixo, de 0 a 1.",
    )
    _imagem(
        relatorio,
        "matrizes",
        "Matriz de cada modelo no teste. Verde mais claro é a célula com mais clientes.",
    )

    st.markdown("### Curvas ROC")
    _imagem(
        relatorio,
        "roc",
        "Curvas no teste. A diagonal é o chute. O número da legenda é o ROC AUC medido.",
    )

    st.markdown("### Faixas de alerta")
    nome_ref = next(m["nome"] for m in relatorio["modelos"] if m["slug"] == relatorio["referencia"])
    st.markdown(
        f"""
        <div class="treat">
            <h3>Proposta desta tela · {nome_ref}</h3>
            <p>
                Verde até {_pct(relatorio['verde_max'])}: aprovar.
                Amarelo até {_pct(relatorio['amarelo_max'])}: condicionar garantia.
                Vermelho acima disso: recusar.
                Hart ainda crava estes cortes. A taxa em cada faixa, abaixo, é a que o teste realizou.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    celulas = []
    for info in relatorio["faixas"]:
        celulas.append(
            f'<div class="cell"><div class="k">{info["faixa"]}</div>'
            f'<div class="n" style="font-size:1.15rem">{info["acao"]}</div>'
            f'<div class="s">{_mil(info["n"])} clientes · taxa realizada {_pct(info["taxa_observada"])}</div></div>'
        )
    st.markdown(f'<div class="treat"><div class="treat-grid">{"".join(celulas)}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="diag">{relatorio["textos"]["faixas"]}</div>', unsafe_allow_html=True)
    _imagem(
        relatorio,
        "faixas",
        f"Distribuição da chance e taxa realizada por faixa. Modelo de referência: {nome_ref}.",
    )

    st.markdown("### Peso dos blocos")
    st.markdown(f'<div class="diag">{relatorio["textos"]["blocos"]}</div>', unsafe_allow_html=True)
    st.caption(
        f"Permutação no teste: {relatorio['importancia']['amostra']} clientes, "
        f"{relatorio['importancia']['repeticoes']} repetições, queda de ROC AUC. "
        "A fatia usa só a queda positiva. O sinal plantado é a soma dos módulos dos coeficientes do gerador, não uma medida de mercado."
    )
    _imagem(
        relatorio,
        "blocos",
        "Fatia de cada bloco nos quatro modelos. A ordem das linhas é a do modelo de referência.",
    )

    st.markdown("### O que justifica recusa ou garantia")
    st.markdown(
        """
        <div class="treat">
            <p>
                A logística é o scorecard. Coral empurra o dossiê para o calote; verde afasta.
                Números: efeito de um desvio-padrão. Categorias: efeito contra a média da própria variável.
                É com este peso que o comitê escreve o motivo de uma recusa ou de um pedido de garantia.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _imagem(
        relatorio,
        "explicacao",
        "Doze maiores pesos da regressão logística, no treino padronizado.",
    )

    st.markdown("### Visão de cada algoritmo")
    st.caption("A mesma prova, modelo a modelo: técnica, matriz, distribuição da chance e índices do teste.")
    for modelo in relatorio["modelos"]:
        m = modelo["matriz"]
        st.markdown(f"#### {modelo['nome']}")
        st.markdown(
            f"""
            <div class="treat">
                <h3>{modelo['papel']}</h3>
                <p>{modelo['tecnica']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        extra = ""
        if modelo["slug"] == "mlp" and relatorio.get("mlp_epocas"):
            extra = f" A rede parou em {relatorio['mlp_epocas']} épocas."
        st.markdown(
            f"""
            <div class="mkpi-grid">
                <div class="kpi" style="--accent:#6ea8fe"><div class="lbl">ROC AUC</div><div class="val">{_num(modelo['roc_auc'])}</div><div class="sub">Teste</div></div>
                <div class="kpi" style="--accent:#E8B923"><div class="lbl">F1</div><div class="val">{_pct(m['f1'])}</div><div class="sub">Corte 0,50</div></div>
                <div class="kpi" style="--accent:#3dd68c"><div class="lbl">Precisão</div><div class="val">{_pct(m['precisao'])}</div><div class="sub">{m['fp']} alarmes a mais</div></div>
                <div class="kpi" style="--accent:#c084fc"><div class="lbl">Recall</div><div class="val">{_pct(m['recall'])}</div><div class="sub">{m['fn']} calotes perdidos</div></div>
                <div class="kpi" style="--accent:#6ea8fe"><div class="lbl">Acurácia</div><div class="val">{_pct(m['acuracia'])}</div><div class="sub">Acertos no teste</div></div>
                <div class="kpi" style="--accent:#ff8a65"><div class="lbl">Brier</div><div class="val">{_num(modelo['brier'])}</div><div class="sub">Quanto menor, mais calibrada a chance</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if extra:
            st.caption(extra.strip())
        _imagem(
            relatorio,
            f"visao_{modelo['slug']}",
            f"{modelo['nome']}: matriz no corte 0,50 e chance prevista separada pelo que aconteceu no teste.",
        )

    st.markdown(
        """
        <div class="diag">
            <div class="diag-title">O que esta página deixa com o Hart</div>
            A base é sintética e o processo está no código. Falta a carteira real, a política de faixas,
            o modelo que o comitê assina e o corte no tempo. O simulador ao lado usa os modelos gravados aqui.
        </div>
        """,
        unsafe_allow_html=True,
    )
