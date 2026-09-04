"""Geração do relatório completo em PDF."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF

from src.data import FEATURE_LABELS
from src.diagnostics import diagnostic_sections, method_narrative
from src.modeling import EvaluationResult

FONT_REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

AUTHOR = "Hart Botelho"
NAVY = (11, 16, 32)
GOLD = (184, 140, 20)
INK = (28, 32, 44)
MUTED = (90, 90, 98)


def _plain(text: str) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.replace("`", "")


def _pct(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    return f"{value:.1f}%"


def _auc(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.3f}"


class PortalPDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 14, "F")
        self.set_text_color(244, 241, 232)
        self.set_font("DejaVu", "", 8)
        self.set_xy(self.l_margin, 4)
        self.cell(self.epw, 6, "Portal Interativo de Machine Learning  |  Pokémon Lendários")
        self.set_xy(self.l_margin, 16)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_x(self.l_margin)
        self.set_text_color(*MUTED)
        self.set_font("DejaVu", "", 8)
        self.cell(self.epw, 8, f"Página {self.page_no()}  |  {AUTHOR}", align="C")


def _reset(pdf: PortalPDF) -> None:
    pdf.set_x(pdf.l_margin)


def _section(pdf: PortalPDF, title: str) -> None:
    pdf.ln(4)
    _reset(pdf)
    pdf.set_text_color(*GOLD)
    pdf.set_font("DejaVu", "B", 12)
    pdf.multi_cell(pdf.epw, 7, title)
    pdf.set_draw_color(*GOLD)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
    pdf.ln(3)
    _reset(pdf)
    pdf.set_text_color(*INK)


def _para(pdf: PortalPDF, text: str, size: int = 10) -> None:
    _reset(pdf)
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw, 5.2, _plain(text))
    pdf.ln(1)
    _reset(pdf)


def _line(pdf: PortalPDF, label: str, value: str) -> None:
    _reset(pdf)
    if pdf.get_y() > pdf.h - 24:
        pdf.add_page()
        _reset(pdf)
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(pdf.epw, 5, label)
    _reset(pdf)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw, 5.2, _plain(value))
    pdf.ln(1)
    _reset(pdf)


def _norm_widths(pdf: PortalPDF, n: int, widths: list[float] | None) -> list[float]:
    epw = pdf.epw
    if not widths:
        w = [epw / n] * n
    else:
        total = sum(widths) or 1.0
        w = [x / total * epw for x in widths]
    w[-1] = epw - sum(w[:-1])
    return w


def _table(pdf: PortalPDF, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    w = _norm_widths(pdf, len(headers), widths)
    _reset(pdf)

    def header_row() -> None:
        _reset(pdf)
        pdf.set_font("DejaVu", "B", 7)
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(244, 241, 232)
        for i, h in enumerate(headers):
            pdf.cell(w[i], 7, _plain(h)[:28], fill=True)
        pdf.ln()
        _reset(pdf)
        pdf.set_text_color(*INK)
        pdf.set_font("DejaVu", "", 7)

    header_row()
    fill = False
    for row in rows:
        if pdf.get_y() > pdf.h - 24:
            pdf.add_page()
            header_row()
        pdf.set_fill_color(245, 242, 232) if fill else pdf.set_fill_color(255, 255, 255)
        _reset(pdf)
        for i, cell in enumerate(row):
            pdf.cell(w[i], 6.2, _plain(cell)[:36], fill=True)
        pdf.ln()
        fill = not fill
    pdf.ln(2)
    _reset(pdf)


def build_pdf_report(
    *,
    meta: dict,
    train_pct: int,
    threshold: float,
    features: list[str],
    preset_label: str,
    n_test: int,
    real: int,
    focus: EvaluationResult,
    results: list[EvaluationResult],
    profile: pd.DataFrame,
    ablation: pd.DataFrame | None,
) -> bytes:
    pdf = PortalPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 20, 16)
    pdf.add_font("DejaVu", "", str(FONT_REG))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.add_page()
    _reset(pdf)

    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(*GOLD)
    pdf.multi_cell(pdf.epw, 8, f"Desenvolvido por {AUTHOR}")
    pdf.ln(1)
    _reset(pdf)
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(pdf.epw, 8, "Relatório completo do simulador de acerto")
    _reset(pdf)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*MUTED)
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.multi_cell(pdf.epw, 6, f"Gerado em {agora}  |  alvo: is_legendary  |  seed 42")
    pdf.ln(2)

    _section(pdf, "1. Configuração do experimento")
    attrs = ", ".join(FEATURE_LABELS.get(f, f) for f in features)
    _line(pdf, "Preset de atributos", preset_label)
    _line(pdf, "Colunas no treino", attrs)
    _line(pdf, "Proporção treino / teste", f"{train_pct}% / {100 - train_pct}% (stratify=y)")
    _line(pdf, "Limiar de decisão", f"{threshold:.2f}")
    _line(pdf, "Método em destaque", focus.modelo)
    _line(pdf, "Algoritmos avaliados", ", ".join(r.modelo for r in results))

    _section(pdf, "2. Como a base foi preparada")
    _para(
        pdf,
        f"O CSV original tem {meta['rows_raw']} Pokémon. O portal remove registros com "
        f"valor ausente nas colunas de modelagem (dropna). Foram removidos {meta['dropped']} "
        f"Pokémon. A base limpa fica com {meta['rows_clean']} registros, dos quais "
        f"{meta['legendaries']} são lendários. Treino e teste são sorteados a partir dessa "
        f"base limpa.",
    )

    _section(pdf, "3. Indicadores da amostra de teste")
    _line(pdf, "Total de Pokémon no teste", str(n_test))
    _line(pdf, "Lendários reais (gabarito)", str(real))
    _line(pdf, f"Total predito ({focus.modelo})", str(focus.total_predito))
    _line(pdf, "Desvio relativo do real", f"{focus.margem_pct:+.1f}%")

    _section(pdf, "4. Comparativo de desempenho por método")
    rows = []
    for r in results:
        rows.append(
            [
                r.modelo,
                str(r.total_predito),
                str(r.tp),
                str(r.fp),
                str(r.fn),
                _pct(r.precisao * 100),
                _pct(r.captura_pct),
                _pct(r.f1 * 100),
                f"{r.margem_pct:+.1f}%",
            ]
        )
    _table(
        pdf,
        ["Método", "Predito", "TP", "FP", "FN", "Precisão", "Captura", "F1", "Margem"],
        rows,
        [24, 12, 10, 10, 10, 14, 14, 12, 16],
    )

    _section(pdf, "5. Relatório de cada método")
    for r in results:
        _reset(pdf)
        pdf.set_font("DejaVu", "B", 11)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(pdf.epw, 7, r.modelo)
        auc = _auc(r.roc_auc)
        _para(
            pdf,
            f"Predito {r.total_predito} (margem {r.margem_pct:+.1f}%). "
            f"TP {r.tp} · FP {r.fp} · FN {r.fn} · TN {r.tn}. "
            f"Acurácia {_pct(r.acuracia * 100)} · precisão {_pct(r.precisao * 100)} · "
            f"captura {_pct(r.captura_pct)} · F1 {_pct(r.f1 * 100)} · "
            f"especificidade {_pct(r.especificidade * 100)} · ROC AUC {auc}.",
        )
        _para(pdf, method_narrative(r, real), size=9)

    _section(pdf, "6. Impacto de cada variável")
    if profile is not None and not profile.empty:
        prof_rows = []
        for _, row in profile.iterrows():
            feat = row["atributo"]
            prof_rows.append(
                [
                    FEATURE_LABELS.get(feat, feat),
                    f"{row['media_lendarios']:.2f}",
                    f"{row['media_comuns']:.2f}",
                    f"{row['diferenca']:+.2f}",
                ]
            )
        _table(
            pdf,
            ["Variável", "Média lendários", "Média comuns", "Diferença"],
            prof_rows,
            [40, 35, 35, 30],
        )
    if focus.importancias:
        _para(pdf, f"Peso no método {focus.modelo}: {focus.origem_importancia}.")
        imp_rows = []
        for feat, score in sorted(focus.importancias.items(), key=lambda kv: kv[1], reverse=True):
            direcao = ""
            if feat in focus.direcao:
                sinal = focus.direcao[feat]
                direcao = "aumenta lendário" if sinal > 0 else "diminui lendário" if sinal < 0 else "neutro"
            imp_rows.append([FEATURE_LABELS.get(feat, feat), f"{score:.4f}", direcao])
        _table(pdf, ["Variável", "Importância", "Direção"], imp_rows, [50, 30, 50])
    if ablation is not None and not ablation.empty:
        _para(pdf, f"Simulação: o que acontece no {focus.modelo} se a variável for removida.")
        ab_rows = []
        for _, row in ablation.iterrows():
            feat = row["atributo"]
            ab_rows.append(
                [
                    FEATURE_LABELS.get(feat, feat),
                    f"{row['f1_sem']:.1f}%",
                    f"{row['delta_f1']:+.1f}",
                    str(int(row["predito_sem"])),
                    f"{int(row['delta_predito']):+d}",
                ]
            )
        _table(
            pdf,
            ["Variável", "F1 sem ela", "dF1 (pp)", "Predito sem ela", "d contagem"],
            ab_rows,
            [36, 24, 22, 30, 28],
        )

    _section(pdf, "7. Diagnóstico automático e dicas")
    for title, body in diagnostic_sections(results, real, features, n_test, threshold):
        _reset(pdf)
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(*GOLD)
        pdf.multi_cell(pdf.epw, 6, _plain(title))
        _para(pdf, body, size=9)

    _reset(pdf)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(pdf.epw, 5, f"Fim do relatório. Desenvolvido por {AUTHOR}.")
    return bytes(pdf.output())
