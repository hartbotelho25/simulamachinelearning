"""PDF do experimento do Brasileirão."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF

from src.br_data import FEATURE_LABELS, TARGET_LABELS
from src.br_diagnostics import diagnostic_sections, method_narrative
from src.br_modeling import EvaluationResult

FONT_REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
AUTHOR = "Hart Botelho"


def _plain(text: str) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.replace("`", "")


class PDF(FPDF):
    def header(self):
        self.set_fill_color(11, 28, 20)
        self.rect(0, 0, self.w, 14, "F")
        self.set_text_color(244, 241, 232)
        self.set_font("DejaVu", "", 8)
        self.set_xy(self.l_margin, 4)
        self.cell(self.epw, 6, "Simulador ML Brasileirão  |  CAIXA")
        self.set_xy(self.l_margin, 16)

    def footer(self):
        self.set_y(-12)
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(90, 90, 98)
        self.cell(self.epw, 8, f"Página {self.page_no()}  |  {AUTHOR}", align="C")


def _reset(pdf):
    pdf.set_x(pdf.l_margin)


def _h(pdf, title):
    pdf.ln(3)
    _reset(pdf)
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(184, 140, 20)
    pdf.multi_cell(pdf.epw, 7, title)
    _reset(pdf)


def _p(pdf, text, size=10):
    _reset(pdf)
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(28, 32, 44)
    pdf.multi_cell(pdf.epw, 5.2, _plain(text))
    pdf.ln(1)
    _reset(pdf)


def _table(pdf, headers, rows, widths=None):
    n = len(headers)
    epw = pdf.epw
    if not widths:
        w = [epw / n] * n
    else:
        s = sum(widths) or 1
        w = [x / s * epw for x in widths]
    w[-1] = epw - sum(w[:-1])

    def hdr():
        _reset(pdf)
        pdf.set_font("DejaVu", "B", 7)
        pdf.set_fill_color(11, 28, 20)
        pdf.set_text_color(244, 241, 232)
        for i, h in enumerate(headers):
            pdf.cell(w[i], 7, _plain(h)[:28], fill=True)
        pdf.ln()
        _reset(pdf)
        pdf.set_font("DejaVu", "", 7)
        pdf.set_text_color(28, 32, 44)

    hdr()
    fill = False
    for row in rows:
        if pdf.get_y() > pdf.h - 24:
            pdf.add_page()
            hdr()
        pdf.set_fill_color(245, 242, 232) if fill else pdf.set_fill_color(255, 255, 255)
        _reset(pdf)
        for i, c in enumerate(row):
            pdf.cell(w[i], 6.2, _plain(c)[:36], fill=True)
        pdf.ln()
        fill = not fill
    pdf.ln(2)
    _reset(pdf)


def build_pdf_report(*, meta, target_key, train_pct, threshold, features, preset_label,
                     n_test, real, focus: EvaluationResult, results, profile, ablation) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(True, 16)
    pdf.set_margins(16, 20, 16)
    pdf.add_font("DejaVu", "", str(FONT_REG))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.add_page()
    _reset(pdf)
    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(184, 140, 20)
    pdf.multi_cell(pdf.epw, 8, f"Desenvolvido por {AUTHOR}")
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(11, 28, 20)
    pdf.multi_cell(pdf.epw, 8, "Relatório — Simulador ML do Brasileirão")
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(90, 90, 98)
    pdf.multi_cell(pdf.epw, 6, f"{datetime.now():%d/%m/%Y %H:%M}  |  {TARGET_LABELS[target_key]}")
    pdf.ln(2)
    _h(pdf, "1. Configuração")
    _p(pdf, f"Alvo: {TARGET_LABELS[target_key]}. Preset: {preset_label}.")
    _p(pdf, "Colunas: " + ", ".join(FEATURE_LABELS.get(f, f) for f in features))
    _p(pdf, f"Treino/teste {train_pct}/{100-train_pct}% · limiar {threshold:.2f} · destaque {focus.modelo}")
    _h(pdf, "2. Base")
    _p(pdf, f"{meta['rows_clean']} clubes limpos. Campeões {meta['n_campeoes']}, "
            f"rebaixados {meta['n_rebaixados']}.")
    _h(pdf, "3. Indicadores do teste")
    _p(pdf, f"Teste {n_test} · SIM reais {real} · predito {focus.total_predito} · "
            f"desvio {focus.margem_pct:+.1f}%")
    _h(pdf, "4. Comparativo")
    rows = [[r.modelo, str(r.total_predito), str(r.tp), str(r.fp), str(r.fn),
             f"{r.precisao*100:.1f}", f"{r.f1*100:.1f}", f"{r.margem_pct:+.1f}"] for r in results]
    _table(pdf, ["Modelo", "Pred", "VP", "FP", "FN", "Prec", "F1", "Margem"], rows)
    _h(pdf, "5. Cada método")
    for r in results:
        pdf.set_font("DejaVu", "B", 11)
        pdf.multi_cell(pdf.epw, 7, r.modelo)
        _p(pdf, method_narrative(r, real, target_key), 9)
    _h(pdf, "6. Variáveis")
    if profile is not None and not profile.empty:
        pr = [
            [
                FEATURE_LABELS.get(row.atributo, row.atributo),
                f"{row.media_sim:.2f}",
                f"{row.media_nao:.2f}",
                f"{row.diferenca:+.2f}",
            ]
            for row in profile.itertuples()
        ]
        _table(pdf, ["Variável", "Média SIM", "Média NÃO", "Dif"], pr)
    if ablation is not None and not ablation.empty:
        ab = [[FEATURE_LABELS.get(a.atributo, a.atributo), f"{a.f1_sem:.1f}", f"{a.delta_f1:+.1f}",
               str(int(a.predito_sem))] for a in ablation.itertuples()]
        _table(pdf, ["Variável", "F1 sem", "dF1", "Pred sem"], ab)
    _h(pdf, "7. Diagnóstico")
    for title, body in diagnostic_sections(results, real, features, n_test, threshold, target_key):
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(184, 140, 20)
        pdf.multi_cell(pdf.epw, 6, _plain(title))
        _p(pdf, body, 9)
    _p(pdf, f"Fim do relatório. Desenvolvido por {AUTHOR}.")
    return bytes(pdf.output())
