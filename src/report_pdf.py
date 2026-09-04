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

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REG = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

AUTHOR = "Hart Botelho"
NAVY = (11, 16, 32)
GOLD = (184, 140, 20)
INK = (28, 32, 44)
MUTED = (90, 90, 98)


def _plain(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.replace("`", "")


def _pct(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.1f}%"


def _auc(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}"


class PortalPDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 18, "F")
        self.set_text_color(244, 241, 232)
        self.set_font("DejaVu", "B", 10)
        self.set_xy(12, 6)
        self.cell(0, 6, "Portal Interativo de Machine Learning  ·  Pokémon Lendários", align="L")
        self.set_xy(12, 6)
        self.set_font("DejaVu", "", 9)
        self.cell(0, 6, f"Desenvolvido por {AUTHOR}", align="R")
        self.ln(16)
        self.set_text_color(*INK)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_text_color(*MUTED)
        self.set_font("DejaVu", "", 8)
        self.cell(
            0,
            8,
            f"Página {self.page_no()}  ·  Relatório do simulador de acerto  ·  {AUTHOR}",
            align="C",
        )


def _section(pdf: PortalPDF, title: str) -> None:
    pdf.ln(3)
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*GOLD)
    pdf.set_font("DejaVu", "B", 13)
    pdf.multi_cell(0, 8, title)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.4)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)
    pdf.set_text_color(*INK)
    pdf.set_font("DejaVu", "", 10)


def _para(pdf: PortalPDF, text: str, size: int = 10) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.2, _plain(text))
    pdf.ln(1)


def _kv(pdf: PortalPDF, rows: list[tuple[str, str]]) -> None:
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    label_w = usable * 0.34
    value_w = usable * 0.66
    for label, value in rows:
        pdf.set_x(pdf.l_margin)
        y0 = pdf.get_y()
        if y0 > pdf.h - 28:
            pdf.add_page()
            y0 = pdf.get_y()
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(label_w, 6, label)
        y1 = pdf.get_y()
        pdf.set_xy(pdf.l_margin + label_w, y0)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(*INK)
        pdf.multi_cell(value_w, 6, value)
        pdf.set_y(max(y1, pdf.get_y()))
    pdf.ln(1)


def _table(pdf: PortalPDF, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    if widths is None:
        widths = [usable / len(headers)] * len(headers)
    pdf.set_font("DejaVu", "B", 7.5)
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(244, 241, 232)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, border=0, fill=True)
    pdf.ln()
    pdf.set_text_color(*INK)
    pdf.set_font("DejaVu", "", 7.5)
    fill = False
    for row in rows:
        pdf.set_x(pdf.l_margin)
        if pdf.get_y() > pdf.h - 28:
            pdf.add_page()
            pdf.set_x(pdf.l_margin)
            pdf.set_font("DejaVu", "B", 7.5)
            pdf.set_fill_color(*NAVY)
            pdf.set_text_color(244, 241, 232)
            for i, h in enumerate(headers):
                pdf.cell(widths[i], 7, h, border=0, fill=True)
            pdf.ln()
            pdf.set_x(pdf.l_margin)
            pdf.set_text_color(*INK)
            pdf.set_font("DejaVu", "", 7.5)
        pdf.set_fill_color(245, 242, 232) if fill else pdf.set_fill_color(255, 255, 255)
        for i, cell in enumerate(row):
            pdf.cell(widths[i], 6.4, str(cell)[:42], border=0, fill=True)
        pdf.ln()
        fill = not fill
    pdf.set_x(pdf.l_margin)
    pdf.ln(2)


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
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(14, 22, 14)
    pdf.add_font("DejaVu", "", str(FONT_REG))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.add_page()

    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, "Relatório completo do simulador de acerto")
    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(*GOLD)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, f"Desenvolvido por {AUTHOR}")
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*MUTED)
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, f"Gerado em {agora}  ·  alvo: is_legendary  ·  seed 42")
    pdf.ln(2)

    _section(pdf, "1. Configuração do experimento")
    attrs = ", ".join(FEATURE_LABELS.get(f, f) for f in features)
    _kv(
        pdf,
        [
            ("Preset de atributos", preset_label),
            ("Colunas no treino", attrs),
            ("Proporção treino / teste", f"{train_pct}% / {100 - train_pct}%  (stratify=y)"),
            ("Limiar de decisão", f"{threshold:.2f}"),
            ("Método em destaque", focus.modelo),
            ("Algoritmos avaliados", ", ".join(r.modelo for r in results)),
        ],
    )

    _section(pdf, "2. Como a base foi preparada")
    _para(
        pdf,
        f"O CSV original tem {meta['rows_raw']} Pokémon. O portal remove registros com "
        f"valor ausente nas colunas de modelagem (dropna). Foram removidos {meta['dropped']} "
        f"Pokémon. A base limpa fica com {meta['rows_clean']} registros, dos quais "
        f"{meta['legendaries']} são lendários. Treino e teste são sorteados a partir dessa "
        f"base limpa — não de uma amostra fixa de 12 nomes.",
    )

    _section(pdf, "3. Indicadores da amostra de teste")
    _kv(
        pdf,
        [
            ("Total de Pokémon no teste", str(n_test)),
            ("Lendários reais (gabarito)", str(real)),
            (f"Total predito ({focus.modelo})", str(focus.total_predito)),
            ("Desvio relativo do real", f"{focus.margem_pct:+.1f}%"),
        ],
    )

    _section(pdf, "4. Comparativo de desempenho por método")
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    widths = [usable * w for w in (0.22, 0.10, 0.08, 0.08, 0.08, 0.11, 0.11, 0.10, 0.12)]
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
        widths,
    )

    _section(pdf, "5. Relatório de cada método")
    for r in results:
        pdf.set_font("DejaVu", "B", 11)
        pdf.set_text_color(*NAVY)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, r.modelo)
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
        pdf.ln(1)

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
        _table(pdf, ["Variável", "Importância", "Direção"], imp_rows)
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
        _table(pdf, ["Variável", "F1 sem ela", "Δ F1 (pp)", "Predito sem ela", "Δ contagem"], ab_rows)

    _section(pdf, "7. Diagnóstico automático e dicas")
    for title, body in diagnostic_sections(results, real, features, n_test, threshold):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(*GOLD)
        pdf.multi_cell(0, 6, title)
        _para(pdf, body, size=9)

    pdf.ln(4)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        0,
        5,
        f"Fim do relatório. Simulador de acerto para is_legendary. Desenvolvido por {AUTHOR}.",
    )
    return bytes(pdf.output())
