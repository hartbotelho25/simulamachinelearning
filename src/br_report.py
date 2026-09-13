"""PDF do experimento do Brasileirão."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src.br_data import FEATURE_LABELS, TARGET_LABELS
from src.br_diagnostics import METHOD_TIPS

_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
FONT_REG = _FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = _FONT_DIR / "DejaVuSans-Bold.ttf"
if not FONT_REG.exists():
    FONT_REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
AUTHOR = "Hart Botelho"
HEADER_H = 38

PANORAMA_ROWS = [
    ("Total de Clubes Analisados", "rows_clean", None),
    ("Já foram rebaixados (1+ vez)", "n_rebaixados", "rebaixado"),
    ("Caíram 2 vezes ou mais (2+)", "n_recorrentes", "recorrente"),
    ("Já tiveram artilheiro da Série A", "n_artilheiros", "artilheiro"),
    ("Multicampeões (2+ títulos)", "n_multicampeoes", "multicampeao"),
]


def _plain(text: str) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return (
        text.replace("`", "")
        .replace("—", "-")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _clubs(xs: list[str]) -> str:
    if not xs:
        return "Nenhum"
    return ", ".join(str(x) for x in xs)


def _auc(r) -> float:
    return r.roc_auc if r.roc_auc == r.roc_auc else -1.0


class PDF(FPDF):
    report_title = "Relatório - Simulador ML do Brasileirão"
    report_target = ""
    report_when = ""

    def header(self):
        self.set_fill_color(11, 28, 20)
        self.rect(0, 0, self.w, HEADER_H, "F")
        self.set_text_color(232, 185, 35)
        self.set_font("DejaVu", "B", 12)
        self.set_xy(self.l_margin, 5)
        self.cell(self.epw, 6, _plain(self.report_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(244, 241, 232)
        self.set_x(self.l_margin)
        self.cell(self.epw, 5, f"Alvo selecionado: {_plain(self.report_target)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin)
        self.cell(self.epw, 5, f"Gerado em {self.report_when}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin)
        self.cell(self.epw, 5, f"Desenvolvido por {AUTHOR}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(HEADER_H + 8)

    def footer(self):
        self.set_y(-12)
        self.set_x(self.l_margin)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(90, 90, 98)
        self.cell(
            self.epw,
            8,
            f"Página {self.page_no()}  |  {AUTHOR}",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )


def _reset(pdf: PDF) -> None:
    pdf.set_x(pdf.l_margin)


def _h(pdf: PDF, title: str) -> None:
    pdf.ln(4)
    _reset(pdf)
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(184, 140, 20)
    pdf.multi_cell(pdf.epw, 7, _plain(title))
    pdf.ln(1)
    _reset(pdf)


def _p(pdf: PDF, text: str, size: int = 10) -> None:
    _reset(pdf)
    pdf.set_font("DejaVu", "", size)
    pdf.set_text_color(28, 32, 44)
    pdf.multi_cell(pdf.epw, 5.4, _plain(text))
    pdf.ln(1)
    _reset(pdf)


def _fit_cell(pdf: PDF, w: float, h: float, text: str, *, fill: bool, bold: bool, nxt, nyt) -> None:
    raw = _plain(text)
    pdf.set_font("DejaVu", "B" if bold else "", 7)
    size = 7
    while size >= 5.5 and pdf.get_string_width(raw) > w - 1.2:
        size -= 0.4
        pdf.set_font("DejaVu", "B" if bold else "", size)
    pdf.cell(w, h, raw, fill=fill, new_x=nxt, new_y=nyt)


def _table(
    pdf: PDF,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
    highlight: set[int] | None = None,
) -> None:
    n = len(headers)
    epw = pdf.epw
    if not widths:
        w = [epw / n] * n
    else:
        s = sum(widths) or 1
        w = [x / s * epw for x in widths]
    w[-1] = epw - sum(w[:-1])
    highlight = highlight or set()

    def hdr():
        _reset(pdf)
        pdf.set_fill_color(11, 28, 20)
        pdf.set_text_color(244, 241, 232)
        for i, h in enumerate(headers):
            nxt = XPos.RIGHT if i < n - 1 else XPos.LMARGIN
            nyt = YPos.TOP if i < n - 1 else YPos.NEXT
            _fit_cell(pdf, w[i], 8, h, fill=True, bold=True, nxt=nxt, nyt=nyt)
        _reset(pdf)
        pdf.set_text_color(28, 32, 44)

    hdr()
    for idx, row in enumerate(rows):
        if pdf.get_y() > pdf.h - 26:
            pdf.add_page()
            hdr()
        top = idx in highlight
        if top:
            pdf.set_fill_color(226, 226, 226)
        elif idx % 2:
            pdf.set_fill_color(245, 242, 232)
        else:
            pdf.set_fill_color(255, 255, 255)
        _reset(pdf)
        for i, c in enumerate(row):
            nxt = XPos.RIGHT if i < n - 1 else XPos.LMARGIN
            nyt = YPos.TOP if i < n - 1 else YPos.NEXT
            _fit_cell(pdf, w[i], 7, c, fill=True, bold=top, nxt=nxt, nyt=nyt)
    pdf.ln(3)
    _reset(pdf)


def _box(pdf: PDF, text: str) -> None:
    _reset(pdf)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(28, 32, 44)
    body = _plain(text)
    h = pdf.get_string_width(body) / max(pdf.epw - 6, 1) * 5.2 + 12
    if pdf.get_y() + max(h, 22) > pdf.h - 20:
        pdf.add_page()
    x, y = pdf.l_margin, pdf.get_y()
    pdf.set_fill_color(245, 242, 232)
    pdf.set_draw_color(184, 140, 20)
    pdf.rect(x, y, pdf.epw, 1, "F")
    pdf.set_xy(x + 3, y + 3)
    pdf.multi_cell(pdf.epw - 6, 5.2, body)
    bottom = pdf.get_y() + 3
    pdf.set_draw_color(210, 205, 190)
    pdf.rect(x, y, pdf.epw, bottom - y)
    pdf.set_y(bottom + 2)
    _reset(pdf)


def build_pdf_report(*, meta, target_key, train_pct, threshold, features, preset_label,
                     n_test, real, results) -> bytes:
    when = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf = PDF()
    pdf.report_title = "Relatório - Simulador ML do Brasileirão"
    pdf.report_target = TARGET_LABELS[target_key]
    pdf.report_when = when
    pdf.set_auto_page_break(True, 18)
    pdf.set_margins(14, HEADER_H + 8, 14)
    pdf.add_font("DejaVu", "", str(FONT_REG))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD))
    pdf.add_page()

    _h(pdf, "1. Configuração")
    _p(pdf, f"Preset: {preset_label}. Treino/teste {train_pct}/{100 - train_pct}% · limiar {threshold:.2f}.")
    _p(pdf, "Colunas: " + ", ".join(FEATURE_LABELS.get(f, f) for f in features))

    _h(pdf, "2. Panorama da Base de Dados")
    panorama = []
    for label, key, tgt in PANORAMA_ROWS:
        qty = meta.get(key, "—")
        name = label + (" (Alvo Atual)" if tgt == target_key else "")
        panorama.append([name, str(qty)])
    _table(pdf, ["Indicador", "Quantidade"], panorama, widths=[5.2, 1.3])

    _h(pdf, "3. Indicadores do teste")
    _p(pdf, f"Teste {n_test} clubes · SIM reais {real}.")

    _h(pdf, "4. Comparativo")
    ranked = sorted(results, key=lambda r: (_auc(r), r.f1, r.acuracia), reverse=True)
    best_auc = max((_auc(r) for r in ranked), default=-1)
    best_f1 = max((r.f1 for r in ranked), default=-1)
    highlight = {
        i
        for i, r in enumerate(ranked)
        if abs(_auc(r) - best_auc) < 1e-9 or abs(r.f1 - best_f1) < 1e-9
    }
    rows = [
        [
            r.modelo,
            "—" if _auc(r) < 0 else f"{r.roc_auc:.3f}",
            f"{r.f1 * 100:.1f}",
            f"{r.acuracia * 100:.1f}",
            f"{r.precisao * 100:.1f}",
            f"{r.recall * 100:.1f}",
            str(r.fp),
            str(r.fn),
        ]
        for r in ranked
    ]
    _table(
        pdf,
        ["Modelo", "AUC", "F1 (%)", "Acur (%)", "Prec (%)", "Rec (%)", "FP (alarme)", "FN (perdidos)"],
        rows,
        widths=[2.55, 0.85, 0.9, 1.05, 1.05, 0.95, 1.45, 1.55],
        highlight=highlight,
    )

    _h(pdf, "5. Cada método")
    for r in results:
        if pdf.get_y() > pdf.h - 78:
            pdf.add_page()
        _reset(pdf)
        pdf.set_fill_color(11, 28, 20)
        pdf.set_text_color(244, 241, 232)
        pdf.set_font("DejaVu", "B", 10)
        auc = "—" if _auc(r) < 0 else f"{r.roc_auc:.3f}"
        kpi = (
            f"{r.modelo}  |  F1: {r.f1 * 100:.1f}%  |  AUC: {auc}  |  "
            f"Previu: {r.total_predito} SIM (Gabarito: {real})"
        )
        pdf.multi_cell(pdf.epw, 7, _plain(kpi), fill=True)
        pdf.ln(3)
        _reset(pdf)
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(11, 28, 20)
        pdf.multi_cell(pdf.epw, 5, "Classificação dos clubes")
        _reset(pdf)
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(28, 32, 44)
        pdf.multi_cell(pdf.epw, 5, f"- Acertos (TP): {_clubs(r.exemplos_tp)}")
        _reset(pdf)
        pdf.multi_cell(pdf.epw, 5, f"- Alarmes (FP): {_clubs(r.exemplos_fp)}")
        _reset(pdf)
        pdf.multi_cell(pdf.epw, 5, f"- Perdidos (FN): {_clubs(r.exemplos_fn)}")
        pdf.ln(2)
        _box(pdf, METHOD_TIPS.get(r.modelo, ""))
        pdf.ln(8)

    _p(pdf, f"Fim do relatório. Desenvolvido por {AUTHOR}.")
    return bytes(pdf.output())
