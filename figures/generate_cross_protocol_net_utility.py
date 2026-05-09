#!/usr/bin/env python3
"""Generate a compact cross-protocol CASP net-utility figure as vector PDF."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import inch
from reportlab.pdfgen import canvas


OUT = "paper/figures/cross_protocol_net_utility.pdf"

ROWS = [
    {
        "protocol": "REES46",
        "role": "method anchor",
        "net100": 54,
        "ratio100": 0.443,
        "base": 6392,
        "casp": 6446,
        "color": colors.HexColor("#1b7837"),
    },
    {
        "protocol": "Tmall",
        "role": "stress replication",
        "net100": 1070,
        "ratio100": 0.039,
        "base": 25010,
        "casp": 26080,
        "color": colors.HexColor("#2166ac"),
    },
    {
        "protocol": "Synerise",
        "role": "confirmatory",
        "net100": 362,
        "ratio100": 0.0028,
        "base": 1630,
        "casp": 1992,
        "color": colors.HexColor("#8c510a"),
    },
]


TEXT = colors.HexColor("#222222")
MUTED = colors.HexColor("#666666")
GRID = colors.HexColor("#d9d9d9")
AXIS = colors.HexColor("#333333")
CAP = colors.HexColor("#2ca25f")


def draw_panel(
    c,
    x0,
    y0,
    w,
    title,
    subtitle,
    y_positions,
    values,
    x_max,
    ticks,
    tick_fmt,
    value_fmt,
    value_label=None,
    threshold=None,
):
    axis_y = y0
    top_y = max(y_positions) + 0.16 * inch

    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x0, top_y + 0.30 * inch, title)

    c.setStrokeColor(GRID)
    c.setLineWidth(0.4)
    for tick in ticks:
        xx = x0 + (tick / x_max) * w
        c.line(xx, axis_y, xx, top_y)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 5.8)
        c.drawCentredString(xx, axis_y - 0.11 * inch, tick_fmt(tick))

    if threshold is not None:
        tx = x0 + (threshold / x_max) * w
        c.setStrokeColor(CAP)
        c.setDash(2, 2)
        c.line(tx, axis_y, tx, top_y)
        c.setDash()
        c.setFillColor(CAP)
        c.setFont("Helvetica", 5.8)
        c.drawRightString(tx - 3, top_y + 0.04 * inch, f"cap {threshold:g}")

    c.setStrokeColor(AXIS)
    c.setLineWidth(0.6)
    c.line(x0, axis_y, x0 + w, axis_y)

    bar_h = 0.105 * inch
    for row, y in zip(ROWS, y_positions):
        val = values(row)
        bw = max(0.8, min(w, (val / x_max) * w))
        c.setFillColor(row["color"])
        c.rect(x0, y - bar_h / 2, bw, bar_h, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 6.2)
        label = value_label(row, val) if value_label else value_fmt(val)
        c.drawString(x0 + bw + 2.5, y - 0.030 * inch, label)


def draw_row_labels(c, x0, y_positions):
    for row, y in zip(ROWS, y_positions):
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 7.1)
        c.drawRightString(x0, y - 0.018 * inch, row["protocol"])


def main():
    width, height = 3.25 * inch, 3.20 * inch
    c = canvas.Canvas(OUT, pagesize=(width, height))
    c.setTitle("Cross-protocol CASP net utility")

    net_y = [2.48 * inch, 2.15 * inch, 1.82 * inch]
    ratio_y = [0.87 * inch, 0.54 * inch, 0.21 * inch]
    label_x = 0.58 * inch
    panel_x = 0.66 * inch
    panel_w = 2.15 * inch

    draw_row_labels(c, label_x, net_y)

    draw_panel(
        c,
        x0=panel_x,
        y0=1.60 * inch,
        w=panel_w,
        title="Net@100",
        subtitle="CASP - protected list",
        y_positions=net_y,
        x_max=1100,
        ticks=[0, 550, 1100],
        values=lambda r: r["net100"],
        tick_fmt=lambda v: f"{v:.0f}",
        value_fmt=lambda v: f"{v:+.0f}",
        value_label=lambda r, v: f"{v:+.0f} ({v / r['base'] * 100:.1f}%)",
    )
    draw_row_labels(c, label_x, ratio_y)
    draw_panel(
        c,
        x0=panel_x,
        y0=0.02 * inch,
        w=panel_w,
        title="Ratio@100",
        subtitle="cannibalized / gross",
        y_positions=ratio_y,
        x_max=0.5,
        ticks=[0, 0.25, 0.5],
        values=lambda r: r["ratio100"],
        tick_fmt=lambda v: f"{v:.2f}" if v else "0",
        value_fmt=lambda v: f"{v:.3f}" if v < 0.1 else f"{v:.2f}",
        threshold=0.5,
    )
    c.showPage()
    c.save()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
