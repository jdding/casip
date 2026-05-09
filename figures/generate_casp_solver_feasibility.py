#!/usr/bin/env python3
"""Generate the CASP solver feasibility scatter plot.

The plotting environment for this workspace does not include matplotlib, so the
figure is generated directly as vector PDF using reportlab.
"""

import csv

from reportlab.lib import colors
from reportlab.lib.pagesizes import inch
from reportlab.pdfgen import canvas


OUT = "paper/figures/casp_solver_feasibility.pdf"
DATA = "results/20260506_rees46_casp_solver_comparison/casp_solver_comparison.csv"


STYLE = {
    "Exact validation fusion": {
        "plot_name": "Exact fusion",
        "color": colors.HexColor("#8c2d04"),
        "shape": "square",
        "label_x": -372,
        "label_y": 2.22,
    },
    "Always-on promotion": {
        "plot_name": "Always-on",
        "color": colors.HexColor("#d95f02"),
        "shape": "circle",
        "label_side": "right",
    },
    "CASP P-A: confidence solver": {
        "plot_name": "CASP P-A",
        "color": colors.HexColor("#1b7837"),
        "shape": "circle",
        "label_side": "above_left",
    },
    "CASP-L2: compact logistic gate": {
        "plot_name": "CASP-L2",
        "color": colors.HexColor("#006d2c"),
        "shape": "circle",
        "label_side": "above_right",
    },
    "P-B: shallow tree": {
        "plot_name": "P-B tree",
        "color": colors.HexColor("#7570b3"),
        "shape": "triangle",
        "label_side": "above",
    },
    "P-B: interaction rules": {
        "plot_name": "P-B rules",
        "color": colors.HexColor("#7570b3"),
        "shape": "triangle",
        "label_side": "below",
    },
    "P-B: list-residual rules": {
        "plot_name": "P-B residual",
        "color": colors.HexColor("#4d4d4d"),
        "shape": "diamond",
        "label_side": "below",
    },
    "P-C: learned two-head HGB": {
        "plot_name": "HGB 2-head",
        "color": colors.HexColor("#2166ac"),
        "shape": "square",
        "label_side": "left",
    },
    "P-C: learned utility HGB": {
        "plot_name": "HGB utility",
        "color": colors.HexColor("#2166ac"),
        "shape": "square",
        "label_side": "right",
    },
    "P-C: LGBM action ranker": {
        "plot_name": "LGBM action",
        "color": colors.HexColor("#1f78b4"),
        "shape": "square",
        "label_side": "right",
    },
    "DLCM slate reranker": {
        "plot_name": "DLCM slate",
        "color": colors.HexColor("#6a3d9a"),
        "shape": "diamond",
        "label_x": -18,
        "label_y": 0.690,
    },
    "PRM slate reranker": {
        "plot_name": "PRM slate",
        "color": colors.HexColor("#984ea3"),
        "shape": "diamond",
        "label_x": -18,
        "label_y": 0.655,
    },
}


def load_points():
    points = []
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            if row["solver"] == "Existing source":
                continue
            style = STYLE[row["solver"]]
            x = float(row["net100"])
            # Keep the residual grid visible when it exactly returns CASP P-A.
            if row["solver"] == "P-B: list-residual rules":
                x += 4
            points.append({
                "name": style["plot_name"],
                "x": x,
                "y": float(row["ratio100"]),
                "color": style["color"],
                "shape": style["shape"],
                "label_side": style.get("label_side", "right"),
            })
    return points


def draw_adjacent_label(c, x, y, text, side, font_size=7):
    """Draw a label with a fixed small gap from its marker."""
    c.setFillColor(colors.black)
    c.setFont("Helvetica", font_size)
    if side == "left":
        c.drawRightString(x - 7, y - 2, text)
    elif side == "above_left":
        c.drawRightString(x - 3, y + 8, text)
    elif side == "above_right":
        c.drawString(x + 3, y + 8, text)
    elif side == "above":
        c.drawCentredString(x, y + 8, text)
    elif side == "below":
        c.drawCentredString(x, y - 13, text)
    else:
        c.drawString(x + 7, y - 2, text)


def main():
    points = load_points()
    width, height = 6.5 * inch, 3.05 * inch
    margin_l, margin_r = 0.62 * inch, 0.34 * inch
    margin_b, margin_t = 0.50 * inch, 0.26 * inch
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_b - margin_t

    x_min, x_max = -35, 65
    y_min, y_max = 0.35, 0.72

    outliers = [p for p in points if p["x"] < x_min or p["x"] > x_max or p["y"] > y_max]
    points = [p for p in points if p not in outliers]

    def sx(x):
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return margin_b + (y - y_min) / (y_max - y_min) * plot_h

    c = canvas.Canvas(OUT, pagesize=(width, height))
    c.setTitle("CASP solver feasibility")
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, stroke=0, fill=1)

    # Feasible region: positive net and ratio <= 0.5, clipped to the visible
    # plot box because the y-axis is zoomed to start above zero.
    c.setFillColor(colors.HexColor("#e6f4ea"))
    c.rect(
        sx(0),
        margin_b,
        margin_l + plot_w - sx(0),
        sy(0.5) - margin_b,
        stroke=0,
        fill=1,
    )

    # Grid lines.
    c.setStrokeColor(colors.HexColor("#dddddd"))
    c.setLineWidth(0.5)
    for x in [-25, 0, 25, 50]:
        c.line(sx(x), margin_b, sx(x), margin_b + plot_h)
    for y in [0.4, 0.5, 0.6, 0.7]:
        c.line(margin_l, sy(y), margin_l + plot_w, sy(y))

    # Axes.
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(margin_l, margin_b, margin_l + plot_w, margin_b)
    c.line(margin_l, margin_b, margin_l, margin_b + plot_h)

    # Constraint lines.
    c.setStrokeColor(colors.HexColor("#2ca25f"))
    c.setLineWidth(1.1)
    c.line(sx(0), margin_b, sx(0), margin_b + plot_h)
    c.line(margin_l, sy(0.5), margin_l + plot_w, sy(0.5))
    # Ticks and labels.
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    for x in [-25, 0, 25, 50]:
        c.line(sx(x), margin_b - 3, sx(x), margin_b + 3)
        c.drawCentredString(sx(x), margin_b - 14, str(x))
    for y in [0.4, 0.5, 0.6, 0.7]:
        c.line(margin_l - 3, sy(y), margin_l + 3, sy(y))
        c.drawRightString(margin_l - 8, sy(y) - 3, f"{y:.1f}")

    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(margin_l + plot_w / 2, 0.16 * inch, "Net gain @100 vs. existing source")
    c.saveState()
    c.translate(0.16 * inch, margin_b + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Cannibalized / gross @100")
    c.restoreState()

    # Severe outliers are annotated inside the zoomed panel so the feasible
    # region remains legible.
    c.setFont("Helvetica", 6.5)
    outlier_slots = {
        "Exact fusion": (10, 0.704),
        "DLCM slate": (39, 0.704),
        "PRM slate": (39, 0.670),
    }
    for idx, p in enumerate(sorted(outliers, key=lambda row: row["x"])):
        label_x, label_y = outlier_slots.get(p["name"], (37, 0.704 - idx * 0.035))
        y = sy(label_y)
        x = sx(label_x)
        c.setFillColor(p["color"])
        if p["shape"] == "square":
            c.rect(x, y - 3, 6, 6, stroke=0, fill=1)
        elif p["shape"] == "diamond":
            path = c.beginPath()
            path.moveTo(x + 3, y + 4)
            path.lineTo(x - 1, y)
            path.lineTo(x + 3, y - 4)
            path.lineTo(x + 7, y)
            path.close()
            c.drawPath(path, stroke=0, fill=1)
        else:
            c.circle(x + 3, y, 3, stroke=0, fill=1)
        draw_adjacent_label(c, x + 3, y, p["name"], "right", font_size=6.5)

    # Points.
    for p in points:
        x, y = sx(p["x"]), sy(p["y"])
        c.setFillColor(p["color"])
        c.setStrokeColor(colors.white)
        c.setLineWidth(0.6)
        if p["shape"] == "circle":
            c.circle(x, y, 4.5, stroke=1, fill=1)
        elif p["shape"] == "square":
            c.rect(x - 4, y - 4, 8, 8, stroke=1, fill=1)
        elif p["shape"] == "triangle":
            c.setStrokeColor(colors.white)
            path = c.beginPath()
            path.moveTo(x, y + 5)
            path.lineTo(x - 5, y - 4)
            path.lineTo(x + 5, y - 4)
            path.close()
            c.drawPath(path, stroke=1, fill=1)
        elif p["shape"] == "diamond":
            path = c.beginPath()
            path.moveTo(x, y + 5)
            path.lineTo(x - 5, y)
            path.lineTo(x, y - 5)
            path.lineTo(x + 5, y)
            path.close()
            c.drawPath(path, stroke=1, fill=1)

        draw_adjacent_label(c, x, y, p["name"], p["label_side"])

    # Small feasibility label.
    c.setFillColor(colors.HexColor("#1b7837"))
    c.setFont("Helvetica-Bold", 7)
    c.drawRightString(sx(62), sy(0.358), "CASP feasible region")

    c.showPage()
    c.save()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
