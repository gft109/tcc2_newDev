"""Copies the figures used by latex/Resultados.tex into latex/resultados/, to be uploaded
to the Overleaf project as Figuras/resultados/. Images are flattened to RGB: pdfLaTeX embeds
PNGs without alpha directly, which keeps compilation fast on Overleaf's free plan."""

import pathlib
import shutil

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "latex" / "resultados"
INSTANCES = ["pequeno", "medio", "grande", "muito_grande"]
REPRESENTATIVE_SET = "set_01"
SET_FIGURES = ["boxplot.png", "convergence_broken_axis.png", "gurobi_bounds.png", "gurobi_vs_iterations.png"]
AGGREGATE_FIGURES = ["boxplot_gurobi_vs_heuristica.png", "operational_metrics_across_sets.png"]
CALIBRATION_FIGURE = ROOT / "experiments" / "capacity" / "results" / "fitness_vs_capacity_penalty.png"


def copy_as_rgb(source: pathlib.Path, target: pathlib.Path) -> None:
    image = Image.open(source)
    if image.mode != "RGB":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A") if "A" in image.getbands() else None)
        image = background
    image.save(target, optimize=True)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    for instance in INSTANCES:
        target = OUT / instance
        target.mkdir(parents=True)
        for figure in AGGREGATE_FIGURES:
            copy_as_rgb(RESULTS / "aggregation" / instance / figure, target / figure)

        (target / REPRESENTATIVE_SET).mkdir()
        for figure in SET_FIGURES:
            copy_as_rgb(RESULTS / instance / REPRESENTATIVE_SET / figure, target / REPRESENTATIVE_SET / figure)

    (OUT / "calibracao").mkdir()
    copy_as_rgb(CALIBRATION_FIGURE, OUT / "calibracao" / CALIBRATION_FIGURE.name)

    print(f"Pasta pronta: {OUT}")


if __name__ == "__main__":
    main()
