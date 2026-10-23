"""Copies the result figures into latex/Figuras/resultados/, the figure folder of the Overleaf
project in latex/. By default only set_01 of each instance; --all-sets copies every set. Images are flattened (no alpha) and reduced to a
256-colour palette: pdfLaTeX embeds such PNGs directly and the files stay small, which keeps
compilation fast on Overleaf's free plan."""

import pathlib
import shutil
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "latex" / "Figuras" / "resultados"
INSTANCES = ["pequeno", "medio", "grande", "muito_grande"]
REPRESENTATIVE_SET = "set_01"
SET_FIGURES = ["convergence_broken_axis.png", "convergence_log.png", "capacity_violation.png"]
AGGREGATE_FIGURES = ["boxplot_sets.png"]
CROSS_INSTANCE_FIGURES = ["convergence_log.png", "capacity_violation_sets.png"]
CALIBRATION_FIGURE = ROOT / "experiments" / "capacity" / "results" / "fitness_vs_capacity_penalty.png"


def copy_light(source: pathlib.Path, target: pathlib.Path) -> None:
    image = Image.open(source)
    if image.mode != "RGB":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A") if "A" in image.getbands() else None)
        image = background
    # plots use few flat colours, so a palette without dithering is visually lossless
    image = image.quantize(colors=256, dither=Image.Dither.NONE)
    image.save(target, optimize=True)


def main() -> None:
    all_sets = "--all-sets" in sys.argv[1:]
    if OUT.exists():
        shutil.rmtree(OUT)

    for instance in INSTANCES:
        target = OUT / instance
        target.mkdir(parents=True)
        for figure in AGGREGATE_FIGURES:
            copy_light(RESULTS / "aggregation" / instance / figure, target / figure)

        set_dirs = (
            sorted(d.name for d in (RESULTS / instance).iterdir() if d.is_dir() and d.name.startswith("set_"))
            if all_sets
            else [REPRESENTATIVE_SET]
        )
        for set_dir in set_dirs:
            (target / set_dir).mkdir()
            for figure in SET_FIGURES:
                if (RESULTS / instance / set_dir / figure).exists():
                    copy_light(RESULTS / instance / set_dir / figure, target / set_dir / figure)

    for figure in CROSS_INSTANCE_FIGURES:
        copy_light(RESULTS / "aggregation" / figure, OUT / figure)

    (OUT / "calibracao").mkdir()
    copy_light(CALIBRATION_FIGURE, OUT / "calibracao" / CALIBRATION_FIGURE.name)

    print(f"Pasta pronta: {OUT}")


if __name__ == "__main__":
    main()
