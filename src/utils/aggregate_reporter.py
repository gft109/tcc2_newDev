"""Aggregates results across however many generated sets (set_01..set_NN) of an
instance size have already been run, into one consolidated report under
results/aggregation/<instancia>/ — the cross-INSTANCE-GENERATION view, complementary
to the cross-REPETITION view main.py already reports within a single set.

Two different sources of variance, both worth seeing:
  - Across repetitions (existing, within one set_NN/): how much does the outcome vary
    due to the heuristic's own search randomness, holding the instance fixed?
  - Across sets (this file): how much does the outcome vary due to the random
    instance GENERATION itself, holding instance SIZE fixed? Running main.py with
    "all"/a range deliberately does NOT answer this on its own (see CLAUDE.md) — each
    set is reported independently, by design, until now.

Reads whatever results/<instancia>/set_NN/ directories already exist on disk
(gurobi_run.csv, heuristic_runs.csv, solution_metrics.csv) — does not re-run any
solver, and works with however many sets have been run so far (not all 10 need to
exist). solution_metrics.csv in particular may be missing for sets generated before
that file existed; those columns are simply NaN for that set, not an error.

Usage:
    python src/utils/aggregate_reporter.py <instancia>
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
AGGREGATION_DIR = os.path.join(RESULTS_DIR, "aggregation")

SET_DIR_PATTERN = re.compile(r"set_(\d{2})$")


def discover_sets(instance_name: str) -> list[str]:
    """Sorted set_NN directory names that actually exist under results/<instance_name>/
    — not assumed to be 1..10, so this works with partial completion."""
    base = os.path.join(RESULTS_DIR, instance_name)
    if not os.path.isdir(base):
        return []
    matches = [name for name in os.listdir(base) if SET_DIR_PATTERN.fullmatch(name)]
    return sorted(matches, key=lambda name: int(SET_DIR_PATTERN.fullmatch(name).group(1)))


def _read_csv_or_none(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def collect_set_row(instance_name: str, set_dir: str) -> dict:
    """One row of aggregate_summary.csv for a single set — pulls together whatever of
    gurobi_run.csv / heuristic_runs.csv / solution_metrics.csv exists for it."""
    set_number = int(SET_DIR_PATTERN.fullmatch(set_dir).group(1))
    base = os.path.join(RESULTS_DIR, instance_name, set_dir)
    row: dict = {"set": set_number}

    gurobi_df = _read_csv_or_none(os.path.join(base, "gurobi_run.csv"))
    if gurobi_df is not None and not gurobi_df.empty:
        g = gurobi_df.iloc[0]
        row["gurobi_total_cost"] = g.get("total_cost")
        row["gurobi_upper_bound"] = g.get("upper_bound")
        row["gurobi_lower_bound"] = g.get("lower_bound")
        row["gurobi_gap"] = g.get("gap")
        row["gurobi_is_optimal"] = g.get("is_optimal")

    heuristic_df = _read_csv_or_none(os.path.join(base, "heuristic_runs.csv"))
    if heuristic_df is not None and not heuristic_df.empty:
        melhor = heuristic_df.loc[heuristic_df["total_cost"].idxmin()]
        row["heur_melhor_total_cost"] = melhor["total_cost"]
        feasible = heuristic_df.dropna(subset=["best_feasible_fitness"])
        if not feasible.empty:
            row["heur_melhor_viavel_fitness"] = feasible.loc[
                feasible["best_feasible_fitness"].idxmin(), "best_feasible_fitness"
            ]

    if "gurobi_total_cost" in row and "heur_melhor_total_cost" in row:
        row["heur_bate_gurobi"] = row["heur_melhor_total_cost"] < row["gurobi_total_cost"]

    metrics_df = _read_csv_or_none(os.path.join(base, "solution_metrics.csv"))
    if metrics_df is not None:
        for _, metric_row in metrics_df.iterrows():
            prefix = "melhor" if metric_row["solucao"] == "melhor" else "viavel"
            for col in metrics_df.columns:
                if col == "solucao":
                    continue
                row[f"{prefix}_{col}"] = metric_row[col]

    return row


def build_aggregate_summary(instance_name: str, sets: list[str]) -> pd.DataFrame:
    rows = [collect_set_row(instance_name, set_dir) for set_dir in sets]
    return pd.DataFrame(rows).sort_values("set").reset_index(drop=True)


def plot_boxplot_gurobi_vs_heuristica(df: pd.DataFrame, instance_name: str, out_dir: str) -> None:
    data, labels = [], []
    if "gurobi_total_cost" in df.columns:
        values = df["gurobi_total_cost"].dropna().tolist()
        if values:
            data.append(values)
            labels.append("Gurobi")
    if "heur_melhor_total_cost" in df.columns:
        values = df["heur_melhor_total_cost"].dropna().tolist()
        if values:
            data.append(values)
            labels.append("Heurística (melhor)")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Custo total (função objetivo)")

    title = f"Gurobi vs. Heurística entre conjuntos — {instance_name}"
    if "heur_bate_gurobi" in df.columns:
        comparable = df["heur_bate_gurobi"].dropna()
        if len(comparable) > 0:
            title += f"\nHeurística superou o Gurobi em {int(comparable.sum())}/{len(comparable)} conjuntos"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boxplot_gurobi_vs_heuristica.png"), dpi=150)
    plt.close(fig)


OPERATIONAL_METRICS = [
    ("melhor_occupancy_rate_mean", "Ocupação média"),
    ("melhor_specialty_mismatch_count", "Especialidade incorreta\n(contagem)"),
    ("melhor_transfer_count", "Transferências\n(contagem)"),
    ("melhor_mixed_room_day_count", "Quarto-dias mistos\n(contagem)"),
    ("melhor_capacity_violation", "Violação de capacidade\n(instâncias de paciente)"),
]


def plot_operational_metrics(df: pd.DataFrame, instance_name: str, out_dir: str) -> None:
    """Mean ± std ACROSS SETS (not across room-days within one solution, nor across
    repetitions within one set) of each operational metric from solution_metrics.csv's
    "melhor" row — how consistent is each one across different random instances of
    this same size?"""
    available = [(col, label) for col, label in OPERATIONAL_METRICS if col in df.columns and df[col].notna().any()]
    if not available:
        print("  (sem solution_metrics.csv em nenhum conjunto — pulando métricas operacionais)")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(3.2 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        values = df[col].dropna().tolist()
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ax.bar(["todos\nos conjuntos"], [mean], yerr=[std], color="steelblue", capsize=6)
        ax.set_title(label, fontsize=10)
        ax.text(0, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Métricas operacionais entre conjuntos (média ± DP) — {instance_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "operational_metrics_across_sets.png"), dpi=150)
    plt.close(fig)


SUMMARY_TABLE_WIDTH = 100
SUMMARY_LABEL_WIDTH = 42
SUMMARY_COLUMN_WIDTH = 26

# (aggregate_summary.csv column suffix, display label, is a percentage)
OPERATIONAL_METRIC_SPECS = [
    ("horizon_days", "Horizonte (dias)", False),
    ("specialty_mismatch_count", "Especialidade Incorreta (contagem)", False),
    ("transfer_count", "Transferências (contagem)", False),
    ("mixed_room_day_count", "Quarto-dias Mistos (contagem)", False),
    ("capacity_violation", "Violação de Capacidade (instâncias)", False),
    ("occupancy_rate_mean", "Ocupação Média", True),
    ("occupancy_rate_std", "Ocupação (DP intra-solução)", True),
    ("peak_occupancy_patients", "Pico de Ocupação (pacientes)", False),
    ("peak_occupancy_total_capacity", "Pico de Ocupação (capacidade total)", False),
    ("peak_occupancy_day", "Pico de Ocupação (dia)", False),
]


def _mean_std_n(series: pd.Series) -> tuple[float, float, int]:
    """Mean/std/count across sets, dropping NaN AND non-finite values (e.g. Gurobi's
    lower_bound can be -inf when it never leaves presolve, see muito_grande) — those
    aren't a real number to average, so they're excluded rather than poisoning the
    mean with inf/nan."""
    values = [v for v in series.dropna().tolist() if math.isfinite(v)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return mean, std, n


def _metric_stats(df: pd.DataFrame, col: str) -> dict | None:
    if col not in df.columns:
        return None
    mean, std, n = _mean_std_n(df[col])
    if n == 0:
        return None
    return {"mean": mean, "std": std, "n": n}


def _count_stats(df: pd.DataFrame, col: str) -> dict | None:
    if col not in df.columns:
        return None
    series = df[col].dropna()
    if len(series) == 0:
        return None
    return {"count": int(series.sum()), "total": len(series)}


def build_summary_rows(df: pd.DataFrame) -> list[dict]:
    """All of aggregate_summary.csv's information, one row per metric, each holding
    mean±std ACROSS SETS (or a count/total pair for boolean columns) for both the
    "melhor" and "melhor_viavel" solutions side by side, grouped into sections. No
    per-row coverage count — once every set has run this is uniform, and partial runs
    are easy to tell apart already (that metric's row is just missing)."""
    rows: list[dict] = []

    def add(section: str, label: str, melhor=None, viavel=None, is_pct: bool = False, collapse_viavel: bool = False) -> None:
        if melhor is None and viavel is None:
            return
        rows.append({
            "section": section, "label": label, "melhor": melhor, "viavel": viavel,
            "is_pct": is_pct, "collapse_viavel": collapse_viavel,
        })

    add("Gurobi", "Custo Total (Upper Bound)", _metric_stats(df, "gurobi_total_cost"))
    add("Gurobi", "Lower Bound", _metric_stats(df, "gurobi_lower_bound"))
    add("Gurobi", "Gap de Otimalidade", _metric_stats(df, "gurobi_gap"), is_pct=True)
    add("Gurobi", "Rodadas com Ótimo Provado", {"count_pair": _count_stats(df, "gurobi_is_optimal")})

    add("Heurística", "Custo Total", _metric_stats(df, "heur_melhor_total_cost"), _metric_stats(df, "heur_melhor_viavel_fitness"))
    add("Heurística", "Vitórias sobre Gurobi", {"count_pair": _count_stats(df, "heur_bate_gurobi")})

    # True only when EVERY set that found a feasible solution had it coincide exactly
    # with that set's overall winner (viavel_mesma_que_melhor, written by
    # ResultsReporter.export_solution_metrics_csv) — in that case every operational
    # metric below is, by construction, identical between the two columns, so the
    # "Melhor Viável" column is collapsed to a plain note instead of repeating every
    # number as if it were new information.
    coincidence = _count_stats(df, "viavel_mesma_que_melhor")
    if coincidence:
        add("Heurística", "Conjuntos onde Melhor Viável = Melhor", {"count_pair": coincidence})
    all_coincide = coincidence is not None and coincidence["count"] == coincidence["total"]

    specialty_suffixes = sorted({
        col[len("melhor_occupancy_rate_"):]
        for col in df.columns
        if col.startswith("melhor_occupancy_rate_") and col not in ("melhor_occupancy_rate_mean", "melhor_occupancy_rate_std")
    })
    metrics = list(OPERATIONAL_METRIC_SPECS) + [
        (f"occupancy_rate_{suffix}", f"Ocupação — {suffix.title()}", True) for suffix in specialty_suffixes
    ]
    for col, label, is_pct in metrics:
        add(
            "Métricas Operacionais", label, _metric_stats(df, f"melhor_{col}"), _metric_stats(df, f"viavel_{col}"),
            is_pct, collapse_viavel=all_coincide,
        )

    # Drop placeholder rows whose only content was a None count_pair (e.g. no is_optimal data at all)
    return [row for row in rows if row["melhor"] is not None or row["viavel"] is not None]


def _format_value_txt(value: dict | None, is_pct: bool) -> str:
    if value is None:
        return "—"
    if "count_pair" in value:
        pair = value["count_pair"]
        return f"{pair['count']}/{pair['total']}" if pair else "—"
    mean, std = value["mean"], value["std"]
    return f"{mean * 100:.1f}% ± {std * 100:.1f}%" if is_pct else f"{mean:.1f} ± {std:.1f}"


def render_txt_table(rows: list[dict], instance_name: str, n_sets: int) -> str:
    lines = ["=" * SUMMARY_TABLE_WIDTH]
    title = f"RESUMO AGREGADO ENTRE CONJUNTOS — {instance_name.upper()} ({n_sets} conjuntos)"
    lines.append(title.center(SUMMARY_TABLE_WIDTH))
    lines.append("=" * SUMMARY_TABLE_WIDTH)
    lines.append(
        f"{'Métrica':<{SUMMARY_LABEL_WIDTH}}| {'Melhor':<{SUMMARY_COLUMN_WIDTH}}| {'Melhor Viável'}"
    )
    lines.append("-" * SUMMARY_TABLE_WIDTH)

    current_section = None
    for row in rows:
        if row["section"] != current_section:
            current_section = row["section"]
            header = f"-- {current_section} "
            lines.append(header + "-" * max(0, SUMMARY_TABLE_WIDTH - len(header)))
        melhor_str = _format_value_txt(row["melhor"], row["is_pct"])
        if row["collapse_viavel"] and row["viavel"] is not None:
            viavel_str = "= Melhor (ver acima)"
        else:
            viavel_str = _format_value_txt(row["viavel"], row["is_pct"])
        lines.append(f"{row['label']:<{SUMMARY_LABEL_WIDTH}}| {melhor_str:<{SUMMARY_COLUMN_WIDTH}}| {viavel_str}")

    lines.append("=" * SUMMARY_TABLE_WIDTH)
    return "\n".join(lines)


def _latex_number(value: float, decimals: int = 1) -> str:
    """Matches analiseResultados.tex's existing number style: \\, as the thousands
    separator, comma as the decimal point (e.g. 10225.0 -> "10\\,225,0")."""
    formatted = f"{value:,.{decimals}f}"
    integer_part, _, decimal_part = formatted.partition(".")
    integer_part = integer_part.replace(",", r"\,")
    return f"{integer_part},{decimal_part}" if decimal_part else integer_part


def _format_value_latex(value: dict | None, is_pct: bool) -> str:
    if value is None:
        return "--"
    if "count_pair" in value:
        pair = value["count_pair"]
        return f"{pair['count']}/{pair['total']}" if pair else "--"
    mean, std = value["mean"], value["std"]
    if is_pct:
        return f"{_latex_number(mean * 100)}\\% $\\pm$ {_latex_number(std * 100)}\\%"
    return f"{_latex_number(mean)} $\\pm$ {_latex_number(std)}"


def render_latex_table(rows: list[dict], instance_name: str, n_sets: int) -> str:
    """longtable + booktabs, matching the style already used in analiseResultados.tex
    (both packages are already in its preamble) — plain `table`+`tabular` would risk
    overflowing a page given how many metrics this covers."""
    display_name = instance_name.replace("_", " ").capitalize()
    label = instance_name.replace("_", "-")
    lines = [
        r"\begin{longtable}{lrr}",
        f"\\caption{{Resumo agregado entre conjuntos --- {display_name} ({n_sets} conjuntos)}} "
        f"\\label{{tab:agregado-{label}}} \\\\",
        r"\toprule",
        r"Métrica & Melhor & Melhor Viável \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{c}{\tablename\ \thetable{} -- continuação da página anterior} \\",
        r"\toprule",
        r"Métrica & Melhor & Melhor Viável \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{continua na próxima página} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    current_section = None
    for row in rows:
        if row["section"] != current_section:
            if current_section is not None:
                lines.append(r"\midrule")
            current_section = row["section"]
            lines.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{current_section}}}}} \\\\")
            lines.append(r"\midrule")
        melhor_str = _format_value_latex(row["melhor"], row["is_pct"])
        if row["collapse_viavel"] and row["viavel"] is not None:
            viavel_str = r"\textit{= Melhor}"
        else:
            viavel_str = _format_value_latex(row["viavel"], row["is_pct"])
        lines.append(f"{row['label']} & {melhor_str} & {viavel_str} \\\\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def print_and_save_summary_table(df: pd.DataFrame, instance_name: str, out_dir: str) -> None:
    rows = build_summary_rows(df)
    n_sets = len(df)

    txt = render_txt_table(rows, instance_name, n_sets)
    print(txt)
    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write(txt + "\n")

    latex = render_latex_table(rows, instance_name, n_sets)
    with open(os.path.join(out_dir, "summary_table.tex"), "w") as f:
        f.write(latex + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python aggregate_reporter.py <instancia>")
        sys.exit(1)
    instance_name = sys.argv[1]

    sets = discover_sets(instance_name)
    if not sets:
        raise FileNotFoundError(
            f"Nenhum conjunto encontrado em results/{instance_name}/ — rode "
            f"'python main.py {instance_name} <repetições>' pelo menos uma vez antes."
        )

    out_dir = os.path.join(AGGREGATION_DIR, instance_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print(f"=== AGREGAÇÃO ENTRE CONJUNTOS — {instance_name.upper()} ===")
    print(f"Conjuntos encontrados: {sets}")
    print("=" * 72)

    df = build_aggregate_summary(instance_name, sets)
    df.to_csv(os.path.join(out_dir, "aggregate_summary.csv"), index=False)

    plot_boxplot_gurobi_vs_heuristica(df, instance_name, out_dir)
    plot_operational_metrics(df, instance_name, out_dir)

    print()
    print_and_save_summary_table(df, instance_name, out_dir)
    print()
    print(f"Resultados em {out_dir}/ (aggregate_summary.csv tem a tabela crua, uma linha por conjunto)")


if __name__ == "__main__":
    main()
