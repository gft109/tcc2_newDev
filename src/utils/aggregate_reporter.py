"""Aggregates all saved sets of an instance into results/aggregation/<instancia>/
(summary CSV, plots and LaTeX table), without re-running the solvers.

Usage: python src/utils/aggregate_reporter.py <instancia>
       python src/utils/aggregate_reporter.py all   # combined summary and viability LaTeX tables
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys

# allows running this file directly as a script
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.solution import WEIGHTS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data_base")
AGGREGATION_DIR = os.path.join(RESULTS_DIR, "aggregation")

SET_DIR_PATTERN = re.compile(r"set_(\d{2})$")


def discover_sets(instance_name: str) -> list[str]:
    base = os.path.join(RESULTS_DIR, instance_name)
    if not os.path.isdir(base):
        return []
    matches = [name for name in os.listdir(base) if SET_DIR_PATTERN.fullmatch(name)]
    return sorted(matches, key=lambda name: int(SET_DIR_PATTERN.fullmatch(name).group(1)))


def _read_csv_or_none(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def patient_days(instance_name: str, set_dir: str) -> int | None:
    """N_pd: total patient-days of the instance (sum of every patient's length of stay)."""
    patients = _read_csv_or_none(os.path.join(DATA_BASE_DIR, instance_name, set_dir, "patient.csv"))
    return None if patients is None else int(patients["los"].sum())


def min_daily_occupancy_rate(instance_name: str, set_dir: str) -> float | None:
    """Lowest (patients in the hospital / total beds) over the horizon; depends only on the data."""
    base = os.path.join(DATA_BASE_DIR, instance_name, set_dir)
    patients = _read_csv_or_none(os.path.join(base, "patient.csv"))
    rooms = _read_csv_or_none(os.path.join(base, "room.csv"))
    if patients is None or rooms is None:
        return None
    last_days = patients["admission_day"] + patients["los"] - 1
    census = [((patients["admission_day"] <= day) & (last_days >= day)).sum() for day in range(1, last_days.max() + 1)]
    return min(census) / rooms["capacity"].sum()


def collect_set_row(instance_name: str, set_dir: str) -> dict:
    """One aggregate_summary.csv row, from whichever CSVs exist for the set."""
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
        # counts recovered from the cost breakdown: each occurrence costs exactly its weight
        row["gurobi_specialty_mismatch_count"] = g.get("specialty_cost") / WEIGHTS["W_SPEC"]
        row["gurobi_transfer_count"] = g.get("transfer_cost") / WEIGHTS["W_TRANSF"]
        row["gurobi_mixed_room_day_count"] = g.get("gender_cost") / WEIGHTS["W_GEN"]
        row["gurobi_capacity_violation"] = g.get("capacity_cost") / WEIGHTS["W_CAP"]

    heuristic_df = _read_csv_or_none(os.path.join(base, "heuristic_runs.csv"))
    if heuristic_df is not None and not heuristic_df.empty:
        melhor = heuristic_df.loc[heuristic_df["total_cost"].idxmin()]
        row["heur_melhor_z"] = melhor["total_cost"] - melhor["capacity_cost"]
        row["heur_melhor_v_cap"] = melhor["capacity_cost"] / WEIGHTS["W_CAP"]
        row["heur_execucoes"] = len(heuristic_df)
        row["heur_execucoes_viaveis"] = int(heuristic_df["best_feasible_fitness"].notna().sum())
        row["n_pd"] = patient_days(instance_name, set_dir)
        if row["n_pd"]:
            row["heur_melhor_rho_cap"] = row["heur_melhor_v_cap"] / row["n_pd"]
        feasible = heuristic_df.dropna(subset=["best_feasible_fitness"])
        if not feasible.empty:
            row["heur_melhor_viavel_fitness"] = feasible.loc[
                feasible["best_feasible_fitness"].idxmin(), "best_feasible_fitness"
            ]

    if "gurobi_total_cost" in row and "heur_melhor_z" in row:
        row["heur_bate_gurobi"] = row["heur_melhor_z"] < row["gurobi_total_cost"]

    metrics_df = _read_csv_or_none(os.path.join(base, "solution_metrics.csv"))
    if metrics_df is not None:
        for _, metric_row in metrics_df.iterrows():
            prefix = "melhor" if metric_row["solucao"] == "melhor" else "viavel"
            for col in metrics_df.columns:
                if col == "solucao":
                    continue
                row[f"{prefix}_{col}"] = metric_row[col]
            if metric_row.get("peak_occupancy_total_capacity"):
                row[f"{prefix}_peak_occupancy_rate"] = (
                    metric_row["peak_occupancy_patients"] / metric_row["peak_occupancy_total_capacity"]
                )
                row[f"{prefix}_min_occupancy_rate"] = min_daily_occupancy_rate(instance_name, set_dir)

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
    if "heur_melhor_z" in df.columns:
        values = df["heur_melhor_z"].dropna().tolist()
        if values:
            data.append(values)
            labels.append("Heurística (melhor)")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Função objetivo Z")

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
    ("melhor_capacity_violation", "Violação de capacidade\n(pacientes-dia)"),
]


def plot_operational_metrics(df: pd.DataFrame, instance_name: str, out_dir: str) -> None:
    """Mean ± std across sets of each operational metric."""
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
# properties of the generated data: identical for any solution of the same set
INSTANCE_METRIC_SPECS = [
    ("horizon_days", "Horizonte (dias)", False),
    ("occupancy_rate_mean", "Ocupação Média (todo o horizonte)", True),
    ("peak_occupancy_rate", "Pico de Ocupação Diária", True),
    ("min_occupancy_rate", "Mínimo de Ocupação Diária", True),
]

OPERATIONAL_METRIC_SPECS = [
    ("specialty_mismatch_count", "Especialidade Incorreta (contagem)", False),
    ("transfer_count", "Transferências (contagem)", False),
    ("mixed_room_day_count", "Quarto-dias Mistos (contagem)", False),
    ("capacity_violation", "Violação de Capacidade (pacientes-dia)", False),
]


def _mean_std_n(series: pd.Series) -> tuple[float, float, int]:
    """Mean/std/count ignoring NaN and infinite values (e.g. Gurobi lower bound = -inf)."""
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


def build_summary_rows(df: pd.DataFrame) -> list[dict]:
    """Summary table rows (mean ± std across sets); operational metrics have one sub-row per approach."""
    rows: list[dict] = []

    def add(section: str, label: str, value: dict | None, is_pct: bool = False, group: str | None = None) -> None:
        if value is not None:
            rows.append({"section": section, "group": group, "label": label, "value": value, "is_pct": is_pct})

    for col, label, is_pct in INSTANCE_METRIC_SPECS:
        add("Instâncias", label, _metric_stats(df, f"melhor_{col}"), is_pct)

    add("Gurobi", "Z (Limite Superior)", _metric_stats(df, "gurobi_total_cost"))
    add("Gurobi", "Limite Inferior", _metric_stats(df, "gurobi_lower_bound"))
    add("Gurobi", "Gap de Otimalidade", _metric_stats(df, "gurobi_gap"), is_pct=True)

    add("Heurística", "Z", _metric_stats(df, "heur_melhor_z"))

    for col, label, is_pct in OPERATIONAL_METRIC_SPECS:
        add("Métricas Operacionais", "Gurobi", _metric_stats(df, f"gurobi_{col}"), is_pct, group=label)
        add("Métricas Operacionais", "Heurística", _metric_stats(df, f"melhor_{col}"), is_pct, group=label)

    return rows


def _format_value_txt(value: dict | None, is_pct: bool) -> str:
    if value is None:
        return "—"
    mean, std = value["mean"], value["std"]
    return f"{mean * 100:.1f}% ± {std * 100:.1f}%" if is_pct else f"{mean:.1f} ± {std:.1f}"


def render_txt_table(rows: list[dict], instance_name: str, n_sets: int) -> str:
    lines = ["=" * SUMMARY_TABLE_WIDTH]
    title = f"RESUMO AGREGADO ENTRE CONJUNTOS — {instance_name.upper()} ({n_sets} conjuntos)"
    lines.append(title.center(SUMMARY_TABLE_WIDTH))
    lines.append("=" * SUMMARY_TABLE_WIDTH)
    lines.append(f"{'Métrica':<{SUMMARY_LABEL_WIDTH}}| Média ± DP")
    lines.append("-" * SUMMARY_TABLE_WIDTH)

    current_section, current_group = None, None
    for row in rows:
        if row["section"] != current_section:
            current_section, current_group = row["section"], None
            header = f"-- {current_section} "
            lines.append(header + "-" * max(0, SUMMARY_TABLE_WIDTH - len(header)))
        if row["group"] and row["group"] != current_group:
            current_group = row["group"]
            lines.append(current_group)
        label = f"    {row['label']}" if row["group"] else row["label"]
        lines.append(f"{label:<{SUMMARY_LABEL_WIDTH}}| {_format_value_txt(row['value'], row['is_pct'])}")

    lines.append("=" * SUMMARY_TABLE_WIDTH)
    return "\n".join(lines)


def _latex_number(value: float, decimals: int = 1) -> str:
    """Formats numbers like 10\\,225,0 (thin-space thousands, decimal comma)."""
    formatted = f"{value:,.{decimals}f}"
    integer_part, _, decimal_part = formatted.partition(".")
    integer_part = integer_part.replace(",", r"\,")
    return f"{integer_part},{decimal_part}" if decimal_part else integer_part


def _format_value_latex(value: dict | None, is_pct: bool) -> str:
    if value is None:
        return "--"
    mean, std = value["mean"], value["std"]
    if is_pct:
        return f"{_latex_number(mean * 100)}\\% $\\pm$ {_latex_number(std * 100)}\\%"
    return f"{_latex_number(mean)} $\\pm$ {_latex_number(std)}"


INSTANCE_DISPLAY_NAMES = {"pequeno": "pequena", "medio": "média", "grande": "grande", "muito_grande": "muito grande"}


def render_latex_table(rows_by_instance: dict[str, list[dict]], caption: str, label: str) -> str:
    """Table in the TCC's style (table + tabular with \\hline), one value column per instance."""
    instances = list(rows_by_instance)
    ordered: list[tuple[str, str | None, str]] = []
    values: dict[tuple[str, str | None, str], dict[str, str]] = {}
    for instance, rows in rows_by_instance.items():
        for row in rows:
            key = (row["section"], row["group"], row["label"])
            if key not in values:
                ordered.append(key)
                values[key] = {}
            values[key][instance] = _format_value_latex(row["value"], row["is_pct"])

    n_cols = len(instances) + 1
    header = " & ".join(f"\\textbf{{{INSTANCE_DISPLAY_NAMES.get(i, i).capitalize()}}}" for i in instances)
    lines = [
        r"\begin{table}[htbp]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        f"    \\begin{{tabular}}{{l{'r' * len(instances)}}}",
        r"        \hline",
        f"        \\textbf{{Métrica}} & {header} \\\\",
        r"        \hline",
    ]
    current_section, current_group = None, None
    for section, group, row_label in ordered:
        if section != current_section:
            if current_section is not None:
                lines.append(r"        \hline")
            current_section, current_group = section, None
            lines.append(f"        \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{section}}}}} \\\\")
        if group and group != current_group:
            current_group = group
            lines.append(f"        \\multicolumn{{{n_cols}}}{{l}}{{{group}}} \\\\")
        label_tex = "$Z$" + row_label[1:] if row_label.startswith("Z") else row_label
        if group:
            label_tex = r"\quad " + label_tex
        cells = " & ".join(values[(section, group, row_label)].get(i, "--") for i in instances)
        lines.append(f"        {label_tex} & {cells} \\\\")
    lines += [r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def render_viability_latex_table(dfs: dict[str, pd.DataFrame]) -> str:
    """Capacity feasibility of the best heuristic solution of each set, one row per instance."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \caption{Viabilidade das soluções heurísticas por porte (melhor solução de cada conjunto, "
        r"média $\pm$ desvio padrão entre os 10 conjuntos)}",
        r"    \label{tab:res-viabilidade}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{l*{4}{>{\centering\arraybackslash}p{3.3cm}}}",
        r"        \hline",
        r"        \textbf{Porte} & \textbf{Execuções que encontraram solução viável} & "
        r"\textbf{Nº Violações de capacidade ($V_{cap}$)} & \textbf{Carga de internação ($N_{pd}$)} & "
        r"\textbf{Taxa de violação de capacidade ($\rho_{cap} = V_{cap} / N_{pd}$)} \\",
        r"        \hline",
    ]
    for instance, df in dfs.items():
        name = INSTANCE_DISPLAY_NAMES.get(instance, instance).capitalize()
        feasible = f"{int(df['heur_execucoes_viaveis'].sum())}/{int(df['heur_execucoes'].sum())}"
        v_cap = _format_value_latex(_metric_stats(df, "heur_melhor_v_cap"), is_pct=False)
        n_pd = _format_value_latex(_metric_stats(df, "n_pd"), is_pct=False)
        rho = _metric_stats(df, "heur_melhor_rho_cap")
        rho_tex = f"{_latex_number(rho['mean'] * 100, 3)}\\% $\\pm$ {_latex_number(rho['std'] * 100, 3)}\\%"
        lines.append(f"        {name} & {feasible} & {v_cap} & {n_pd} & {rho_tex} \\\\")
    lines += [r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def save_combined_latex_table() -> None:
    dfs = {}
    for instance in INSTANCE_DISPLAY_NAMES:
        sets = discover_sets(instance)
        if sets:
            dfs[instance] = build_aggregate_summary(instance, sets)
    rows_by_instance = {instance: build_summary_rows(df) for instance, df in dfs.items()}
    latex = render_latex_table(
        rows_by_instance,
        r"Resumo agregado entre os 10 conjuntos de cada porte (média $\pm$ desvio padrão)",
        "tab:agregado",
    )
    path = os.path.join(AGGREGATION_DIR, "summary_table.tex")
    with open(path, "w") as f:
        f.write(latex + "\n")
    print(f"Tabela combinada em {path}")

    path = os.path.join(AGGREGATION_DIR, "viability_table.tex")
    with open(path, "w") as f:
        f.write(render_viability_latex_table(dfs) + "\n")
    print(f"Tabela de viabilidade em {path}")


def print_and_save_summary_table(df: pd.DataFrame, instance_name: str, out_dir: str) -> None:
    rows = build_summary_rows(df)
    n_sets = len(df)

    txt = render_txt_table(rows, instance_name, n_sets)
    print(txt)
    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write(txt + "\n")

    display_name = INSTANCE_DISPLAY_NAMES.get(instance_name, instance_name)
    latex = render_latex_table(
        {instance_name: rows},
        f"Resumo agregado entre os {n_sets} conjuntos --- instância {display_name} (média $\\pm$ desvio padrão)",
        f"tab:agregado-{instance_name.replace('_', '-')}",
    )
    with open(os.path.join(out_dir, "summary_table.tex"), "w") as f:
        f.write(latex + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python aggregate_reporter.py <instancia | all>")
        sys.exit(1)
    instance_name = sys.argv[1]
    if instance_name == "all":
        save_combined_latex_table()
        return

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
