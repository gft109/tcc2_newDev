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
from matplotlib.ticker import FuncFormatter
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


def minimum_specialty_mismatch(instance_name: str, set_dir: str) -> int | None:
    """M: patient-days that must sit in a room of another specialty in any feasible solution,
    i.e. per day and specialty, the inpatients exceeding the beds of that specialty."""
    base = os.path.join(DATA_BASE_DIR, instance_name, set_dir)
    patients = _read_csv_or_none(os.path.join(base, "patient.csv"))
    rooms = _read_csv_or_none(os.path.join(base, "room.csv"))
    if patients is None or rooms is None:
        return None
    beds = rooms.groupby("specialty")["capacity"].sum()
    last_days = patients["admission_day"] + patients["los"] - 1
    excess = 0
    for day in range(1, last_days.max() + 1):
        demand = patients.loc[(patients["admission_day"] <= day) & (last_days >= day), "required_specialty"].value_counts()
        excess += sum(max(0, n - beds.get(specialty, 0)) for specialty, n in demand.items())
    return int(excess)


def occupancy_structure(instance_name: str, set_dir: str) -> dict | None:
    """Room count, days with every bed taken and the parity minimum of mixed room-days: on such a day
    all rooms are full, so if the number of women is not a multiple of the (common) room capacity,
    at least one room must hold both genders."""
    base = os.path.join(DATA_BASE_DIR, instance_name, set_dir)
    patients = _read_csv_or_none(os.path.join(base, "patient.csv"))
    rooms = _read_csv_or_none(os.path.join(base, "room.csv"))
    if patients is None or rooms is None:
        return None
    total_beds = rooms["capacity"].sum()
    common_capacity = rooms["capacity"].iloc[0] if rooms["capacity"].nunique() == 1 else None
    last_days = patients["admission_day"] + patients["los"] - 1
    full_days = mixed_minimum = 0
    for day in range(1, last_days.max() + 1):
        inpatients = patients[(patients["admission_day"] <= day) & (last_days >= day)]
        if len(inpatients) == total_beds:
            full_days += 1
            women = int((inpatients["gender"] == "F").sum())
            if common_capacity is not None and women % common_capacity != 0:
                mixed_minimum += 1
    return {"n_quartos": len(rooms), "dias_lotacao_total": full_days, "min_quartos_mistos": mixed_minimum}


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
        executions_z = heuristic_df["total_cost"] - heuristic_df["capacity_cost"]
        row["heur_media_z"] = executions_z.mean()
        row["heur_dp_z"] = executions_z.std() if len(executions_z) > 1 else 0.0
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

    structure = occupancy_structure(instance_name, set_dir)
    if structure is not None:
        row.update(structure)

    minimum = minimum_specialty_mismatch(instance_name, set_dir)
    if minimum is not None:
        row["min_especialidade_inadequada"] = minimum
        row["lb_comb"] = WEIGHTS["W_SPEC"] * minimum
        # best lower bound available: the solver's or the combinatorial one (eq:limite-combinatorio)
        row["lb_efetivo"] = max(row["lb_comb"], row.get("gurobi_lower_bound") or 0.0)
        if "heur_melhor_z" in row and row["lb_efetivo"] > 0:
            row["heur_melhor_excesso"] = (row["heur_melhor_z"] - row["lb_efetivo"]) / row["lb_efetivo"]

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


def _executions_z(instance_name: str, set_dir: str) -> list[float]:
    runs = _read_csv_or_none(os.path.join(RESULTS_DIR, instance_name, set_dir, "heuristic_runs.csv"))
    if runs is None:
        return []
    return (runs["specialty_cost"] + runs["transfer_cost"] + runs["gender_cost"]).tolist()


def plot_sets_boxplot(instance_name: str, sets: list[str], out_dir: str) -> None:
    """Z of every execution, one box per set: within-set vs. between-set variation side by side."""
    data = [_executions_z(instance_name, set_dir) for set_dir in sets]
    labels = [SET_DIR_PATTERN.fullmatch(set_dir).group(1) for set_dir in sets]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("Conjunto")
    ax.set_ylabel("Função objetivo Z")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.set_title(f"Z das execuções da heurística por conjunto — instância {INSTANCE_DISPLAY_NAMES.get(instance_name, instance_name)}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boxplot_sets.png"), dpi=150)
    plt.close(fig)


def plot_convergence_log(instances: list[str], set_dir: str = "set_01") -> None:
    """Best fitness per generation of every execution of one set per instance, log y-axis, 2x2 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, instance in zip(axes.flat, instances):
        curves = _read_csv_or_none(os.path.join(RESULTS_DIR, instance, set_dir, "convergence.csv"))
        if curves is None:
            ax.set_visible(False)
            continue
        histories = [group.sort_values("generation")["best_fitness"].tolist() for _, group in curves.groupby("iteration")]
        best = min(histories, key=lambda history: history[-1])
        for history in histories:
            ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)
        ax.plot(best, color="steelblue", linewidth=2.5, label="Execução de menor F")
        ax.set_yscale("log")
        ax.set_title(f"Instância {INSTANCE_DISPLAY_NAMES.get(instance, instance)} (conjunto {int(set_dir[-2:])})")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Melhor fitness F (escala log)")
        ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(AGGREGATION_DIR, "convergence_log.png"), dpi=150)
    plt.close(fig)


def plot_capacity_violation_sets(instances: list[str]) -> None:
    """V_cap of the best individual per generation, one curve per set (its lowest-F execution), one
    broken-axis panel per instance."""
    from utils.results_representation import plot_broken_axis
    fig, axes = plt.subplots(
        2, len(instances), figsize=(6 * len(instances), 6), sharex="col", squeeze=False,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    for col, instance in enumerate(instances):
        best_curves = []
        for set_dir in discover_sets(instance):
            curves = _read_csv_or_none(os.path.join(RESULTS_DIR, instance, set_dir, "convergence.csv"))
            if curves is None or "best_capacity_violation" not in curves:
                continue
            finals = curves.sort_values("generation").groupby("iteration").tail(1)
            best_iteration = finals.loc[finals["best_fitness"].idxmin(), "iteration"]
            best = curves[curves["iteration"] == best_iteration].sort_values("generation")
            best_curves.append((int(set_dir[-2:]), best["best_capacity_violation"].tolist()))

        def draw(ax, curves=best_curves):
            for set_number, values in curves:
                ax.plot(range(len(values)), values, linewidth=1.5, alpha=0.8, label=f"Conjunto {set_number}")

        ax_top, ax_bottom = axes[0][col], axes[1][col]
        plot_broken_axis([values for _, values in best_curves], draw, axes=(ax_top, ax_bottom))
        ax_top.set_title(f"Instância {INSTANCE_DISPLAY_NAMES.get(instance, instance)}")
        ax_top.legend(fontsize=10, ncol=2, loc="upper right")
        ax_bottom.set_xlabel("Geração")
        ax_bottom.set_ylabel("$V_{cap}$ do melhor indivíduo (pacientes-dia)")
    fig.tight_layout()
    fig.savefig(os.path.join(AGGREGATION_DIR, "capacity_violation_sets.png"), dpi=150)
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


def render_objective_latex_table(df: pd.DataFrame, instance_name: str) -> str:
    """Two tables per instance: (A) bounds on the optimum, (B) the heuristic's results per set."""
    display_name = INSTANCE_DISPLAY_NAMES.get(instance_name, instance_name)
    label_suffix = instance_name.replace("_", "-")
    n_executions = int(df["heur_execucoes"].max())

    bounds = [
        r"\begin{table}[htbp]",
        f"    \\caption{{Limites do valor ótimo de $Z$ por conjunto --- instância {display_name}}}",
        f"    \\label{{tab:res-limites-{label_suffix}}}",
        r"    \centering",
        r"    \begin{tabular}{crrrr}",
        r"        \hline",
        r"        & \multicolumn{3}{c}{\textbf{Gurobi}} & \textbf{Limite combinatório} \\",
        r"        \cline{2-4} \cline{5-5}",
        r"        \textbf{Conjunto} & \textbf{Limite superior} & \textbf{Limite inferior} & \textbf{\textit{Gap}} & "
        r"\textbf{$LB_{comb}$} \\",
        r"        \hline",
    ]
    for _, row in df.iterrows():
        bounds.append(
            f"        {int(row['set']):02d} & {_latex_number(row['gurobi_upper_bound'])} & "
            f"{_latex_number(row['gurobi_lower_bound'])} & {_gap_latex(row)} & {_latex_number(row['lb_comb'])} \\\\"
        )
    bounds += [r"        \hline", r"    \end{tabular}", r"    \par\smallskip"]
    notes = []
    if (df["gurobi_lower_bound"] <= 0).any():
        notes.append(r"--- indica que o \textit{solver} não obteve limite inferior não trivial dentro do teto de tempo.")
    notes.append(r"\textit{Gap} na convenção do \textit{solver}. $LB_{comb}$: limite inferior combinatório, conforme a Equação~\eqref{eq:limite-combinatorio}.")
    bounds += ["    {\\footnotesize Nota - " + " ".join(notes) + "}", r"\end{table}"]

    heuristic = [
        r"\begin{table}[htbp]",
        f"    \\caption{{Resultados da heurística por conjunto --- instância {display_name} "
        f"({n_executions} execuções por conjunto; $Z$ sem a penalidade de capacidade)}}",
        f"    \\label{{tab:res-z-{label_suffix}}}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{crrrrrcr}",
        r"        \hline",
        r"        \textbf{Conjunto} & \textbf{$Z$ da execução de menor $F$} & \textbf{$V_{cap}$} & "
        r"\textbf{$Z$ médio} & \textbf{CV} & \textbf{Melhor $Z$ viável} & \textbf{Execuções viáveis} & "
        r"\textbf{Excesso máx.} \\",
        r"        \hline",
    ]
    for _, row in df.iterrows():
        feasible = row.get("heur_melhor_viavel_fitness")
        feasible_tex = "---" if pd.isna(feasible) else _latex_number(feasible)
        heuristic.append(
            f"        {int(row['set']):02d} & {_latex_number(row['heur_melhor_z'])} & {int(row['heur_melhor_v_cap'])} & "
            f"{_latex_number(row['heur_media_z'])} & {_latex_number(row['heur_dp_z'] / row['heur_media_z'] * 100)}\\% & "
            f"{feasible_tex} & {int(row['heur_execucoes_viaveis'])}/{int(row['heur_execucoes'])} & "
            f"{_latex_number(row['heur_melhor_excesso'] * 100)}\\% \\\\"
        )
    heuristic += [
        r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"    \par\smallskip",
        r"    {\footnotesize Nota - $Z$ da execução de menor $F$: valor de $Z$ da execução com menor \textit{fitness}"
        r", conforme a Equação~\eqref{eq:fitness}. $V_{cap}$: violação de capacidade dessa execução, em pacientes-dia"
        r", conforme a Equação~\eqref{eq:violacao-capacidade}, sendo $V_{cap} > 0$ uma solução inviável para o modelo exato. "
        r"CV: coeficiente de variação de $Z$ entre as execuções (desvio padrão dividido pela média), como medida de "
        r"dispersão. Melhor $Z$ viável: menor $Z$ entre as soluções viáveis encontradas nas execuções (--- se "
        r"nenhuma). Excesso máx.: excesso máximo da execução de menor $F$ sobre o ótimo, conforme a Equação~\eqref{eq:excesso-otimo}, "
        r"calculado com o maior entre o limite inferior do Gurobi e $LB_{comb}$.}",
        r"\end{table}",
    ]
    return "\n".join(bounds) + "\n\n" + "\n".join(heuristic)


def _gap_latex(row: pd.Series) -> str:
    """Gurobi gap, or --- when the lower bound is the trivial 0 (gap would be 100% by construction)."""
    if row["gurobi_lower_bound"] <= 0:
        return "---"
    return f"{_latex_number(row['gurobi_gap'] * 100)}\\%"


def save_objective_latex_table(df: pd.DataFrame, instance_name: str) -> None:
    out_dir = os.path.join(AGGREGATION_DIR, instance_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "objective_table.tex")
    with open(path, "w") as f:
        f.write(render_objective_latex_table(df, instance_name) + "\n")
    print(f"Tabela de Z por conjunto em {path}")


def render_sigma_viability_latex_table(dfs: dict[str, pd.DataFrame]) -> str:
    """Spread of Z between executions vs. W_cap, next to how many executions found a feasible solution."""
    w_cap = WEIGHTS["W_CAP"]
    lines = [
        r"\begin{table}[htbp]",
        f"    \\caption{{Desvio padrão de $Z$ entre execuções e viabilidade das soluções heurísticas por porte "
        f"($W_{{cap}} = {int(w_cap)}$)}}",
        r"    \label{tab:res-viabilidade-sigma}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{l*{3}{>{\centering\arraybackslash}p{3.8cm}}}",
        r"        \hline",
        r"        \textbf{Porte} & \textbf{$\sigma$ de $Z$ entre execuções (mínimo--máximo)} & "
        r"\textbf{Conjuntos com $\sigma > W_{cap}$} & \textbf{Execuções que encontraram solução viável} \\",
        r"        \hline",
    ]
    for instance, df in dfs.items():
        name = INSTANCE_DISPLAY_NAMES.get(instance, instance).capitalize()
        sigma = df["heur_dp_z"]
        lines.append(
            f"        {name} & {_latex_number(sigma.min(), 0)} -- {_latex_number(sigma.max(), 0)} & "
            f"{int((sigma > w_cap).sum())}/{len(df)} & "
            f"{int(df['heur_execucoes_viaveis'].sum())}/{int(df['heur_execucoes'].sum())} \\\\"
        )
    lines += [r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def render_per_patient_day_latex_table(dfs: dict[str, pd.DataFrame]) -> str:
    """Z per patient-day of both methods next to the structural minimum W_spec*M/N_pd. The heuristic's
    excess over that minimum is computed per set and only then summarized, so the data effect cancels."""
    def mean_std(values: pd.Series) -> str:
        return f"{_latex_number(values.mean())} $\\pm$ {_latex_number(values.std())}"

    lines = [
        r"\begin{table}[htbp]",
        r"    \caption{Função objetivo por paciente-dia e mínimo estrutural por porte (média $\pm$ desvio padrão "
        r"entre os 10 conjuntos)}",
        r"    \label{tab:res-z-paciente-dia}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{l*{4}{>{\centering\arraybackslash}p{3.3cm}}}",
        r"        \hline",
        r"        \textbf{Porte} & \textbf{$Z/N_{pd}$ do Gurobi} & \textbf{$Z/N_{pd}$ da heurística} & "
        r"\textbf{Mínimo estrutural ($W_{spec} \cdot M/N_{pd}$)} & "
        r"\textbf{Excedente da heurística sobre o mínimo estrutural} \\",
        r"        \hline",
    ]
    for instance, df in dfs.items():
        name = INSTANCE_DISPLAY_NAMES.get(instance, instance).capitalize()
        gurobi = df["gurobi_upper_bound"] / df["n_pd"]
        heuristic = df["heur_melhor_z"] / df["n_pd"]
        minimum = df["lb_comb"] / df["n_pd"]
        lines.append(
            f"        {name} & {mean_std(gurobi)} & {mean_std(heuristic)} & {mean_std(minimum)} & "
            f"{mean_std(heuristic - minimum)} \\\\"
        )
    lines += [
        r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"    \par\smallskip",
        r"    {\footnotesize Nota - valores por paciente-dia. O excedente é calculado em cada conjunto, como $Z/N_{pd}$ "
        r"da heurística menos o mínimo estrutural do próprio conjunto, e só então resumido por porte.}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _mean_std_latex(values: pd.Series, pct: bool = False) -> str:
    suffix = "\\%" if pct else ""
    return f"{_latex_number(values.mean())}{suffix} $\\pm$ {_latex_number(values.std())}{suffix}"


GUROBI_ONLY_SMALL_NOTE = (
    r"Gurobi reportado apenas na instância pequena. Nas demais, sua solução incumbente está distante do ótimo "
    r"e não é um termo de comparação legítimo (\autoref{sec:res-operacionais})."
)


def _criterion_table(dfs, caption, label, columns, row_values, note) -> str:
    lines = [
        r"\begin{table}[htbp]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        f"    \\begin{{tabular}}{{l*{{{len(columns)}}}{{>{{\\centering\\arraybackslash}}p{{3.0cm}}}}}}",
        r"        \hline",
        "        \\textbf{Porte} & " + " & ".join(f"\\textbf{{{c}}}" for c in columns) + " \\\\",
        r"        \hline",
    ]
    for instance, df in dfs.items():
        name = INSTANCE_DISPLAY_NAMES.get(instance, instance).capitalize()
        lines.append(f"        {name} & " + " & ".join(row_values(instance, df)) + " \\\\")
    lines += [r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"    \par\smallskip",
              f"    {{\\footnotesize Nota - {note}}}", r"\end{table}"]
    return "\n".join(lines)


def render_operational_latex_tables(dfs: dict[str, pd.DataFrame]) -> str:
    """One table per criterion; every rate is computed within each set and only then aggregated."""
    def gurobi(instance, values):
        return _mean_std_latex(values) if instance == "pequeno" else "---"

    def specialty(instance, df):
        heuristic = df["melhor_specialty_mismatch_count"] / df["n_pd"] * 100
        minimum = df["min_especialidade_inadequada"] / df["n_pd"] * 100
        exact = df["gurobi_specialty_mismatch_count"] / df["n_pd"] * 100
        return [_mean_std_latex(minimum), _mean_std_latex(heuristic), _mean_std_latex(heuristic - minimum),
                gurobi(instance, exact)]

    def transfers(instance, df):
        patients = df["n_quartos"] * 10
        heuristic = df["melhor_transfer_count"] / patients * 100
        exact = df["gurobi_transfer_count"] / patients * 100
        return [_mean_std_latex(heuristic), gurobi(instance, exact)]

    def mixed(instance, df):
        room_days = df["n_quartos"] * df["melhor_horizon_days"]
        heuristic = df["melhor_mixed_room_day_count"] / room_days * 100
        minimum = df["min_quartos_mistos"] / room_days * 100
        exact = df["gurobi_mixed_room_day_count"] / room_days * 100
        return [_mean_std_latex(minimum), _mean_std_latex(heuristic), _mean_std_latex(heuristic - minimum),
                gurobi(instance, exact)]

    tables = [
        _criterion_table(
            dfs, r"Especialidade inadequada por porte (\% dos pacientes-dia, média $\pm$ desvio padrão entre os 10 conjuntos)",
            "tab:res-op-especialidade",
            [r"Mínimo estrutural ($M/N_{pd}$)", "Heurística", "Folga da heurística (p.p.)", "Gurobi"], specialty,
            r"Folga calculada em cada conjunto, como a taxa da heurística menos o mínimo estrutural do próprio "
            r"conjunto, e só então agregada. " + GUROBI_ONLY_SMALL_NOTE),
        _criterion_table(
            dfs, r"Transferências por porte (por 100 pacientes, média $\pm$ desvio padrão entre os 10 conjuntos)",
            "tab:res-op-transferencias", ["Heurística", "Gurobi"], transfers,
            r"Não há limite inferior não trivial para o número de transferências. " + GUROBI_ONLY_SMALL_NOTE),
        _criterion_table(
            dfs, r"Quartos-dia mistos por porte (\% dos quartos-dia, média $\pm$ desvio padrão entre os 10 conjuntos)",
            "tab:res-op-mistos",
            [r"Mínimo por paridade ($G$)", "Heurística", "Folga da heurística (p.p.)", "Gurobi"], mixed,
            r"Mínimo por paridade conforme a Equação~\eqref{eq:minimo-misto}. Folga calculada em cada conjunto e só então "
            r"agregada. " + GUROBI_ONLY_SMALL_NOTE),
    ]
    return "\n\n".join(tables)


def render_instance_characteristics_latex_table(dfs: dict[str, pd.DataFrame]) -> str:
    """Properties of the generated data only (identical for any solution of a set)."""
    rows = [
        ("Horizonte (dias)", lambda df: _mean_std_latex(df["melhor_horizon_days"])),
        ("Dias com lotação total", lambda df: _mean_std_latex(df["dias_lotacao_total"])),
        ("Ocupação média (todo o horizonte)", lambda df: _mean_std_latex(df["melhor_occupancy_rate_mean"] * 100, pct=True)),
        ("Pico de ocupação diária", lambda df: _mean_std_latex(df["melhor_peak_occupancy_rate"] * 100, pct=True)),
        ("Mínimo de ocupação diária", lambda df: _mean_std_latex(df["melhor_min_occupancy_rate"] * 100, pct=True)),
    ]
    header = " & ".join(f"\\textbf{{{INSTANCE_DISPLAY_NAMES.get(i, i).capitalize()}}}" for i in dfs)
    lines = [
        r"\begin{table}[htbp]",
        r"    \caption{Características das instâncias geradas (média $\pm$ desvio padrão entre os 10 conjuntos)}",
        r"    \label{tab:caracteristicas-instancias}",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        f"    \\begin{{tabular}}{{l{'r' * len(dfs)}}}",
        r"        \hline",
        f"        \\textbf{{Característica}} & {header} \\\\",
        r"        \hline",
    ]
    for label, value in rows:
        lines.append(f"        {label} & " + " & ".join(value(df) for df in dfs.values()) + " \\\\")
    lines += [r"        \hline", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def save_combined_latex_table() -> None:
    dfs = {}
    for instance in INSTANCE_DISPLAY_NAMES:
        sets = discover_sets(instance)
        if sets:
            dfs[instance] = build_aggregate_summary(instance, sets)
    for instance, df in dfs.items():
        save_objective_latex_table(df, instance)
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

    plot_convergence_log(list(dfs))
    plot_capacity_violation_sets([i for i in ("grande", "muito_grande") if i in dfs])
    print(f"Figura de convergência em escala log em {os.path.join(AGGREGATION_DIR, 'convergence_log.png')}")

    path = os.path.join(AGGREGATION_DIR, "operational_tables.tex")
    with open(path, "w") as f:
        f.write(render_operational_latex_tables(dfs) + "\n")
    print(f"Tabelas operacionais por critério em {path}")

    path = os.path.join(AGGREGATION_DIR, "instance_characteristics_table.tex")
    with open(path, "w") as f:
        f.write(render_instance_characteristics_latex_table(dfs) + "\n")
    print(f"Tabela de características das instâncias em {path}")

    path = os.path.join(AGGREGATION_DIR, "per_patient_day_table.tex")
    with open(path, "w") as f:
        f.write(render_per_patient_day_latex_table(dfs) + "\n")
    print(f"Tabela de Z por paciente-dia em {path}")

    path = os.path.join(AGGREGATION_DIR, "sigma_viability_table.tex")
    with open(path, "w") as f:
        f.write(render_sigma_viability_latex_table(dfs) + "\n")
    print(f"Tabela de desvio padrão vs. viabilidade em {path}")

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

    plot_operational_metrics(df, instance_name, out_dir)
    plot_sets_boxplot(instance_name, sets, out_dir)

    print()
    print_and_save_summary_table(df, instance_name, out_dir)
    print()
    save_objective_latex_table(df, instance_name)
    print(f"Resultados em {out_dir}/ (aggregate_summary.csv tem a tabela crua, uma linha por conjunto)")


if __name__ == "__main__":
    main()
