"""Terminal reporting, CSV export and plots (boxplot, convergence) comparing the exact
(Gurobi) and heuristic (memetic) solvers, per Section 3.5 of the TCC. Called by main.py
once both solvers have finished running on an instance.

Plot generation (ResultsReporter.plot_all) is decoupled from the CSV export: main.py's
live run does both (export_csv writes the CSVs, plot_all reads live SolverResult
objects still in memory), but the CSVs alone carry everything plot_all needs, so plots
can also be rebuilt later without re-running Gurobi/the heuristic at all — see
regenerate_plots() / `python results_representation.py <instancia>` at the bottom of
this file, which replays saved CSVs back into ResultsReporter through _ReplayedSolution.
"""

from __future__ import annotations

import os
import statistics
import sys

# Lets `python utils/results_representation.py <instancia>` work run directly (see
# regenerate_plots below), not just `python -m utils.results_representation`: running
# this file as a script puts its own directory (src/utils/) on sys.path, not src/, so
# the `from utils.solution import ...` below would otherwise fail to resolve the
# `utils` package. Only applies when this IS the entry point — importing this module
# normally (main.py's `from utils.results_representation import ResultsReporter`)
# already has src/ on sys.path via main.py itself, so __package__ is set and this is a
# no-op.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from utils.solution import WEIGHTS, COMPARISON_MODE, SolverResult

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TABLE_WIDTH = 90
LABEL_WIDTH = 33
COLUMN_WIDTH = 16


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


class ResultsReporter:
    def __init__(
        self,
        instance_name: str,
        gurobi_result: SolverResult | None,
        heuristic_results: list[SolverResult],
    ):
        self.instance_name = instance_name
        self.gurobi_result = gurobi_result
        self.heuristic_results = heuristic_results
        self.output_dir = os.path.join(RESULTS_DIR, instance_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_result = min(heuristic_results, key=lambda r: r.solution.sort_key()) if heuristic_results else None

        feasible_results = [r for r in heuristic_results if r.best_feasible_fitness is not None]
        self.best_feasible_result = (
            min(feasible_results, key=lambda r: r.best_feasible_fitness) if feasible_results else None
        )

    def report_all(self) -> None:
        self.print_summary_table()
        self.export_csv()
        self.plot_all()

    def plot_all(self) -> None:
        """Every plot this reporter can produce, from whatever gurobi_result/
        heuristic_results it was built with — either the live SolverResult objects
        from a just-finished run (report_all's path) or ones replayed from saved CSVs
        (regenerate_plots' path). The single place both paths call into, so there's
        only one definition of "all the plots" to keep in sync."""
        self.plot_gurobi_bounds()
        if self.heuristic_results:
            self.plot_boxplot()
            self.plot_convergence()
            self.plot_convergence_broken_axis()
            self.plot_gurobi_vs_iterations()

    # -- terminal table --

    def _row(self, label: str, gurobi_value: str, best_value: str, mean_std_value: str) -> str:
        return (
            f"{label:<{LABEL_WIDTH}}| {gurobi_value:<{COLUMN_WIDTH}}| "
            f"{best_value:<{COLUMN_WIDTH}}| {mean_std_value}"
        )

    def _cost_row(self, label: str, attr: str) -> str:
        gurobi_value = f"{getattr(self.gurobi_result, attr):.1f}" if self.gurobi_result else "N/A"
        best_value = f"{getattr(self.best_result, attr):.1f}" if self.best_result else "N/A"
        values = [getattr(r, attr) for r in self.heuristic_results]
        mean, std = _mean_std(values)
        mean_std_value = f"{mean:.1f} (±{std:.1f})" if values else "N/A"
        return self._row(label, gurobi_value, best_value, mean_std_value)

    def print_summary_table(self) -> None:
        print("=" * TABLE_WIDTH)
        print("CONSOLIDADO EXPERIMENTAL (GUROBI VS HEURÍSTICA)".center(TABLE_WIDTH))
        print("=" * TABLE_WIDTH)
        print(self._row("Métrica / Tipo de Custo", "Gurobi", "Heur. (Melhor)", "Heur. (Média ± DP)"))
        print("-" * TABLE_WIDTH)
        print(self._cost_row("Custo Clínico (Especialidades)", "specialty_cost"))
        print(self._cost_row("Penalidades por Transferência", "transfer_cost"))
        print(self._cost_row("Penalidades por Quarto Misto", "gender_cost"))
        print(self._cost_row("Penalidades por Excesso de Leito", "capacity_cost"))
        print("-" * TABLE_WIDTH)

        gurobi_initial = "NULL"
        best_initial = f"{self.best_result.initial_fitness:.1f}" if self.best_result else "N/A"
        initial_values = [r.initial_fitness for r in self.heuristic_results if r.initial_fitness is not None]
        mean, std = _mean_std(initial_values)
        mean_initial = f"{mean:.1f} (±{std:.1f})" if initial_values else "N/A"
        print(self._row("Custo Inicial (Geração 00)", gurobi_initial, best_initial, mean_initial))

        print(self._cost_row("CUSTO TOTAL (FUNÇÃO OBJETIVO)", "total_cost"))
        print("-" * TABLE_WIDTH)

        # Gurobi's solution is always capacity-feasible by construction (hard
        # constraint, Eq. 3.8), so its own total_cost doubles as its "best feasible"
        # value here. For the heuristic, this is best_feasible_fitness (see
        # SolverResult) — the best capacity_violation == 0 individual ever seen during
        # search, which can differ from "Heur. (Melhor)" above since that one is
        # chosen by sort_key/fitness_only and may itself be infeasible.
        gurobi_feasible = f"{self.gurobi_result.total_cost:.1f}" if self.gurobi_result else "N/A"
        best_feasible_value = (
            f"{self.best_feasible_result.best_feasible_fitness:.1f}" if self.best_feasible_result else "N/A"
        )
        feasible_values = [r.best_feasible_fitness for r in self.heuristic_results if r.best_feasible_fitness is not None]
        mean, std = _mean_std(feasible_values)
        mean_feasible_value = f"{mean:.1f} (±{std:.1f})" if feasible_values else "N/A"
        print(self._row("Melhor Solução Viável (Heur.)", gurobi_feasible, best_feasible_value, mean_feasible_value))
        print("-" * TABLE_WIDTH)

        gurobi_runtime = f"{self.gurobi_result.runtime:.4f} s" if self.gurobi_result else "N/A"
        best_runtime = f"{self.best_result.runtime:.4f} s" if self.best_result else "N/A"
        runtimes = [r.runtime for r in self.heuristic_results]
        mean, std = _mean_std(runtimes)
        mean_runtime = f"{mean:.4f} s (±{std:.4f} s)" if runtimes else "N/A"
        print(self._row("Tempo de Processamento", gurobi_runtime, best_runtime, mean_runtime))

        gurobi_evals = str(self.gurobi_result.evaluation_count) if self.gurobi_result else "N/A"
        best_evals = str(self.best_result.evaluation_count) if self.best_result else "N/A"
        eval_values = [r.evaluation_count for r in self.heuristic_results if r.evaluation_count is not None]
        mean, std = _mean_std(eval_values)
        mean_evals = f"{mean:.1f} (±{std:.1f})" if eval_values else "N/A"
        print(self._row("Número de Comparações (Custo)", gurobi_evals, best_evals, mean_evals))
        print("=" * TABLE_WIDTH)

    # -- CSV export --

    def export_csv(self) -> None:
        rows = []
        for i, result in enumerate(self.heuristic_results, start=1):
            rows.append({
                "iteration": i,
                "specialty_cost": result.specialty_cost,
                "transfer_cost": result.transfer_cost,
                "gender_cost": result.gender_cost,
                "capacity_cost": result.capacity_cost,
                "initial_fitness": result.initial_fitness,
                "total_cost": result.total_cost,
                "runtime_seconds": round(result.runtime),
                "evaluation_count": result.evaluation_count,
                "best_feasible_fitness": result.best_feasible_fitness,
                "best_feasible_generation": result.best_feasible_generation,
            })
        pd.DataFrame(rows).to_csv(os.path.join(self.output_dir, "heuristic_runs.csv"), index=False)

        if self.gurobi_result is not None:
            pd.DataFrame([{
                "specialty_cost": self.gurobi_result.specialty_cost,
                "transfer_cost": self.gurobi_result.transfer_cost,
                "gender_cost": self.gurobi_result.gender_cost,
                "capacity_cost": self.gurobi_result.capacity_cost,
                "total_cost": self.gurobi_result.total_cost,
                "runtime_seconds": round(self.gurobi_result.runtime),
                "evaluation_count": self.gurobi_result.evaluation_count,
                "upper_bound": self.gurobi_result.upper_bound,
                "lower_bound": self.gurobi_result.lower_bound,
                "gap": self.gurobi_result.gap,
                "is_optimal": self.gurobi_result.is_optimal,
            }]).to_csv(os.path.join(self.output_dir, "gurobi_run.csv"), index=False)

        convergence_rows = []
        for i, result in enumerate(self.heuristic_results, start=1):
            for generation, fitness in enumerate(result.convergence_history):
                convergence_rows.append({"iteration": i, "generation": generation, "best_fitness": fitness})
        pd.DataFrame(convergence_rows).to_csv(os.path.join(self.output_dir, "convergence.csv"), index=False)

    # -- plots --

    def plot_boxplot(self) -> None:
        # No Gurobi reference line here: for instances where Gurobi's total cost is
        # several times the heuristic's (grande, muito_grande), a horizontal line at
        # that value forces the y-axis to stretch to include it, compressing the actual
        # boxplot into a sliver. The terminal table and CSVs already report the Gurobi
        # value directly, so nothing is lost by leaving it out of this plot.
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([r.total_cost for r in self.heuristic_results], tick_labels=["Heurística"])
        ax.set_ylabel("Custo total (função objetivo)")
        ax.set_title(f"Distribuição do custo final — instância {self.instance_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "boxplot.png"), dpi=150)
        plt.close(fig)

    def plot_gurobi_bounds(self) -> None:
        """Two vertical bars — Lower Bound (ObjBound, the proven floor) and Upper
        Bound (ObjVal, the incumbent — same value as gurobi_result.total_cost
        elsewhere) — each capped by its own dashed reference line spanning the full
        plot width (rather than one diagonal line between them), plus a vertical
        dashed connector between the bars labeled with the relative gap. Doesn't
        depend on heuristic_results, so it's generated even for a Gurobi-only run.
        Optimality status (is_optimal) goes in a legend below the axes, clear of the
        bars/labels; when is_optimal, both bars/reference lines coincide (gap == 0),
        which is the correct picture rather than a special case to hide."""
        if self.gurobi_result is None:
            return

        lower = self.gurobi_result.lower_bound
        upper = self.gurobi_result.upper_bound
        gap = self.gurobi_result.gap
        is_optimal = self.gurobi_result.is_optimal
        if lower is None or upper is None:
            return

        status_color = "forestgreen" if is_optimal else "firebrick"
        status_label = "✓ Ótimo provado" if is_optimal else "⚠ Ótimo NÃO provado (bateu o teto de tempo)"

        fig, ax = plt.subplots(figsize=(6, 5))

        labels = ["Lower Bound", "Upper Bound"]
        values = [lower, upper]
        ax.bar(labels, values, color=["steelblue", "darkorange"], width=0.5, zorder=3)

        # Headroom above the taller bar so its value label and the gap label have
        # room without getting clipped.
        top = max(lower, upper)
        ax.set_ylim(0, top * 1.15 if top > 0 else 1.0)

        for x, value in enumerate(values):
            ax.text(x, value + top * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # A faint dashed reference line capping each bar's own height, colored to
        # match that bar (not the status color) and spanning the full plot width, so
        # the two heights read directly off a shared horizontal reference instead of
        # one diagonal line between two points. Kept low-opacity and bar-colored, not
        # status-colored, so red/green stays reserved for the actual gap indicator
        # below (the vertical connector + label) instead of competing with it.
        x_min, x_max = ax.get_xlim()
        ax.hlines(
            [lower, upper], x_min, x_max, colors=["steelblue", "darkorange"],
            linestyle="--", linewidth=1.2, alpha=0.5, zorder=4,
        )
        ax.set_xlim(x_min, x_max)

        # Vertical connector between the two reference lines, between the bars,
        # labeled with the gap — this is what actually shows the gap's size, and the
        # only dashed element that carries the status color.
        ax.plot([0.5, 0.5], [lower, upper], linestyle="--", color=status_color, linewidth=1.5, zorder=5)
        mid_y = (lower + upper) / 2
        ax.text(
            0.55, mid_y, f"gap: {gap * 100:.1f}%", ha="left", va="center",
            fontsize=11, fontweight="bold", color=status_color,
        )

        # No line swatch in front of the text: a blank/invisible handle
        # (handlelength=0) with the label colored directly (labelcolor) reads as
        # plain colored status text, not a legend entry for a line that's drawn
        # nowhere near it.
        status_handle = Line2D([0], [0], color="none", label=status_label)
        ax.legend(
            handles=[status_handle], loc="upper center", bbox_to_anchor=(0.5, -0.14),
            frameon=False, fontsize=10, handlelength=0, handletextpad=0, labelcolor=status_color,
        )

        ax.set_ylabel("Custo total (função objetivo)")
        ax.set_title(f"Gurobi: intervalo de otimalidade — instância {self.instance_name}")
        fig.savefig(os.path.join(self.output_dir, "gurobi_bounds.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _padded_convergence_histories(self) -> list[list[float]]:
        """Pads every repetition's convergence_history to the same length by holding its
        last value constant, so they can be averaged/plotted generation-by-generation
        even when repetitions ran different numbers of generations before time ran out."""
        max_len = max(len(r.convergence_history) for r in self.heuristic_results)
        padded = []
        for result in self.heuristic_results:
            history = list(result.convergence_history)
            if len(history) < max_len:
                history = history + [history[-1]] * (max_len - len(history))
            padded.append(history)
        return padded

    def _best_result_index(self) -> int:
        """Index (within self.heuristic_results / _padded_convergence_histories) of the
        repetition with the best sort_key() — the same repetition reported as
        "Heur. (Melhor)" in the summary table."""
        return next(i for i, r in enumerate(self.heuristic_results) if r is self.best_result)

    def _global_best_feasible_point(self) -> tuple[int | None, float | None, int | None]:
        """(generation, fitness, iteration) of the best capacity-feasible individual
        found across ALL heuristic repetitions (self.best_feasible_result — same value
        as "Melhor Solução Viável (Heur.)" in the summary table/CSV). Deliberately NOT
        scoped to the bolded "melhor repetição" curve (chosen by sort_key/fitness_only,
        see COMPARISON_MODE in solution.py): the repetition that wins overall isn't
        necessarily the one that stumbled onto the best feasible individual, so the
        marker can legitimately land off that curve, on a fainter one. `iteration` (the
        1-based repetition number, matching heuristic_runs.csv) lets the plot label
        which run it came from when that happens. (None, None, None) if no repetition
        ever found a feasible individual."""
        if self.best_feasible_result is None:
            return None, None, None
        iteration = next(
            i for i, r in enumerate(self.heuristic_results, start=1) if r is self.best_feasible_result
        )
        return (
            self.best_feasible_result.best_feasible_generation,
            self.best_feasible_result.best_feasible_fitness,
            iteration,
        )

    def plot_convergence(self) -> None:
        self._save_plain_convergence_plot("convergence.png")

    def _save_plain_convergence_plot(self, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))

        padded_histories = self._padded_convergence_histories()
        for history in padded_histories:
            ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)

        best_history = padded_histories[self._best_result_index()]
        ax.plot(best_history, color="steelblue", linewidth=2.5, label="Melhor repetição")

        initial_value = best_history[0]
        final_value = best_history[-1]
        ax.scatter([0], [initial_value], color="darkorange", zorder=5, label=f"Início: {initial_value:.1f}")
        ax.scatter(
            [len(best_history) - 1], [final_value], color="seagreen", zorder=5,
            label=f"Fim: {final_value:.1f}",
        )

        feasible_gen, feasible_value, feasible_iteration = self._global_best_feasible_point()
        if feasible_gen is not None:
            ax.scatter(
                [feasible_gen], [feasible_value], color="crimson", marker="D", zorder=6,
                label=f"Melhor viável: {feasible_value:.1f} (rodada {feasible_iteration}, ger. {feasible_gen})",
            )

        ax.set_xlabel("Geração")
        ax.set_ylabel("Melhor custo encontrado até então")
        ax.set_title(f"Convergência do Algoritmo Memético — instância {self.instance_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_convergence_broken_axis(self) -> None:
        """Two stacked panels sharing the x-axis: a thin top panel isolates the
        generation-0 outlier, a large bottom panel zooms in (linearly) on the range
        where the actual convergence detail and inter-repetition spread live. Fixes what
        log-scale couldn't: a single huge outlier point compressing the interesting
        range into a few pixels."""
        padded_histories = self._padded_convergence_histories()
        best_history = padded_histories[self._best_result_index()]
        initial_value = best_history[0]
        final_value = best_history[-1]

        if len(best_history) < 2:
            # nothing to "zoom into" beyond the single point — fall back to the plain plot
            self._save_plain_convergence_plot("convergence_broken_axis.png")
            return

        # No Gurobi reference line/bound here: including its value (often several times
        # the heuristic's, on grande/muito_grande) in the tail range would re-introduce
        # the exact scale-compression problem this broken-axis plot exists to fix.
        tail_values = [value for history in padded_histories for value in history[1:]]
        tail_min, tail_max = min(tail_values), max(tail_values)
        tail_margin = (tail_max - tail_min) * 0.15 or tail_max * 0.05

        initial_values = [history[0] for history in padded_histories]
        top_min, top_max = min(initial_values), max(initial_values)

        feasible_gen, feasible_value, feasible_iteration = self._global_best_feasible_point()
        if feasible_value is not None:
            # Unlike the excluded Gurobi reference line (see the comment above), this
            # marker is one repetition's own data point, not an external scale — so
            # it's safe (and necessary) to extend the top panel to keep it visible
            # instead of leaving it out.
            top_max = max(top_max, feasible_value)

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 3]}
        )

        for ax in (ax_top, ax_bottom):
            for history in padded_histories:
                ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)
            ax.plot(best_history, color="steelblue", linewidth=2.5, label="Melhor repetição")
            ax.scatter([0], [initial_value], color="darkorange", zorder=5, label=f"Início: {initial_value:.1f}")
            ax.scatter(
                [len(best_history) - 1], [final_value], color="seagreen", zorder=5,
                label=f"Fim: {final_value:.1f}",
            )
            if feasible_gen is not None:
                ax.scatter(
                    [feasible_gen], [feasible_value], color="crimson", marker="D", zorder=6,
                    label=f"Melhor viável: {feasible_value:.1f} (rodada {feasible_iteration}, ger. {feasible_gen})",
                )

        ax_top.set_ylim(tail_max + tail_margin, top_max * 1.05)
        ax_bottom.set_ylim(max(0.0, tail_min - tail_margin), tail_max + tail_margin)

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()

        break_marker = dict(
            marker=[(-1, -0.5), (1, 0.5)], markersize=12, linestyle="none",
            color="k", mec="k", mew=1, clip_on=False,
        )
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_marker)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **break_marker)

        ax_bottom.set_xlabel("Geração")
        ax_bottom.set_ylabel("Melhor custo encontrado até então")
        ax_top.set_title(f"Convergência do Algoritmo Memético — instância {self.instance_name} (eixo quebrado)")
        ax_top.legend(loc="upper right", fontsize=8)

        fig.savefig(os.path.join(self.output_dir, "convergence_broken_axis.png"), dpi=150)
        plt.close(fig)

    def plot_gurobi_vs_iterations(self) -> None:
        """Bar chart directly comparing Gurobi's total cost against each heuristic
        iteration's (one full independent run of the algorithm — same "iteration"
        used as the CSV column name, not a GA generation) — the comparison the dashed
        reference line used to show on the convergence/boxplot charts, before it was
        removed for distorting their shared y-axis (see
        plot_boxplot/plot_convergence_broken_axis). A bar chart doesn't have that
        problem: a tall Gurobi bar next to shorter heuristic bars, each with its own
        independent height, IS the comparison being shown, not an artifact to work
        around like it was on a shared-axis line plot."""
        if self.gurobi_result is None:
            return

        labels = ["Gurobi"] + [f"Iteração {i}" for i in range(1, len(self.heuristic_results) + 1)]
        values = [self.gurobi_result.total_cost] + [r.total_cost for r in self.heuristic_results]
        colors = ["darkorange"] + ["steelblue"] * len(self.heuristic_results)

        # Extra bar for the best capacity-feasible individual found across ALL
        # repetitions (self.best_feasible_result — same value as "Melhor Solução
        # Viável (Heur.)" in the summary table/CSV), distinct from whichever
        # iteration's own final result happens to win by sort_key/fitness_only.
        # Skipped if no repetition ever found a feasible individual.
        if self.best_feasible_result is not None:
            labels = labels + ["Melhor Viável"]
            values = values + [self.best_feasible_result.best_feasible_fitness]
            colors = colors + ["forestgreen"]

        # Sorted ascending by cost for readability — color stays tied to what each bar
        # IS (Gurobi vs. a heuristic iteration vs. the best-viável reference), not to
        # its position after sorting.
        labels, values, colors = zip(*sorted(zip(labels, values, colors), key=lambda row: row[1]))

        fig, ax = plt.subplots(figsize=(max(6.0, 1.0 + 0.6 * len(labels)), 5))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Custo total (função objetivo)")
        ax.set_title(f"Gurobi vs. iterações da heurística — instância {self.instance_name}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "gurobi_vs_iterations.png"), dpi=150)
        plt.close(fig)


# -- standalone plot regeneration, from saved CSVs, without re-running Gurobi/the
# heuristic (see module docstring) --


class _ReplayedSolution:
    """Minimal stand-in for utils.solution.Solution, built from a saved CSV row
    instead of a live assignment. Exposes exactly what SolverResult's properties and
    ResultsReporter's plots read from a Solution (specialty_cost, transfer_cost,
    gender_cost, capacity_cost, fitness, sort_key()) so regenerate_plots() can drive
    ResultsReporter without changing anything about how it plots."""

    def __init__(self, specialty_cost: float, transfer_cost: float, gender_cost: float, capacity_violation: int):
        self.specialty_cost = specialty_cost
        self.transfer_cost = transfer_cost
        self.gender_cost = gender_cost
        self.capacity_violation = capacity_violation

    @property
    def capacity_cost(self) -> float:
        return WEIGHTS["W_CAP"] * self.capacity_violation

    @property
    def fitness(self) -> float:
        return self.specialty_cost + self.transfer_cost + self.gender_cost + self.capacity_cost

    def sort_key(self):
        # Mirrors Solution.sort_key() (solution.py) exactly, via the same
        # COMPARISON_MODE constant, so a replayed ranking matches what the live run
        # would have produced regardless of which mode was active.
        if COMPARISON_MODE == "fitness_only":
            return self.fitness
        return (self.capacity_violation, self.fitness)


def _none_or_float(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _none_or_int(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _replayed_solution_from_row(row: pd.Series) -> _ReplayedSolution:
    capacity_violation = round(row["capacity_cost"] / WEIGHTS["W_CAP"]) if WEIGHTS["W_CAP"] else 0
    return _ReplayedSolution(
        specialty_cost=row["specialty_cost"],
        transfer_cost=row["transfer_cost"],
        gender_cost=row["gender_cost"],
        capacity_violation=capacity_violation,
    )


def _load_gurobi_result(output_dir: str) -> SolverResult | None:
    path = os.path.join(output_dir, "gurobi_run.csv")
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0]
    return SolverResult(
        solution=_replayed_solution_from_row(row),
        runtime=row["runtime_seconds"],
        evaluation_count=_none_or_int(row.get("evaluation_count")),
        upper_bound=_none_or_float(row.get("upper_bound")),
        lower_bound=_none_or_float(row.get("lower_bound")),
        gap=_none_or_float(row.get("gap")),
        is_optimal=None if pd.isna(row.get("is_optimal")) else bool(row["is_optimal"]),
    )


def _load_heuristic_results(output_dir: str) -> list[SolverResult]:
    heuristic_path = os.path.join(output_dir, "heuristic_runs.csv")
    if not os.path.exists(heuristic_path):
        return []
    heuristic_df = pd.read_csv(heuristic_path)

    convergence_by_iteration: dict[int, list[float]] = {}
    convergence_path = os.path.join(output_dir, "convergence.csv")
    if os.path.exists(convergence_path):
        convergence_df = pd.read_csv(convergence_path)
        for iteration, group in convergence_df.groupby("iteration"):
            convergence_by_iteration[iteration] = group.sort_values("generation")["best_fitness"].tolist()

    results = []
    for _, row in heuristic_df.iterrows():
        results.append(SolverResult(
            solution=_replayed_solution_from_row(row),
            runtime=row["runtime_seconds"],
            initial_fitness=_none_or_float(row.get("initial_fitness")),
            evaluation_count=_none_or_int(row.get("evaluation_count")),
            convergence_history=convergence_by_iteration.get(row["iteration"], []),
            best_feasible_fitness=_none_or_float(row.get("best_feasible_fitness")),
            best_feasible_generation=_none_or_int(row.get("best_feasible_generation")),
        ))
    return results


def regenerate_plots(instance_name: str) -> None:
    """Rebuilds every plot (plot_all) for an instance from its already-saved CSVs in
    results/<instance_name>/ — gurobi_run.csv, heuristic_runs.csv, convergence.csv —
    without re-running Gurobi or the heuristic. Requires main.py to have run at least
    once for this instance already (the CSVs must exist); raises if neither result CSV
    is found."""
    output_dir = os.path.join(RESULTS_DIR, instance_name)
    gurobi_result = _load_gurobi_result(output_dir)
    heuristic_results = _load_heuristic_results(output_dir)

    if gurobi_result is None and not heuristic_results:
        raise FileNotFoundError(
            f"Nenhum CSV de resultado encontrado em {output_dir} — rode "
            f"'python main.py {instance_name} <repetições>' pelo menos uma vez antes "
            f"de regenerar os gráficos."
        )

    reporter = ResultsReporter(instance_name, gurobi_result, heuristic_results)
    reporter.plot_all()
    print(f"Gráficos regenerados a partir dos CSVs em {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python results_representation.py <instancia> [conjunto]")
        sys.exit(1)
    # Optional 2nd arg mirrors main.py's set numbering (data_base/<instancia>/set_NN/,
    # results/<instancia>/set_NN/); omit it to target a flat results/<instancia>/
    # folder directly, e.g. one predating the per-set layout.
    instance_arg = sys.argv[1]
    if len(sys.argv) > 2:
        instance_arg = f"{instance_arg}/set_{int(sys.argv[2]):02d}"
    regenerate_plots(instance_arg)
