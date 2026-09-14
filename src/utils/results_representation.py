"""Terminal reporting, CSV export and plots (boxplot, convergence) comparing the exact
(Gurobi) and heuristic (memetic) solvers, per Section 3.5 of the TCC. Called by main.py
once both solvers have finished running on an instance.
"""

from __future__ import annotations

import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.solution import SolverResult

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

    def report_all(self) -> None:
        self.print_summary_table()
        self.export_csv()
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

    def plot_convergence(self) -> None:
        self._save_plain_convergence_plot("convergence.png")

    def _save_plain_convergence_plot(self, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))

        padded_histories = self._padded_convergence_histories()
        for history in padded_histories:
            ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)

        average_history = [statistics.mean(values) for values in zip(*padded_histories)]
        ax.plot(average_history, color="steelblue", linewidth=2.5, label="Média das repetições")

        initial_value = average_history[0]
        final_value = average_history[-1]
        ax.scatter([0], [initial_value], color="darkorange", zorder=5, label=f"Início: {initial_value:.1f}")
        ax.scatter(
            [len(average_history) - 1], [final_value], color="seagreen", zorder=5,
            label=f"Fim: {final_value:.1f}",
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
        average_history = [statistics.mean(values) for values in zip(*padded_histories)]
        initial_value = average_history[0]
        final_value = average_history[-1]

        if len(average_history) < 2:
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

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 3]}
        )

        for ax in (ax_top, ax_bottom):
            for history in padded_histories:
                ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)
            ax.plot(average_history, color="steelblue", linewidth=2.5, label="Média das repetições")
            ax.scatter([0], [initial_value], color="darkorange", zorder=5, label=f"Início: {initial_value:.1f}")
            ax.scatter(
                [len(average_history) - 1], [final_value], color="seagreen", zorder=5,
                label=f"Fim: {final_value:.1f}",
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

        # Sorted ascending by cost for readability — color stays tied to what each bar
        # IS (Gurobi vs. a heuristic iteration), not to its position after sorting.
        labels, values, colors = zip(*sorted(zip(labels, values, colors), key=lambda row: row[1]))

        fig, ax = plt.subplots(figsize=(max(6.0, 1.0 + 0.6 * len(labels)), 5))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Custo total (função objetivo)")
        ax.set_title(f"Gurobi vs. iterações da heurística — instância {self.instance_name}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "gurobi_vs_iterations.png"), dpi=150)
        plt.close(fig)
