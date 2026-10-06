"""Summary table, CSV export and plots comparing Gurobi and the heuristic for one set.

Standalone, regenerates only the plots from saved CSVs:
    python src/utils/results_representation.py <instancia> [conjunto]
"""

from __future__ import annotations

import os
import statistics
import sys

# allows running this file directly as a script
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from utils.solution import WEIGHTS, COMPARISON_MODE, Solution, SolverResult

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
        self.plot_gurobi_bounds()
        if self.heuristic_results:
            self.plot_boxplot()
            self.plot_convergence()
            self.plot_convergence_broken_axis()
            self.plot_gurobi_vs_iterations()

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

        # Gurobi solutions are always feasible, so their best feasible value is total_cost.
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

        self.export_solution_metrics_csv()

    def export_solution_metrics_csv(self) -> None:
        """Operational metrics of the best and best feasible heuristic solutions
        (live runs only: needs the full assignment, which the CSV replay doesn't have)."""
        melhor_solution = self.best_result.solution if self.best_result is not None else None
        targets = [
            ("melhor", melhor_solution),
            ("melhor_viavel", self.best_feasible_result.best_feasible_solution if self.best_feasible_result else None),
        ]
        rows = []
        for label, solution in targets:
            if not isinstance(solution, Solution):
                continue

            peak_day, peak_patients, total_capacity = solution.peak_occupancy()
            rates = solution.occupancy_rates()
            mean_occupancy = statistics.mean(rates)
            std_occupancy = statistics.stdev(rates) if len(rates) > 1 else 0.0

            row = {
                "solucao": label,
                "horizon_days": solution.data_manager.horizon,
                "specialty_mismatch_count": solution.specialty_mismatch_count,
                "transfer_count": solution.transfer_count,
                "mixed_room_day_count": solution.mixed_room_day_count,
                "capacity_violation": solution.capacity_violation,
                "occupancy_rate_mean": mean_occupancy,
                "occupancy_rate_std": std_occupancy,
                "peak_occupancy_day": peak_day,
                "peak_occupancy_patients": peak_patients,
                "peak_occupancy_total_capacity": total_capacity,
            }
            for specialty, rate in solution.occupancy_rate_by_specialty().items():
                row[f"occupancy_rate_{specialty}"] = rate
            if label == "melhor_viavel" and isinstance(melhor_solution, Solution):
                row["mesma_que_melhor"] = solution.assignment == melhor_solution.assignment
            rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(self.output_dir, "solution_metrics.csv"), index=False)

    def plot_boxplot(self) -> None:
        # no Gurobi line: its much higher cost on large instances would squash the boxplot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([r.objective_value for r in self.heuristic_results], tick_labels=["Heurística"])
        ax.set_ylabel("Função objetivo Z")
        ax.set_title(f"Distribuição de Z entre execuções — instância {self.instance_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "boxplot.png"), dpi=150)
        plt.close(fig)

    def plot_gurobi_bounds(self) -> None:
        """Gurobi lower vs. upper bound, with the optimality gap between them."""
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

        labels = ["Limite inferior", "Limite superior"]
        values = [lower, upper]
        ax.bar(labels, values, color=["steelblue", "darkorange"], width=0.5, zorder=3)

        top = max(lower, upper)
        ax.set_ylim(0, top * 1.15 if top > 0 else 1.0)

        for x, value in enumerate(values):
            ax.text(x, value + top * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        x_min, x_max = ax.get_xlim()
        ax.hlines(
            [lower, upper], x_min, x_max, colors=["steelblue", "darkorange"],
            linestyle="--", linewidth=1.2, alpha=0.5, zorder=4,
        )
        ax.set_xlim(x_min, x_max)

        ax.plot([0.5, 0.5], [lower, upper], linestyle="--", color=status_color, linewidth=1.5, zorder=5)
        mid_y = (lower + upper) / 2
        ax.text(
            0.55, mid_y, f"gap: {gap * 100:.1f}%", ha="left", va="center",
            fontsize=11, fontweight="bold", color=status_color,
        )

        status_handle = Line2D([0], [0], color="none", label=status_label)
        ax.legend(
            handles=[status_handle], loc="upper center", bbox_to_anchor=(0.5, -0.14),
            frameon=False, fontsize=10, handlelength=0, handletextpad=0, labelcolor=status_color,
        )

        ax.set_ylabel("Função objetivo Z")
        ax.set_title(f"Gurobi: intervalo de otimalidade — instância {self.instance_name}")
        fig.savefig(os.path.join(self.output_dir, "gurobi_bounds.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _convergence_histories(self) -> list[list[float]]:
        return [list(r.convergence_history) for r in self.heuristic_results]

    def _best_result_index(self) -> int:
        return next(i for i, r in enumerate(self.heuristic_results) if r is self.best_result)

    def _global_best_feasible_point(self) -> tuple[int | None, float | None, int | None]:
        """(generation, fitness, repetition) of the best feasible individual across all repetitions."""
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

        histories = self._convergence_histories()
        for history in histories:
            ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)

        best_history = histories[self._best_result_index()]
        ax.plot(best_history, color="steelblue", linewidth=2.5, label="Melhor execução")

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
                label=f"Melhor viável: {feasible_value:.1f} (execução {feasible_iteration}, ger. {feasible_gen})",
            )

        ax.set_xlabel("Geração")
        ax.set_ylabel("Melhor fitness F encontrado até então")
        ax.set_title(f"Convergência do Algoritmo Memético — instância {self.instance_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_convergence_broken_axis(self) -> None:
        """Convergence with a broken y-axis, so the generation-0 outlier doesn't flatten the rest."""
        histories = self._convergence_histories()
        best_history = histories[self._best_result_index()]
        initial_value = best_history[0]
        final_value = best_history[-1]

        if len(best_history) < 2:
            # nothing to "zoom into" beyond the single point — fall back to the plain plot
            self._save_plain_convergence_plot("convergence_broken_axis.png")
            return

        # no Gurobi line here, for the same scale reason as the boxplot
        tail_values = [value for history in histories for value in history[1:]]
        tail_min, tail_max = min(tail_values), max(tail_values)
        tail_margin = (tail_max - tail_min) * 0.15 or tail_max * 0.05

        initial_values = [history[0] for history in histories]
        top_max = max(initial_values)

        feasible_gen, feasible_value, feasible_iteration = self._global_best_feasible_point()
        if feasible_value is not None:
            top_max = max(top_max, feasible_value)

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 3]}
        )

        for ax in (ax_top, ax_bottom):
            for history in histories:
                ax.plot(history, color="steelblue", alpha=0.3, linewidth=1)
            ax.plot(best_history, color="steelblue", linewidth=2.5, label="Melhor execução")
            ax.scatter([0], [initial_value], color="darkorange", zorder=5, label=f"Início: {initial_value:.1f}")
            ax.scatter(
                [len(best_history) - 1], [final_value], color="seagreen", zorder=5,
                label=f"Fim: {final_value:.1f}",
            )
            if feasible_gen is not None:
                ax.scatter(
                    [feasible_gen], [feasible_value], color="crimson", marker="D", zorder=6,
                    label=f"Melhor viável: {feasible_value:.1f} (execução {feasible_iteration}, ger. {feasible_gen})",
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
        ax_bottom.set_ylabel("Melhor fitness F encontrado até então")
        ax_top.set_title(f"Convergência do Algoritmo Memético — instância {self.instance_name} (eixo quebrado)")
        ax_top.legend(loc="upper right", fontsize=8)

        fig.savefig(os.path.join(self.output_dir, "convergence_broken_axis.png"), dpi=150)
        plt.close(fig)

    def plot_gurobi_vs_iterations(self) -> None:
        """Gurobi's Z vs. each heuristic execution's Z, sorted by value."""
        if self.gurobi_result is None:
            return

        labels = ["Gurobi"] + [f"Execução {i}" for i in range(1, len(self.heuristic_results) + 1)]
        values = [self.gurobi_result.objective_value] + [r.objective_value for r in self.heuristic_results]
        colors = ["darkorange"] + ["steelblue"] * len(self.heuristic_results)

        if self.best_feasible_result is not None:
            labels = labels + ["Melhor Viável"]
            values = values + [self.best_feasible_result.best_feasible_fitness]
            colors = colors + ["forestgreen"]

        labels, values, colors = zip(*sorted(zip(labels, values, colors), key=lambda row: row[1]))

        fig, ax = plt.subplots(figsize=(max(6.0, 1.0 + 0.6 * len(labels)), 5))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Função objetivo Z")
        ax.set_title(f"Gurobi vs. execuções da heurística — instância {self.instance_name}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "gurobi_vs_iterations.png"), dpi=150)
        plt.close(fig)


class _ReplayedSolution:
    """Stand-in for Solution rebuilt from a CSV row, with just what the plots need."""

    def __init__(self, specialty_cost: float, transfer_cost: float, gender_cost: float, capacity_violation: int):
        self.specialty_cost = specialty_cost
        self.transfer_cost = transfer_cost
        self.gender_cost = gender_cost
        self.capacity_violation = capacity_violation

    @property
    def capacity_cost(self) -> float:
        return WEIGHTS["W_CAP"] * self.capacity_violation

    @property
    def objective_value(self) -> float:
        return self.specialty_cost + self.transfer_cost + self.gender_cost

    @property
    def fitness(self) -> float:
        return self.objective_value + self.capacity_cost

    def sort_key(self):
        # mirrors Solution.sort_key()
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
    """Rebuilds every plot of a set from its saved CSVs, without running the solvers."""
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
    # optional 2nd arg: set number; without it, reads the flat results/<instancia>/ folder
    instance_arg = sys.argv[1]
    if len(sys.argv) > 2:
        instance_arg = f"{instance_arg}/set_{int(sys.argv[2]):02d}"
    regenerate_plots(instance_arg)
