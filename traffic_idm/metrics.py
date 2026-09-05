"""Performance metrics for comparing entrance-signal strategies."""

from __future__ import annotations

import numpy as np

from .highway import SimulationResult


def summarize(res: SimulationResult, jam_speed: float = 60.0 / 3.6) -> dict[str, float]:
    """Key performance indicators of a run.

    * ``throughput``     vehicles leaving the highway per hour of simulation
    * ``mean_speed``     space-mean speed over all snapshots [km/h]
    * ``jam_fraction``   share of occupied (time, cell) samples slower than ``jam_speed`` [m/s]
    * ``mean_travel_time`` average time in system per exited vehicle, queues included [s]
    * ``total_time_spent`` sum of time in system over exited vehicles [veh·h]
    * ``max_ramp_queue`` / ``mean_ramp_queue`` vehicles waiting at the entrance signal
    * ``forced_merges`` number of merges that violated the comfortable-deceleration bound
    """
    hours = res.config.duration / 3600.0
    occupied = ~np.isnan(res.speed_field)
    speeds = res.speed_field[occupied]
    mean_speed = float(np.nanmean(res.speed_field)) * 3.6 if speeds.size else float("nan")
    jam_fraction = float(np.mean(speeds < jam_speed)) if speeds.size else 0.0
    return {
        "controller": res.controller,
        "throughput": float(res.exit_times.size / hours),
        "mean_speed": mean_speed,
        "jam_fraction": jam_fraction,
        "mean_travel_time": float(res.travel_times.mean()) if res.travel_times.size else float("nan"),
        "total_time_spent": float(res.travel_times.sum() / 3600.0),
        "mean_ramp_travel_time": float(res.travel_times[res.origins == 1].mean()) if np.any(res.origins == 1) else float("nan"),
        "max_ramp_queue": float(res.ramp_queue.max()) if res.ramp_queue.size else 0.0,
        "mean_ramp_queue": float(res.ramp_queue.mean()) if res.ramp_queue.size else 0.0,
        "max_main_queue": float(res.main_queue.max()) if res.main_queue.size else 0.0,
        "forced_merges": float(res.forced_merges),
    }


def format_table(rows: list[dict[str, float]]) -> str:
    cols = [
        ("controller", "Controller", "{}"),
        ("throughput", "Throughput [veh/h]", "{:.0f}"),
        ("mean_speed", "Mean speed [km/h]", "{:.1f}"),
        ("jam_fraction", "Congested (<60 km/h)", "{:.1%}"),
        ("mean_travel_time", "Mean travel time [s]", "{:.0f}"),
        ("total_time_spent", "Total time spent [veh·h]", "{:.1f}"),
        ("max_ramp_queue", "Max ramp queue [veh]", "{:.0f}"),
        ("forced_merges", "Forced merges", "{:.0f}"),
    ]
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt.format(r[k]) for k, _, fmt in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])
