"""Rush-hour scenario comparing entrance-signal strategies, with plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .highway import DemandProfile, HighwayConfig, HighwaySimulation, SimulationResult
from .idm import IDMParams, fundamental_diagram
from .metering import ALINEAMetering, FixedRateMetering, MeteringController, NoMetering
from .metrics import format_table, summarize


def default_scenario(duration: float = 5400.0, seed: int = 1) -> tuple[HighwayConfig, DemandProfile, DemandProfile]:
    """Mainline near capacity plus a 20-minute ramp surge that pushes the merge over capacity."""
    cfg = HighwayConfig(duration=duration, seed=seed)
    main = DemandProfile.rush_hour(base=1250.0, peak=1600.0, t_start=600.0, t_end=1800.0)
    ramp = DemandProfile.rush_hour(base=200.0, peak=600.0, t_start=600.0, t_end=1800.0)
    return cfg, main, ramp


def default_controllers(idm: IDMParams) -> list[MeteringController]:
    fd = fundamental_diagram(idm)
    return [
        NoMetering(),
        FixedRateMetering(rate=400.0),
        ALINEAMetering(target_density=0.9 * fd["rho_crit"], gain=60.0),
    ]


def run_comparison(
    duration: float = 5400.0,
    seed: int = 1,
    controllers: list[MeteringController] | None = None,
) -> list[SimulationResult]:
    cfg, main, ramp = default_scenario(duration, seed)
    controllers = controllers or default_controllers(cfg.idm)
    results = []
    for ctrl in controllers:
        sim = HighwaySimulation(cfg, ctrl, main, ramp)
        results.append(sim.run())
    return results


def plot_comparison(results: list[SimulationResult], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cfg = results[0].config
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 11), constrained_layout=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes = np.atleast_2d(axes).reshape(3, n)

    vmax = cfg.idm.v0 * 3.6
    im = None
    for j, res in enumerate(results):
        ax = axes[0, j]
        t_min = res.times / 60.0
        im = ax.pcolormesh(
            t_min,
            res.x_cells / 1000.0,
            (res.speed_field * 3.6).T,
            cmap="RdYlGn",
            vmin=0,
            vmax=vmax,
            shading="nearest",
        )
        ax.axhline(cfg.x_ramp / 1000.0, color="k", ls="--", lw=1)
        ax.text(t_min[-1], cfg.x_ramp / 1000.0, " on-ramp", va="bottom", ha="right", fontsize=8)
        ax.set_title(res.controller, fontweight="bold")
        ax.set_xlabel("time [min]")
        ax.set_ylabel("position [km]")

        ax = axes[1, j]
        ct = res.control_times / 60.0
        rate = np.where(np.isinf(res.metering_rate), np.nan, res.metering_rate)
        ax.plot(ct, res.detector_density, color="tab:blue", label="downstream density")
        ax.set_ylabel("density [veh/km]", color="tab:blue")
        ax.set_ylim(0, 80)
        ax2 = ax.twinx()
        if np.all(np.isnan(rate)):
            ax2.text(0.5, 0.85, "signal off", transform=ax2.transAxes, ha="center", color="tab:red")
        else:
            ax2.step(ct, rate, where="post", color="tab:red", label="metering rate")
        ax2.set_ylabel("metering rate [veh/h]", color="tab:red")
        ax2.set_ylim(0, 1900)
        ax.set_xlabel("time [min]")

        ax = axes[2, j]
        ax.plot(res.times / 60.0, res.ramp_queue, color="tab:orange", label="ramp queue")
        ax.plot(res.times / 60.0, res.main_queue, color="tab:gray", label="entrance queue")
        ax.set_ylabel("waiting vehicles")
        ax.set_xlabel("time [min]")
        ax.legend(loc="upper right", fontsize=8)

    fig.colorbar(im, ax=axes[0, :].tolist(), label="speed [km/h]", shrink=0.8, pad=0.01)
    fig.suptitle("IDM highway with on-ramp: entrance-signal strategies under a rush-hour surge", fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_fundamental_diagram(results: list[SimulationResult], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = results[0].config
    fd = fundamental_diagram(cfg.idm)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(fd["density"], fd["flow"], "k-", lw=2, label="IDM equilibrium")
    ax.axvline(fd["rho_crit"], color="k", ls=":", lw=1)
    ax.text(fd["rho_crit"], fd["q_max"] * 1.02, f"  ρ_crit = {fd['rho_crit']:.0f} veh/km\n  capacity = {fd['q_max']:.0f} veh/h", fontsize=8, va="bottom")
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    lo, hi = cfg.detector_start, cfg.detector_end
    sel = (results[0].x_cells >= lo) & (results[0].x_cells < hi)
    for res, c in zip(results, colors):
        rho = res.density_field[:, sel].mean(axis=1)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(res.speed_field[:, sel], axis=1)
        q = rho * np.nan_to_num(v) * 3.6
        ax.scatter(rho, q, s=8, alpha=0.5, color=c, label=res.controller)
    ax.set_xlabel("density downstream of merge [veh/km]")
    ax.set_ylabel("flow [veh/h]")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, fd["q_max"] * 1.25)
    ax.legend(fontsize=8)
    ax.set_title("Fundamental diagram: where each strategy operates")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def run_demand_sweep(
    ramp_peaks: tuple[float, ...] = (300.0, 450.0, 600.0, 750.0, 900.0),
    duration: float = 5400.0,
    seed: int = 1,
) -> list[dict[str, float]]:
    """Total time spent for every controller across a range of ramp surge intensities.

    A fixed-rate signal is tuned for one demand level; ALINEA adapts.  The sweep
    shows how each strategy degrades as the surge grows.
    """
    rows = []
    for peak in ramp_peaks:
        cfg, main, _ = default_scenario(duration, seed)
        ramp = DemandProfile.rush_hour(base=200.0, peak=peak, t_start=600.0, t_end=1800.0)
        for ctrl in default_controllers(cfg.idm):
            res = HighwaySimulation(cfg, ctrl, main, ramp).run()
            row = summarize(res)
            row["ramp_peak"] = peak
            rows.append(row)
    return rows


def plot_demand_sweep(rows: list[dict[str, float]], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["controller"] for r in rows))
    colors = {"No metering": "tab:red", names[1]: "tab:orange", "ALINEA adaptive": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for key, ax, label in [
        ("total_time_spent", axes[0], "total time spent [veh·h]"),
        ("jam_fraction", axes[1], "congested samples (<60 km/h)"),
        ("max_ramp_queue", axes[2], "max ramp queue [veh]"),
    ]:
        for name in names:
            sel = [r for r in rows if r["controller"] == name]
            ax.plot([r["ramp_peak"] for r in sel], [r[key] for r in sel], "o-", color=colors.get(name), label=name)
        ax.set_xlabel("ramp surge demand [veh/h]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Robustness to surge intensity: fixed signal vs adaptive signal")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="traffic_idm", description="IDM highway simulation with adaptive entrance signals")
    ap.add_argument("--duration", type=float, default=5400.0, help="simulated seconds (default 5400)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("outputs"), help="output directory for figures")
    ap.add_argument("--sweep", action="store_true", help="also run the ramp-demand robustness sweep")
    args = ap.parse_args(argv)

    results = run_comparison(duration=args.duration, seed=args.seed)
    rows = [summarize(r) for r in results]
    print(format_table(rows))
    f1 = plot_comparison(results, args.out / "comparison.png")
    f2 = plot_fundamental_diagram(results, args.out / "fundamental_diagram.png")
    print(f"\nwrote {f1}\nwrote {f2}")
    if args.sweep:
        rows = run_demand_sweep(duration=args.duration, seed=args.seed)
        print()
        print("| Ramp surge [veh/h] | " + " | ".join(f"{n} [veh·h]" for n in dict.fromkeys(r["controller"] for r in rows)) + " |")
        peaks = list(dict.fromkeys(r["ramp_peak"] for r in rows))
        print("|---|" + "---|" * len(set(r["controller"] for r in rows)))
        for pk in peaks:
            vals = [r["total_time_spent"] for r in rows if r["ramp_peak"] == pk]
            print(f"| {pk:.0f} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        f3 = plot_demand_sweep(rows, args.out / "demand_sweep.png")
        print(f"wrote {f3}")
    return 0
