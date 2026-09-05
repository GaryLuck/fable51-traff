"""Single-lane highway with an on-ramp, driven by the IDM.

Layout (positions in metres, traffic flows toward +x)::

    x=0 ────────────────── x_ramp ─────────── detector ─────── x=length
    upstream demand         merge point        density probe        exit

* Mainline vehicles arrive at ``x=0`` following a (time-varying) Poisson
  process.  If the entrance is blocked they wait in a virtual queue.
* Ramp vehicles arrive in a ramp queue; the entrance signal releases them at
  the controller's metering rate.  A released vehicle enters the mainline at
  ``x_ramp`` as soon as a gap it can safely use exists.  If it has waited more
  than ``merge_patience`` it forces its way in, braking the follower hard –
  this is the perturbation that seeds stop-and-go waves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .idm import IDMParams, idm_acceleration
from .metering import MeteringController, NoMetering


@dataclass
class DemandProfile:
    """Piecewise-linear demand in veh/h given by (time [s], demand) knots."""

    times: tuple[float, ...]
    values: tuple[float, ...]

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.times, self.values))

    @staticmethod
    def constant(value: float) -> "DemandProfile":
        return DemandProfile((0.0,), (value,))

    @staticmethod
    def rush_hour(base: float, peak: float, t_start: float, t_end: float, ramp: float = 120.0) -> "DemandProfile":
        """Demand rising from ``base`` to ``peak`` between ``t_start`` and ``t_end``."""
        return DemandProfile(
            (0.0, t_start, t_start + ramp, t_end - ramp, t_end),
            (base, base, peak, peak, base),
        )


@dataclass
class HighwayConfig:
    length: float = 6000.0  # m
    x_ramp: float = 3000.0  # merge point [m]
    detector_start: float = 3100.0  # density probe segment downstream of the merge
    detector_end: float = 3400.0
    dt: float = 0.1  # integration step [s]
    duration: float = 3600.0  # s
    accel_length: float = 250.0  # acceleration lane length downstream of x_ramp [m]
    merge_speed: float = 30.0  # top speed a ramp vehicle reaches on the acceleration lane [m/s]
    merge_decel: float = 3.0  # max deceleration a merge may impose on the follower [m/s^2]
    forced_merge_decel: float = 9.0  # same bound once the merger loses patience [m/s^2]
    forced_slot: float = 0.25  # fraction of the equilibrium spacing a forced merger needs on each side
    merge_patience: float = 6.0  # s before a waiting ramp vehicle forces a merge
    record_every: float = 10.0  # s between space-time snapshots
    cell: float = 100.0  # m spatial resolution of the snapshots
    seed: int = 1
    idm: IDMParams = field(default_factory=IDMParams)


@dataclass
class SimulationResult:
    controller: str
    config: HighwayConfig
    times: np.ndarray  # snapshot times [s]
    x_cells: np.ndarray  # cell centres [m]
    speed_field: np.ndarray  # mean speed per (time, cell), nan when empty [m/s]
    density_field: np.ndarray  # veh/km per (time, cell)
    control_times: np.ndarray
    metering_rate: np.ndarray  # veh/h per control interval
    detector_density: np.ndarray  # veh/km per control interval
    ramp_queue: np.ndarray  # vehicles waiting at the ramp per snapshot
    main_queue: np.ndarray  # vehicles waiting at the upstream entrance per snapshot
    exit_times: np.ndarray  # times at which vehicles left the highway
    travel_times: np.ndarray  # total time in system (queue + travel) of exited vehicles [s]
    origins: np.ndarray  # 0 = mainline, 1 = ramp, for exited vehicles
    forced_merges: int
    vehicles_entered: int


class HighwaySimulation:
    def __init__(
        self,
        config: HighwayConfig,
        controller: MeteringController | None = None,
        main_demand: DemandProfile | None = None,
        ramp_demand: DemandProfile | None = None,
    ) -> None:
        self.cfg = config
        self.p = config.idm
        self.controller = controller or NoMetering()
        self.main_demand = main_demand or DemandProfile.constant(1400.0)
        self.ramp_demand = ramp_demand or DemandProfile.constant(300.0)
        self.rng = np.random.default_rng(config.seed)

        # Vehicle state, sorted by position ascending (last element is the leader).
        self.x = np.empty(0)
        self.v = np.empty(0)
        self.origin = np.empty(0, dtype=int)
        self.t_arrive = np.empty(0)

        self.main_queue: list[float] = []  # arrival times waiting at x=0
        self.ramp_queue: list[float] = []  # arrival times waiting at the signal
        self.released: list[float] | None = None  # [arrival time, release time] of the vehicle on the acceleration lane
        self.forced_merges = 0
        self.entered = 0

        self._exit_t: list[float] = []
        self._travel: list[float] = []
        self._origins: list[int] = []

    # ------------------------------------------------------------------ helpers
    def _gap_and_dv(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.x.size
        s = np.full(n, np.inf)
        dv = np.zeros(n)
        if n > 1:
            s[:-1] = self.x[1:] - self.x[:-1] - self.p.length
            dv[:-1] = self.v[:-1] - self.v[1:]
        return s, dv

    def _insert(self, x: float, v: float, origin: int, t_arrive: float) -> None:
        i = int(np.searchsorted(self.x, x))
        self.x = np.insert(self.x, i, x)
        self.v = np.insert(self.v, i, v)
        self.origin = np.insert(self.origin, i, origin)
        self.t_arrive = np.insert(self.t_arrive, i, t_arrive)
        self.entered += 1

    def _safe_speed_behind(self, gap: float, v_lead: float) -> float:
        """Largest speed with which a vehicle can occupy ``gap`` behind a leader without an IDM emergency."""
        # equilibrium-like condition s >= s0 + v*T  → v <= (s - s0)/T, also never faster than leader + margin
        return max(0.0, min(self.p.v0, (gap - self.p.s0) / self.p.T, v_lead + 5.0))

    def _try_upstream_entry(self, t: float) -> None:
        while self.main_queue:
            if self.x.size:
                gap = self.x[0] - self.p.length
                v_new = min(self.p.v0, self.v[0])
                if gap < self.p.s0 + v_new * self.p.T:
                    return  # entrance blocked; the vehicle waits in the virtual queue
            else:
                v_new = self.p.v0
            self._insert(0.0, v_new, 0, self.main_queue.pop(0))

    def _try_ramp_merge(self, t: float) -> None:
        """Let the vehicle on the acceleration lane merge into the best gap it can find.

        Candidate gaps are all leader/follower pairs overlapping the acceleration
        lane ``[x_ramp, x_ramp + accel_length]``.  The merger enters mid-gap at a
        speed adapted to its neighbours.  A merge is accepted when neither the
        merger nor its new follower needs to brake harder than ``merge_decel``;
        after ``merge_patience`` seconds the bound relaxes to ``forced_merge_decel``
        and the required gaps shrink to ``forced_slot`` of the equilibrium spacing.
        """
        if self.released is None:
            return
        t_arr, t_rel = self.released
        p, cfg = self.p, self.cfg
        x0, x1 = cfg.x_ramp, cfg.x_ramp + cfg.accel_length
        forced = (t - t_rel) > cfg.merge_patience
        limit = -(cfg.forced_merge_decel if forced else cfg.merge_decel)
        slot = cfg.forced_slot if forced else 0.5

        n = self.x.size
        if n == 0:
            self._insert(x0, cfg.merge_speed, 1, t_arr)
            self.released = None
            return

        # leader candidates: every vehicle whose gap behind it overlaps the lane
        i_lo = int(np.searchsorted(self.x, x0))
        i_hi = int(np.searchsorted(self.x, x1, side="right"))
        leaders = np.arange(i_lo, min(i_hi + 1, n + 1))  # index n means "no leader"
        best, best_margin = None, -np.inf
        for i in leaders:
            x_lead = self.x[i] if i < n else np.inf
            x_foll = self.x[i - 1] if i > 0 else -np.inf
            v_lead = self.v[i] if i < n else p.v0
            v_foll = self.v[i - 1] if i > 0 else p.v0
            if x_lead <= x0 or x_foll >= x1:
                continue
            mid = 0.5 * (max(x_foll, x0 - 2 * cfg.accel_length) + min(x_lead, x1 + 2 * cfg.accel_length))
            x_ins = float(np.clip(mid, x0, x1))
            v_merge = max(0.0, min(cfg.merge_speed, v_lead + (0.0 if forced else 2.0), v_foll + 2.0))
            margin = np.inf
            if i < n:
                gap_front = x_lead - x_ins - p.length
                if gap_front < p.s0 + slot * v_merge * p.T:
                    continue
                acc_self = idm_acceleration(np.array([v_merge]), np.array([v_merge - v_lead]), np.array([gap_front]), p)[0]
                if acc_self < limit:
                    continue
                margin = min(margin, acc_self)
            if i > 0:
                gap_back = x_ins - x_foll - p.length
                if gap_back < p.s0 + slot * v_foll * p.T:
                    continue
                acc_f = idm_acceleration(np.array([v_foll]), np.array([v_foll - v_merge]), np.array([gap_back]), p)[0]
                if acc_f < limit:
                    continue
                margin = min(margin, acc_f)
            if margin > best_margin:
                best, best_margin = (x_ins, v_merge), margin

        if best is None:
            return
        if forced:
            self.forced_merges += 1
        self._insert(best[0], best[1], 1, t_arr)
        self.released = None

    def _release_from_signal(self, t: float, rate: float) -> None:
        if self.released is not None or not self.ramp_queue:
            return
        if np.isinf(rate):
            self.released = [self.ramp_queue.pop(0), t]
            return
        headway = 3600.0 / max(rate, 1e-9)
        if t - self._last_release >= headway:
            self.released = [self.ramp_queue.pop(0), t]
            self._last_release = t

    # --------------------------------------------------------------------- run
    def run(self) -> SimulationResult:
        cfg, p = self.cfg, self.p
        dt = cfg.dt
        n_steps = int(round(cfg.duration / dt))
        n_cells = int(cfg.length // cfg.cell)
        x_cells = (np.arange(n_cells) + 0.5) * cfg.cell
        rec_every = max(1, int(round(cfg.record_every / dt)))
        ctrl_every = max(1, int(round(self.controller.interval / dt)))

        times, speed_field, density_field, ramp_q, main_q = [], [], [], [], []
        ctrl_times, rates, det_dens = [], [], []
        self._last_release = -np.inf
        rate = self.controller.update(0.0, 0.0)
        det_len_km = (cfg.detector_end - cfg.detector_start) / 1000.0
        det_accum = 0.0
        det_count = 0

        for k in range(n_steps):
            t = k * dt

            # --- arrivals
            n_main = self.rng.poisson(self.main_demand(t) / 3600.0 * dt)
            self.main_queue.extend([t] * int(n_main))
            n_ramp = self.rng.poisson(self.ramp_demand(t) / 3600.0 * dt)
            self.ramp_queue.extend([t] * int(n_ramp))

            # --- entrance signal and merges
            self._release_from_signal(t, rate)
            self._try_ramp_merge(t)
            self._try_upstream_entry(t)

            # --- car-following dynamics (ballistic update)
            if self.x.size:
                s, dv = self._gap_and_dv()
                acc = idm_acceleration(self.v, dv, s, p)
                v_new = self.v + acc * dt
                stop = v_new < 0.0
                dx = self.v * dt + 0.5 * acc * dt * dt
                if np.any(stop):
                    dx[stop] = -0.5 * self.v[stop] ** 2 / np.minimum(acc[stop], -1e-9)
                    v_new[stop] = 0.0
                self.x = self.x + dx
                self.v = v_new
                # crash guard: IDM is essentially collision-free, but forced merges can violate it
                if self.x.size > 1:
                    max_x = self.x[1:] - p.length - 0.1
                    self.x[:-1] = np.minimum(self.x[:-1], max_x)

                # --- exits
                done = self.x >= cfg.length
                if np.any(done):
                    self._exit_t.extend([t] * int(done.sum()))
                    self._travel.extend((t - self.t_arrive[done]).tolist())
                    self._origins.extend(self.origin[done].tolist())
                    keep = ~done
                    self.x, self.v = self.x[keep], self.v[keep]
                    self.origin, self.t_arrive = self.origin[keep], self.t_arrive[keep]

            # --- detector
            in_det = np.count_nonzero((self.x >= cfg.detector_start) & (self.x < cfg.detector_end))
            det_accum += in_det / det_len_km
            det_count += 1

            # --- control
            if (k + 1) % ctrl_every == 0:
                dens = det_accum / det_count
                rate = self.controller.update(t + dt, dens)
                ctrl_times.append(t + dt)
                rates.append(rate)
                det_dens.append(dens)
                det_accum, det_count = 0.0, 0

            # --- snapshots
            if k % rec_every == 0:
                times.append(t)
                idx = np.clip((self.x // cfg.cell).astype(int), 0, n_cells - 1)
                cnt = np.bincount(idx, minlength=n_cells).astype(float)
                vsum = np.bincount(idx, weights=self.v, minlength=n_cells)
                with np.errstate(invalid="ignore", divide="ignore"):
                    speed_field.append(np.where(cnt > 0, vsum / cnt, np.nan))
                density_field.append(cnt / (cfg.cell / 1000.0))
                ramp_q.append(len(self.ramp_queue) + (1 if self.released is not None else 0))
                main_q.append(len(self.main_queue))

        return SimulationResult(
            controller=self.controller.name,
            config=cfg,
            times=np.array(times),
            x_cells=x_cells,
            speed_field=np.array(speed_field),
            density_field=np.array(density_field),
            control_times=np.array(ctrl_times),
            metering_rate=np.array(rates),
            detector_density=np.array(det_dens),
            ramp_queue=np.array(ramp_q),
            main_queue=np.array(main_q),
            exit_times=np.array(self._exit_t),
            travel_times=np.array(self._travel),
            origins=np.array(self._origins, dtype=int),
            forced_merges=self.forced_merges,
            vehicles_entered=self.entered,
        )
