import numpy as np
import pytest

from traffic_idm import (
    ALINEAMetering,
    DemandProfile,
    FixedRateMetering,
    HighwayConfig,
    HighwaySimulation,
    NoMetering,
    summarize,
)


def _run(controller, duration=600.0, main=900.0, ramp=300.0, seed=3):
    cfg = HighwayConfig(duration=duration, seed=seed)
    sim = HighwaySimulation(cfg, controller, DemandProfile.constant(main), DemandProfile.constant(ramp))
    return sim.run(), sim


def test_demand_profile_interpolates():
    d = DemandProfile.rush_hour(base=100.0, peak=500.0, t_start=100.0, t_end=300.0, ramp=50.0)
    assert d(0.0) == 100.0
    assert d(125.0) == 300.0
    assert d(200.0) == 500.0
    assert d(1000.0) == 100.0


def test_vehicles_never_overlap_and_ordering_is_kept():
    cfg = HighwayConfig(duration=300.0, seed=7)
    sim = HighwaySimulation(cfg, NoMetering(), DemandProfile.constant(1500.0), DemandProfile.constant(600.0))
    sim.run()
    gaps = np.diff(sim.x) - cfg.idm.length
    assert np.all(gaps > -1e-6)
    assert np.all(sim.v >= 0.0)


def test_conservation_of_vehicles():
    res, sim = _run(NoMetering())
    on_road = sim.x.size
    waiting = len(sim.main_queue) + len(sim.ramp_queue) + (1 if sim.released is not None else 0)
    assert res.vehicles_entered == res.exit_times.size + on_road
    assert res.vehicles_entered + waiting > 0


def test_free_flow_travel_time_matches_desired_speed():
    res, _ = _run(NoMetering(), duration=900.0, main=300.0, ramp=0.0)
    expected = res.config.length / res.config.idm.v0
    assert np.all(res.origins == 0)
    assert abs(res.travel_times.mean() - expected) < 0.05 * expected


def test_metering_limits_ramp_inflow():
    res_free, _ = _run(NoMetering(), duration=1200.0, main=600.0, ramp=900.0)
    res_metered, _ = _run(FixedRateMetering(rate=300.0), duration=1200.0, main=600.0, ramp=900.0)
    ramp_free = np.sum(res_free.origins == 1)
    ramp_metered = np.sum(res_metered.origins == 1)
    assert ramp_metered < ramp_free
    assert ramp_metered <= 300.0 * 1200.0 / 3600.0 + 2  # at most the release rate (plus in-flight)
    assert res_metered.ramp_queue.max() > res_free.ramp_queue.max()


def test_alinea_records_control_trace():
    res, _ = _run(ALINEAMetering(interval=30.0), duration=300.0)
    assert res.control_times.size == 10
    assert res.metering_rate.size == res.detector_density.size == 10
    assert np.all(np.isfinite(res.metering_rate))


def test_summary_keys_are_finite():
    res, _ = _run(NoMetering())
    s = summarize(res)
    for key in ("throughput", "mean_speed", "jam_fraction", "mean_travel_time", "total_time_spent"):
        assert np.isfinite(s[key])
    assert 0.0 <= s["jam_fraction"] <= 1.0


@pytest.mark.slow
def test_adaptive_signal_beats_no_metering_in_rush_hour():
    from traffic_idm.experiment import default_scenario

    cfg, main, ramp = default_scenario(duration=3600.0)
    base = summarize(HighwaySimulation(cfg, NoMetering(), main, ramp).run())
    alinea = summarize(HighwaySimulation(cfg, ALINEAMetering(target_density=24.0, gain=60.0), main, ramp).run())
    assert alinea["total_time_spent"] < base["total_time_spent"]
    assert alinea["jam_fraction"] < base["jam_fraction"]
