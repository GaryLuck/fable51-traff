import math

from traffic_idm import ALINEAMetering, FixedRateMetering, NoMetering


def test_no_metering_is_unbounded():
    assert math.isinf(NoMetering().update(0.0, 80.0))


def test_fixed_rate_ignores_measurement():
    c = FixedRateMetering(rate=500.0)
    assert c.update(0.0, 5.0) == 500.0
    assert c.update(60.0, 80.0) == 500.0


def test_alinea_moves_rate_against_density_error_and_saturates():
    c = ALINEAMetering(target_density=25.0, gain=50.0, r_min=200.0, r_max=1800.0, rate=900.0)
    assert c.update(0.0, 35.0) == 400.0  # 10 veh/km too dense -> -500 veh/h
    assert c.update(30.0, 15.0) == 900.0  # 10 veh/km spare -> +500 veh/h
    for _ in range(10):
        c.update(0.0, 0.0)
    assert c.rate == 1800.0
    for _ in range(40):
        c.update(0.0, 100.0)
    assert c.rate == 200.0
