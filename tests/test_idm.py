import numpy as np

from traffic_idm import IDMParams, equilibrium_spacing, fundamental_diagram, idm_acceleration


def test_free_road_accelerates_towards_desired_speed():
    p = IDMParams()
    a = idm_acceleration(np.array([0.0, p.v0 / 2, p.v0]), np.zeros(3), np.full(3, np.inf), p)
    assert a[0] == p.a
    assert 0 < a[1] < p.a
    assert abs(a[2]) < 1e-9


def test_closing_in_on_a_stopped_leader_brakes():
    p = IDMParams()
    a = idm_acceleration(np.array([30.0]), np.array([30.0]), np.array([40.0]), p)
    assert a[0] <= -p.b


def test_equilibrium_spacing_is_a_fixed_point():
    p = IDMParams()
    v = np.array([5.0, 15.0, 25.0])
    s = equilibrium_spacing(v, p)
    a = idm_acceleration(v, np.zeros_like(v), s, p)
    assert np.allclose(a, 0.0, atol=1e-9)


def test_fundamental_diagram_has_a_capacity_and_critical_density():
    fd = fundamental_diagram(IDMParams())
    assert 1500 < fd["q_max"] < 2500
    assert 15 < fd["rho_crit"] < 45
    assert fd["flow"].max() == fd["q_max"]
    # flow vanishes at both ends of the density range
    assert fd["flow"][0] < 1e-6
    assert fd["flow"][-1] < 0.05 * fd["q_max"]
