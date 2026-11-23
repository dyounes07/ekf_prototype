"""tests for the local EKF (ekf.py)
hfhgjfjhfghvbcjhgjhjhjgjhjhghfhjhjhhjjhhjhfgjjhgjhgjhjhhjjh
run with: python3 -m pytest -q
if my filter is right these all pass. if i change something and one fails,
that tells me what broke
"""
import numpy as np

from ekf import (
    PlanarEKF, motion_model, motion_jacobian, wrap_angle,
    IX, IY, ITH,
)


def numerical_jacobian(f, x, eps=1e-6):
    # checks my hand-derived F: nudge each state entry a tiny bit up and down,
    # see how f changes, divide by the nudge. this is the Jacobian without any calculus
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return J


def test_wrap_angle():
    # just past +pi should come out just past -pi (and the other way around),
    # and a normal angle shouldn't change
    assert np.isclose(wrap_angle(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1)
    assert np.isclose(wrap_angle(0.3), 0.3)


def test_jacobian_matches_numerical():
    # my analytic F vs the numerical one, at 20 random states, so a sign mistake
    # (usually on the sin entries) can't hide at a lucky angle
    rng = np.random.default_rng(0)
    dt = 0.05
    for _ in range(20):
        x = np.array([rng.normal(), rng.normal(),
                      rng.uniform(-3, 3), rng.normal(), rng.normal()])
        F_analytic = motion_jacobian(x, dt)
        F_numeric = numerical_jacobian(lambda s: motion_model(s, dt), x)
        assert np.allclose(F_analytic, F_numeric, atol=1e-6)


def make_filter():
    # a filter with made-up starting values, used by a few tests below
    x0 = np.zeros(5)
    P0 = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
    Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])
    return PlanarEKF(x0, P0, Q)


def test_covariance_stays_symmetric_positive_definite():
    # P is a covariance so it has to stay symmetric and positive definite
    # (all eigenvalues > 0). run 500 predict+update rounds and check every time.
    # this is what the Joseph form is for
    ekf = make_filter()
    R_odom = np.diag([0.1**2, 0.1**2, 0.05**2])
    rng = np.random.default_rng(1)
    for _ in range(500):
        ekf.predict(0.05)
        ekf.update_wheel_odom(rng.normal(scale=0.1, size=3), R_odom)
        ekf.update_gyro(rng.normal(scale=0.02), 0.02**2)
        assert np.allclose(ekf.P, ekf.P.T)
        assert np.all(np.linalg.eigvalsh(ekf.P) > 0)


def test_covariance_grows_without_measurements():
    # no sensors for 100 steps -> we should get less and less sure
    # (trace = sum of the uncertainties on the diagonal)
    ekf = make_filter()
    trace0 = np.trace(ekf.P)
    for _ in range(100):
        ekf.predict(0.05)
    assert np.trace(ekf.P) > trace0


def test_yaw_innovation_wraps_across_pi():
    # heading just below +pi, measurement just above -pi: they're only 0.2 rad
    # apart, so the filter should move a little, not by ~2*pi.
    # (without wrap_angle on the innovation it would spin the whole way around)
    ekf = PlanarEKF(np.array([0, 0, np.pi - 0.1, 0, 0]),
                    np.eye(5), np.zeros((5, 5)))
    e, _ = ekf.update_wheel_odom([0.0, 0.0, -np.pi + 0.1],
                                 np.diag([0.01, 0.01, 0.01]))
    assert abs(e[2]) < 0.3
    assert -np.pi <= ekf.x[ITH] <= np.pi
    assert abs(wrap_angle(ekf.x[ITH] - (np.pi - 0.1))) < 0.3


def simulate(seed=3, steps=600, dt=0.05):
    """fake rover with known truth, to check the filter against

    truth drives at constant speed + turn rate, and the sensors are the truth
    plus noise. returns the filter's error, the raw sensor's error and the NIS
    at every step
    """
    rng = np.random.default_rng(seed)
    v_true, w_true = 1.0, 0.2
    sx, sy, sth = 0.1, 0.1, 0.05   # wheel odom noise std
    sg = 0.02                       # gyro noise std

    truth = np.array([0.0, 0.0, 0.0, v_true, w_true])
    ekf = PlanarEKF(truth.copy(), np.diag([0.1, 0.1, 0.1, 0.5, 0.5]),
                    np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2]))
    R_odom = np.diag([sx**2, sy**2, sth**2])

    est_err, meas_err, nis = [], [], []
    for _ in range(steps):
        truth = motion_model(truth, dt)   # the real rover moves
        ekf.predict(dt)                   # the filter moves its guess too

        # noisy wheel odom reading, update the filter with it
        z = truth[[IX, IY, ITH]] + rng.normal(scale=[sx, sy, sth])
        z[2] = wrap_angle(z[2])
        e, S = ekf.update_wheel_odom(z, R_odom)
        nis.append(float(e @ np.linalg.inv(S) @ e))
        # noisy gyro reading, update with that too
        ekf.update_gyro(truth[4] + rng.normal(scale=sg), sg**2)

        # how far off is the filter from truth, and how far off is the raw sensor?
        est_err.append(np.hypot(*(ekf.x[[IX, IY]] - truth[[IX, IY]])))
        meas_err.append(np.hypot(*(z[:2] - truth[:2])))
    return np.array(est_err), np.array(meas_err), np.array(nis)


def test_filter_beats_raw_measurements():
    # the whole point of the filter: its error should be smaller than just using
    # the noisy sensor directly. RMSE = root mean square error
    est_err, meas_err, _ = simulate()
    # skip the first 50 steps while the filter converges.
    assert np.sqrt(np.mean(est_err[50:] ** 2)) < np.sqrt(np.mean(meas_err[50:] ** 2))


def test_nis_is_consistent():
    # a consistent filter has mean NIS close to the measurement size (3 numbers
    # here). way above = overconfident, way below = too cautious
    _, _, nis = simulate()
    assert 2.0 < nis[50:].mean() < 4.5