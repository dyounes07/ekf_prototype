"""tests for the global EKF (ekf_global.py)"""
import numpy as np
from ekf_global import GlobalEKF, se2_mul, se2_inv, map_to_odom


def test_se2_inverse():
    # a pose composed with its own inverse should be "no change" (0, 0, 0)
    a = np.array([1.0, 2.0, 0.7])
    assert np.allclose(se2_mul(a, se2_inv(a)), [0, 0, 0], atol=1e-12)


def test_map_odom_composition():
    # lesson 2 example: odom says we're at (10, 0) but in the map we're at (10.5, 0),
    # so the map->odom correction should be +0.5 in x. then check it goes both ways:
    # map_T_odom * odom_T_base should give map_T_base back
    odom_T_base = np.array([10.0, 0.0, 0.0]); map_T_base = np.array([10.5, 0.0, 0.0])
    mo = map_to_odom(map_T_base, odom_T_base)
    assert np.allclose(mo, [0.5, 0, 0])
    assert np.allclose(se2_mul(mo, odom_T_base), map_T_base)


def run(jump_at=None, jump=np.array([0.8, 0.0]), persistent=False, seed=0):
    # fake rover driving a curve with noisy SLAM poses. jump_at = which step the
    # SLAM pose jumps by `jump` meters; persistent=True means it stays shifted
    # for every later step (a real loop closure), False means just that one reading
    # (an outlier). returns the filter's error at each step and the filter itself
    rng = np.random.default_rng(seed)
    g = GlobalEKF(np.array([0, 0, 0, .5, .1]), np.diag([.1,.1,.1,.5,.5]),
                  np.diag([1e-3,1e-3,1e-3,1e-2,1e-2]))
    R3 = np.diag([.05**2, .05**2, .05**2]); dt = .1
    truth = np.array([0.0, 0.0, 0.0]); err = []
    for i in range(300):
        v, w = .5, .1
        truth = truth + [v*np.cos(truth[2])*dt, v*np.sin(truth[2])*dt, w*dt]
        g.predict(dt)
        # speed and turn rate from "/odometry/local" (noisy)
        g.update_velocity(v + rng.normal(0,.02), w + rng.normal(0,.01), np.diag([.02**2,.01**2]))
        zs = truth.copy()
        if jump_at is not None and (i == jump_at or (persistent and i > jump_at)):
            zs[:2] += jump   # shift the SLAM pose
        z = zs + rng.normal(0, .05, 3)
        g.update_slam(z[0], z[1], z[2], R3)
        err.append(np.hypot(*(g.f.x[:2] - truth[:2])))
    return np.array(err), g


def test_tracks_truth():
    # with normal SLAM noise and no jumps, the filter should stay close to truth
    err, _ = run()
    assert err[50:].mean() < 0.05


def test_single_outlier_is_rejected():
    # one bad SLAM reading should get gated out, so the estimate barely moves
    err, g = run(jump_at=150)
    assert any(not ok for _, ok, _ in g.log)
    assert err[150:170].max() < 0.15      # outlier did not drag the estimate


def test_persistent_jump_eventually_accepted():
    # a real loop closure shifts every later SLAM pose; we must not reject forever
    err, g = run(jump_at=150, persistent=True)
    # log has one entry per SLAM update; look at the run of rejects starting at the jump
    run_len = 0
    for _, ok, _ in g.log[150:]:
        if ok: break
        run_len += 1
    assert run_len <= 5            # capped, then the pose is re-initialized
    assert err[-1] > 0.5           # and the estimate now sits at the shifted SLAM pose