"""tests for the replay logic (replay_bag.py) on fake data

these run on my Mac too, because they only test run_filter / align_truth,
not the rosbag reading (that needs ROS and only works in the container)
"""
import numpy as np
from replay_bag import run_filter, report, align_truth


def synth(seed=0, steps=400, dt=0.033):
    # makes a fake "bag": a rover driving a curve at 30 Hz, with noisy wheel odom
    # readings and noisy gyro readings, in the same (time, kind, data) format
    # that run_filter expects. also returns the true path to compare against
    rng = np.random.default_rng(seed)
    x = np.array([0.0, 0.0, 0.0]); v, w = 0.5, 0.15
    ev, tr = [], []
    for i in range(steps):
        t = 100 + i*dt
        x = x + [v*np.cos(x[2])*dt, v*np.sin(x[2])*dt, w*dt]   # true motion
        tr.append((t, *x))
        # wheel odom event: noisy x, y, yaw, speed
        ev.append((t, 'wheel', (x[0]+rng.normal(0,.05), x[1]+rng.normal(0,.05),
                                x[2]+rng.normal(0,.05), v+rng.normal(0,.05))))
        # gyro event 5 ms later: noisy turn rate
        ev.append((t+0.005, 'imu', (w+rng.normal(0,.02),)))
    return ev, np.array(tr)


def test_replay_runs_and_beats_raw():
    # the filter's path should be closer to the true path than the raw wheel
    # odom is (same idea as the test in test_ekf, but through run_filter)
    ev, tr = synth()
    times, states, nis = run_filter(ev)
    raw = np.array([d[:2] for t,k,d in ev if k=='wheel'])
    truth_xy = np.interp(times, tr[:,0], tr[:,1]), np.interp(times, tr[:,0], tr[:,2])
    est = states[:, :2]
    e_est = np.sqrt(np.mean((est[:,0]-truth_xy[0])**2+(est[:,1]-truth_xy[1])**2))
    e_raw = np.sqrt(np.mean((raw[:,0]-tr[:,1])**2+(raw[:,1]-tr[:,2])**2))
    assert e_est < e_raw


def test_late_message_is_dropped_not_crash():
    # lesson 5: a message with an older stamp than the filter's time can't be
    # predicted backwards. run_filter should drop it, not crash
    ev, _ = synth(steps=20)
    ev.insert(10, (ev[5][0], 'imu', (0.1,)))   # stamp earlier than filter time
    times, states, nis = run_filter(ev)
    assert len(times) > 0                       # it still ran and gave results


def test_align_truth():
    # truth starts at (5, 5) facing +y (pi/2). after aligning, it should start at
    # the odom origin and a 1 m move "forward" should show up as +1 in x
    tr = np.array([[0,5.,5.,np.pi/2],[1,5.,6.,np.pi/2]])
    t, xy = align_truth(tr, 0, 0, 0)
    assert np.allclose(xy[1], [1, 0])   # moved 1 m "forward" in own start frame