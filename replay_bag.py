"""replay a recorded bag through the EKF and see how it does

compares the filter's path against /odometry/local (what ekf_local made) and
against ground truth from gazebo, if the bag has it

run this INSIDE the sim container (needs rclpy / rosbag2_py):
    source /opt/ros/humble/setup.bash
    python3 replay_bag.py /ws/src/timbot/bags/loop_test

needs ekf.py (the filter) in the same folder
frames: everything is in the odom frame, which starts at the first wheel_odom
pose. ground truth gets lined up with it using its first sample
"""
import sys
import numpy as np
from ekf import PlanarEKF, wrap_angle, IV, IW, STATE_DIM

# ---- noise settings: these are placeholders i picked, TUNE THEM (lesson 6) ----
# R = how noisy each sensor is, Q = how much our motion model can be off per second
# this is how we make the custom ekf useful!!!
R_WHEEL_POSE = np.diag([0.05**2, 0.05**2, 0.05**2])  # x, y, yaw (m, m, rad)
R_WHEEL_V = 0.05**2                                   # forward speed (m/s)
R_GYRO = 0.02**2                                      # yaw rate (rad/s)
Q = np.diag([1e-3, 1e-3, 1e-3, 1e-1, 1e-1])           # per second
P0 = np.diag([0.1, 0.1, 0.1, 0.5, 0.5])             # how unsure we are at the start


def yaw_from_quat(q):
    # ros gives orientation as a quaternion (x, y, z, w); the planar filter
    # only needs the yaw angle, so this pulls it out
    return np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                      1 - 2 * (q[1] ** 2 + q[2] ** 2))


def run_filter(events):
    """run the filter over a list of events sorted by time

    events: (t, kind, data)
        kind 'wheel' - data = (x, y, yaw, v)
        kind 'imu' - data = (wz,)
    returns (times, states, nis_list)
    """
    ekf, t_f = None, None
    times, states, nis = [], [], []
    for t, kind, d in events:
        if ekf is None:
            # start the filter at the first wheel_odom pose (omega starts at 0)
            if kind != 'wheel':
                continue
            x0 = np.array([d[0], d[1], d[2], d[3], 0.0])
            ekf, t_f = PlanarEKF(x0, P0, Q), t
            continue
        dt = t - t_f # time since the filter was last brought up to date
        if dt < 0: # message is older than the filter: just drop it (lesson 5, option 1)
            continue
        if dt > 1.0: # big gap in the data: don't take one giant step, linearizing breaks
            dt = 1.0
        ekf.predict(dt) # move the guess up to this message's time
        t_f = t
        if kind == 'wheel':
            e, S = ekf.update_wheel_odom(d[:3], R_WHEEL_POSE)
            nis.append(float(e @ np.linalg.inv(S) @ e))   # save NIS to check consistency later
            # wheel odom also reports speed, so use that too
            H = np.zeros((1, STATE_DIM)); H[0, IV] = 1.0
            ekf.update([d[3]], H, np.array([[R_WHEEL_V]]))
        else:
            ekf.update_gyro(d[0], R_GYRO)   # imu: turn rate
        times.append(t); states.append(ekf.x.copy())
    return np.array(times), np.array(states), np.array(nis)


def interp_xy(t_ref, t, xy):
    # resample a path at the times in t_ref so two paths can be compared point by point
    return np.c_[np.interp(t_ref, t, xy[:, 0]), np.interp(t_ref, t, xy[:, 1])]


def rmse(a, b):
    # root mean square distance between two paths (m)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def report(times, states, nis, local, truth):
    # print the numbers that tell me if the filter is any good (lesson 6)
    est = states[:, :2]
    print(f"filter updates: {len(times)}, duration {times[-1]-times[0]:.1f} s")
    path = np.sum(np.linalg.norm(np.diff(est, axis=0), axis=1))
    print(f"path length (est): {path:.2f} m")
    print(f"loop gap (end vs start, est): {np.linalg.norm(est[-1]-est[0]):.3f} m"
          " (meaningful only if you drove back to the start)")
    if len(nis):
        print(f"mean NIS (wheel pose, m=3): {nis.mean():.2f}   target ~3, "
              "above = overconfident")
    if local is not None:
        lt, lxy = local
        print(f"RMSE vs /odometry/local: {rmse(est, interp_xy(times, lt, lxy)):.3f} m"
              "  (agreement, NOT accuracy)")
    if truth is not None:
        tt, txy = truth
        print(f"RMSE vs ground truth:    {rmse(est, interp_xy(times, tt, txy)):.3f} m")
        if local is not None:
            print("RMSE of ekf_local vs ground truth: "
                  f"{rmse(interp_xy(tt, lt, lxy), txy):.3f} m")

# ------------------------- rosbag2 reading (container only) ----------------
def pose_of(msg):
    # works for Odometry (msg.pose.pose) and PoseStamped (msg.pose); i wasn't sure which
    # type the ground truth topic uses
    p = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
    q = p.orientation
    return p.position.x, p.position.y, yaw_from_quat((q.x, q.y, q.z, q.w))


def read_bag(path):
    # open the bag, keep only the 4 topics i care about, turn them into events
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    events, local, truth = [], [], []
    while reader.has_next():
        topic, raw, tns = reader.read_next()
        if topic not in ('/wheel_odom', '/imu/data', '/odometry/local',
                         '/gazebo_world/ground_truth'):
            continue
        msg = deserialize_message(raw, get_message(types[topic]))
        s = msg.header.stamp
        # use header.stamp (when the measurement was TAKEN); fall back to bag time if it's empty
        t = s.sec + s.nanosec * 1e-9 if (s.sec or s.nanosec) else tns * 1e-9
        if topic == '/wheel_odom':
            x, y, yaw = pose_of(msg)
            events.append((t, 'wheel', (x, y, yaw, msg.twist.twist.linear.x)))
        elif topic == '/imu/data':
            events.append((t, 'imu', (msg.angular_velocity.z,)))
        elif topic == '/odometry/local':
            local.append((t, *pose_of(msg)[:2]))
        else:
            truth.append((t, *pose_of(msg)))
    events.sort(key=lambda e: e[0])
    return events, np.array(local), np.array(truth), types


def align_truth(truth, x0, y0, yaw0):
    """gazebo's truth is in the world frame but the filter works in the odom frame,
    which starts at the rover's first pose. this rotates/shifts the truth so both
    start at the same place"""
    tx0, ty0, th0 = truth[0, 1:4]
    c, s = np.cos(-th0), np.sin(-th0)
    d = truth[:, 1:3] - [tx0, ty0]
    xy = np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]
    return truth[:, 0], xy + [x0, y0]


if __name__ == '__main__':
    # read bag -> run filter -> line up truth -> print report -> save csv
    events, local, truth, types = read_bag(sys.argv[1])
    print('topic types in bag:', types.get('/gazebo_world/ground_truth'))
    times, states, nis = run_filter(events)
    loc = (local[:, 0], local[:, 1:3]) if len(local) else None
    tru = None
    if len(truth):
        w0 = next(d for t, k, d in events if k == 'wheel')
        tru = align_truth(truth, w0[0], w0[1], w0[2])
    report(times, states, nis, loc, tru)
    np.savetxt('replay_output.csv', np.c_[times, states],
               delimiter=',', header='t,x,y,yaw,v,w', comments='')
    print('wrote replay_output.csv')