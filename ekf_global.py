"""my global EKF prototype: fuses ABSOLUTE poses (SLAM, maybe GPS) in the map frame

same filter core as the local one (PlanarEKF), plus two new things:
  * gating: throw out a measurement if its NIS is too big (SLAM jump / bad GPS fix)
  * map->odom: the transform ekf_global should publish, built from
    map->base_link and odom->base_link
frames follow lesson 2: map_T_base = map_T_odom * odom_T_base
"""
import numpy as np
from ekf import PlanarEKF, wrap_angle, IX, IY, ITH, IV, IW, STATE_DIM

# chi-square 99% limits for gating, by measurement size (1, 2, 3 numbers)
# if NIS is above this, the measurement is probably an outlier
GATE = {1: 6.63, 2: 9.21, 3: 11.34}
MAX_REJECTS = 5   # if we reject this many in a row it's not an outlier, it's a real shift (loop closure)


# ---------------- SE(2) helpers: a pose is (x, y, yaw) ----------------
def se2_mul(a, b):
    # compose two 2D poses: pose b is expressed in a's frame, result is b in a's parent frame
    # (rotate b's position by a's yaw, add a's position, add the yaws)
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1],
                     a[1] + s * b[0] + c * b[1],
                     wrap_angle(a[2] + b[2])])


def se2_inv(a):
    # inverse of a pose: undoes it (a * inverse(a) = no change)
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), s * a[0] - c * a[1], wrap_angle(-a[2])])


def map_to_odom(map_T_base, odom_T_base):
    """the map->odom transform. map_T_base = map_T_odom * odom_T_base, so
    map_T_odom = map_T_base * inverse(odom_T_base). corrections go here, so
    odom->base_link stays smooth (lesson 2)"""
    return se2_mul(map_T_base, se2_inv(odom_T_base))


# ---------------- gated update ----------------
def preview_nis(ekf, z, H, R, angle_rows=()):
    # compute NIS for a measurement WITHOUT applying it, so we can decide to gate it
    z, H, R = np.atleast_1d(z), np.atleast_2d(H), np.atleast_2d(R)
    e = z - H @ ekf.x
    for r in angle_rows:
        e[r] = wrap_angle(e[r])
    S = H @ ekf.P @ H.T + R
    return float(e @ np.linalg.inv(S) @ e)


class GlobalEKF:
    def __init__(self, x0, P0, Q):
        self.f = PlanarEKF(x0, P0, Q)   # the same filter as the local one
        self.rejects = 0                # consecutive rejected measurements
        self.log = []                   # (kind, accepted?, nis) for every gated update

    def predict(self, dt):
        self.f.predict(dt)

    def _gated(self, kind, z, H, R, angle_rows=()):
        # apply the update only if it passes the NIS gate; returns True if used
        m = len(np.atleast_1d(z))
        nis = preview_nis(self.f, z, H, R, angle_rows)
        if nis > GATE[m]:
            self.rejects += 1
            if self.rejects <= MAX_REJECTS:
                self.log.append((kind, False, nis))
                return False
            # we keep disagreeing -> this is a real shift (loop closure), not an outlier.
            # one update can't close a big gap (i tried, the next ones got rejected
            # again), so just re-initialize the pose from the measurement
            self._reset_pose(z, H, R)
            self.rejects = 0
            self.log.append((kind, True, nis))
            return True
        self.rejects = 0
        self.f.update(z, H, R, angle_rows)
        self.log.append((kind, True, nis))
        return True

    def _reset_pose(self, z, H, R):
        # overwrite the measured state entries with the measurement and set their
        # covariance to R (we trust the new pose as much as the sensor says)
        z = np.atleast_1d(z)
        idx = [int(np.argmax(row)) for row in np.atleast_2d(H)]
        for zi, i in zip(z, idx):
            self.f.x[i] = zi
        self.f.x[ITH] = wrap_angle(self.f.x[ITH])
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                self.f.P[i, j] = R[a, b]
            for k in range(STATE_DIM):
                if k not in idx:
                    self.f.P[i, k] = self.f.P[k, i] = 0.0

    def update_slam(self, x, y, yaw, R3):
        # SLAM pose in the map frame (R3 = covariance from pose_relay)
        H = np.zeros((3, STATE_DIM)); H[0, IX] = H[1, IY] = H[2, ITH] = 1
        return self._gated('slam', [x, y, yaw], H, R3, angle_rows=(2,))

    def update_gps(self, x, y, std):
        # GPS position in the map frame (std in meters)
        H = np.zeros((2, STATE_DIM)); H[0, IX] = H[1, IY] = 1
        return self._gated('gps', [x, y], H, np.eye(2) * std**2)

    def update_velocity(self, v, w, R2):
        """speed + turn rate from /odometry/local, so the filter knows how we are moving
        between SLAM updates"""
        H = np.zeros((2, STATE_DIM)); H[0, IV] = H[1, IW] = 1
        self.f.update([v, w], H, R2)