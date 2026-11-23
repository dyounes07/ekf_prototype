"""my planar EKF for the rover (no ROS needed for this file)

what this is: a filter that keeps a running best guess of where the rover is
AND how unsure we are about that guess. two steps on repeat:
    predict - move the guess forward in time using a motion model
    update  - fix the guess using a sensor reading

state x = [x, y, theta, v, omega]
    x, y = position in the odom frame (m)
    theta = heading (rad), always wrapped to [-pi, pi]
    v = forward speed (m/s)
    omega = how fast we're turning (rad/s)

equations are from Ribeiro ch. 5 (predict w/ f and F, update w/ h and H)
and the Joseph form covariance update from Lacey 11.21
"""

import numpy as np

STATE_DIM = 5
# names for the 5 slots so i can write x[ITH] instead of x[2] (range(5) = 0..4)
IX, IY, ITH, IV, IW = range(5)


def wrap_angle(a):
    """put any angle into [-pi, pi]

    why: 3.1 and -3.1 rad are basically the same direction (0.08 apart) but
    subtracting them gives 6.2. wrapping the difference fixes that.
    trick: shift by +pi, mod 2*pi, shift back
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def motion_model(x, dt):
    """f: where will the rover be after dt seconds?

    assumes it keeps its current speed and turn rate (constant velocity
    unicycle). the sensors fix this assumption later in update()
    """
    px, py, th, v, w = x
    return np.array([
        px + v * np.cos(th) * dt,      # drive forward along the heading
        py + v * np.sin(th) * dt,
        wrap_angle(th + w * dt),       # turn, and keep heading wrapped
        v,                             # speed stays the same
        w,                             # turn rate stays the same
    ])


def motion_jacobian(x, dt):
    """F = df/dx at the current state

    F[i, j] = "if state j changes a tiny bit, how much does predicted state i
    change?". we need it because f is nonlinear, so we swap it for its
    tangent line around where we are right now (that's the "extended" part)
    """
    _, _, th, v, _ = x
    F = np.eye(STATE_DIM)   # default: every entry stays as it was
    # the few spots where one state affects another:
    F[IX, ITH] = -v * np.sin(th) * dt   # x depends on heading (derivative of v*cos(th)*dt)
    F[IX, IV] = np.cos(th) * dt         # x depends on speed
    F[IY, ITH] = v * np.cos(th) * dt    # y depends on heading
    F[IY, IV] = np.sin(th) * dt         # y depends on speed
    F[ITH, IW] = dt                     # heading depends on turn rate
    return F


class PlanarEKF:
    def __init__(self, x0, P0, Q):
        """
        x0: starting state (5,)
        P0: starting covariance (5, 5) = how unsure we are at the start
        Q:  process noise per second (5, 5) = how much extra uncertainty the
            real world adds because our motion model isn't perfect.
            predict() multiplies it by dt so it's "variance per second"
        """
        # .copy() so we never mess with arrays the caller still uses
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float).copy()

    def predict(self, dt):
        """move the guess forward dt seconds. uncertainty grows."""
        # F has to be computed at the OLD state, so do it before moving x
        F = motion_jacobian(self.x, dt)
        self.x = motion_model(self.x, dt)
        # old uncertainty gets carried along by F, plus new uncertainty Q*dt
        self.P = F @ self.P @ F.T + self.Q * dt
        # force P to be exactly symmetric (rounding can break it a tiny bit)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z, H, R, angle_rows=()):
        """fix the guess using a measurement z

        z: the measurement (m,)
        H: (m, 5) matrix that turns the state into "what the sensor reads".
           ours are just selection matrices (the sensors read state entries
           directly) so no Jacobian needed here
        R: (m, m) how noisy the sensor is
        angle_rows: which rows of z are angles, so their innovation gets wrapped

        returns (e, S) so the caller can compute NIS = e^T S^-1 e
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        H = np.atleast_2d(H)
        R = np.atleast_2d(R)

        # innovation e = what we got minus what we expected to get
        e = z - H @ self.x
        for r in angle_rows:
            e[r] = wrap_angle(e[r])

        # S = how big we expect that gap to be (our uncertainty + sensor noise)
        S = H @ self.P @ H.T + R
        # Kalman gain K = how much to trust the sensor vs our own guess.
        # sensor noise small compared to our uncertainty -> K is big
        K = self.P @ H.T @ np.linalg.inv(S)

        # nudge the state by gain * innovation, then re-wrap the heading
        self.x = self.x + K @ e
        self.x[ITH] = wrap_angle(self.x[ITH])

        # Joseph form: works for any K and keeps P symmetric + positive definite
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return e, S

    # shortcuts for the two sensors ekf_local fuses right now
    def update_wheel_odom(self, z_xy_yaw, R):
        """wheel odometry pose: x, y, yaw"""
        H = np.zeros((3, STATE_DIM))
        H[0, IX] = 1.0   # row 0 reads the x entry
        H[1, IY] = 1.0   # row 1 reads y
        H[2, ITH] = 1.0  # row 2 reads yaw
        return self.update(z_xy_yaw, H, R, angle_rows=(2,))  # row 2 is an angle

    def update_gyro(self, omega_z, r):
        """IMU yaw rate (how fast we're turning)"""
        H = np.zeros((1, STATE_DIM))
        H[0, IW] = 1.0   # the gyro reads the omega entry directly
        return self.update([omega_z], H, np.array([[r]]))