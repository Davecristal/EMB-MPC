from __future__ import annotations
import pathlib
from typing import Optional

import numpy as np
from embmpc.utils.cubic_spline import CubicSpline2D


class Raceline:

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        ss: Optional[np.ndarray] = None,
        psis: Optional[np.ndarray] = None,
        kappas: Optional[np.ndarray] = None,
        accxs: Optional[np.ndarray] = None,
        spline: Optional[CubicSpline2D] = None,
    ):
        assert xs.shape == ys.shape == velxs.shape, "inconsistent shapes for x, y, vel"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.ks = kappas
        self.vxs = velxs
        self.axs = accxs
        self.waypoints = np.stack([ss, xs, ys, psis, kappas, velxs, accxs], axis=1)




        self.length = len(self.ss)


        self.spline = spline or CubicSpline2D(x=xs, y=ys)
        self.s_frame_max = self.spline.s[-1]

    @staticmethod
    def from_centerline_file(
        filepath: pathlib.Path,
        delimiter: Optional[str] = ",",
        fixed_speed: Optional[float] = 1.0,
        track_scale: Optional[float] = 1.0,
    ):
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter)
        assert waypoints.shape[1] == 4, "expected waypoints as [x, y, w_left, w_right]"


        xx, yy = waypoints[:, 0], waypoints[:, 1]

        xx, yy = xx * track_scale, yy * track_scale
        

        xx = np.append(xx, xx[0])
        yy = np.append(yy, yy[0])
        spline = CubicSpline2D(x=xx, y=yy)
        ds = 0.1

        ss, xs, ys, yaws, ks = [], [], [], [], []

        for i_s in np.arange(0, spline.s[-1], ds):
            x, y = spline.calc_position(i_s)
            yaw = spline.calc_yaw(i_s)
            k = spline.calc_curvature(i_s)

            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            ks.append(k)
            ss.append(i_s)

        return Raceline(
            ss=np.array(ss).astype(np.float32),
            xs=np.array(xs).astype(np.float32),
            ys=np.array(ys).astype(np.float32),
            psis=np.array(yaws).astype(np.float32),
            kappas=np.array(ks).astype(np.float32),
            velxs=np.ones_like(ss).astype(np.float32) * fixed_speed,
            accxs=np.zeros_like(ss).astype(np.float32),
            spline=spline,
        )

    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", skip_rows: int = 3, track_scale: Optional[float] = 1.0) -> Raceline:
        if type(filepath) is str:
            filepath = pathlib.Path(filepath)

        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter, skiprows=
        skip_rows).astype(np.float32)

        if track_scale != 1.0:

            waypoints[:, 1] *= track_scale
            waypoints[:, 2] *= track_scale
            spline = CubicSpline2D(x=waypoints[:, 1], y=waypoints[:, 2])    
            ss, yaws, ks = spline.ss, spline.psis, spline.ks
            waypoints[:, 0] = ss
            waypoints[:, 3] = yaws
            waypoints[:, 4] = ks
        
        assert (
            waypoints.shape[1] == 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        return Raceline(
            ss=waypoints[:, 0],
            xs=waypoints[:, 1],
            ys=waypoints[:, 2],
            psis=waypoints[:, 3],
            kappas=waypoints[:, 4],
            velxs=waypoints[:, 5],
            accxs=waypoints[:, 6],
        )