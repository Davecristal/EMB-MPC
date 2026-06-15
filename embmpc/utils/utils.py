import pathlib
import tarfile
import tempfile

import numpy as np
import requests
from numba import njit
import jax.numpy as jnp
import jax


def find_track_dir(track_name: str) -> pathlib.Path:
    map_dir = pathlib.Path(__file__).parent.parent.parent.parent / "maps"

    if not (map_dir / track_name).exists():
        print("Downloading Files for: " + track_name)
        tracks_url = "http://api.f1tenth.org/" + track_name + ".tar.xz"
        tracks_r = requests.get(url=tracks_url, allow_redirects=True)
        if tracks_r.status_code == 404:
            raise FileNotFoundError(f"No maps exists for {track_name}.")

        tempdir = tempfile.gettempdir() + "/"

        with open(tempdir + track_name + ".tar.xz", "wb") as f:
            f.write(tracks_r.content)

        print("Extracting Files for: " + track_name)
        tracks_file = tarfile.open(tempdir + track_name + ".tar.xz")
        tracks_file.extractall(map_dir)
        tracks_file.close()


    for subdir in map_dir.iterdir():
        if track_name == str(subdir.stem).replace(" ", ""):
            return subdir

    raise FileNotFoundError(f"no mapdir matching {track_name} in {[map_dir]}")


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> tuple:
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@jax.jit
def nearest_point_on_trajectory_jax(point, trajectory) -> tuple:
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = jnp.sum((point - trajectory[:-1, :]) * diffs[:, :], axis=1)
    t = jnp.clip(dots / (l2s + 1e-8), 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = jnp.linalg.norm(point - projections, axis=1)
    min_dist_segment = jnp.argmin(dists)
    return (
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment
    )