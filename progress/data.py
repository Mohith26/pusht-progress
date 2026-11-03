"""Load LeRobot Push-T episodes and build progress targets."""

import io
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PUSHT_URL = (
    "https://huggingface.co/datasets/lerobot/pusht_image/resolve/main/"
    "data/chunk-000/file-000.parquet"
)


@dataclass
class Episodes:
    episode: np.ndarray  # (N,) episode id per frame
    frame: np.ndarray  # (N,) frame index inside the episode
    length: np.ndarray  # (N,) length of the frame's episode
    images: list  # N decoded RGB uint8 arrays (only kept when needed)

    @property
    def progress(self) -> np.ndarray:
        """Time-normalized progress in [0, 1]: frame / (length - 1)."""
        return self.frame / np.maximum(self.length - 1, 1)


def load_pusht(path: str, decode_images: bool = True) -> Episodes:
    table = pq.read_table(path)
    episode = np.asarray(table.column("episode_index").to_pylist())
    frame = np.asarray(table.column("frame_index").to_pylist())
    order = np.lexsort((frame, episode))
    episode, frame = episode[order], frame[order]
    _, inverse, counts = np.unique(episode, return_inverse=True, return_counts=True)
    length = counts[inverse]
    images = []
    if decode_images:
        raw = table.column("observation.image").to_pylist()
        images = [np.asarray(Image.open(io.BytesIO(raw[i]["bytes"])).convert("RGB")) for i in order]
    return Episodes(episode=episode, frame=frame, length=length, images=images)


def first_frame_index(episode: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """For every row, the row index of frame 0 of the same episode."""
    first = {}
    for i, (e, f) in enumerate(zip(episode, frame)):
        if f == 0:
            first[e] = i
    return np.asarray([first[e] for e in episode])
