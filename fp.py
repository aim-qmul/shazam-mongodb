from collections import defaultdict
from os import PathLike
import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from typing import Any, Callable, Iterable, Tuple, List
import librosa
from functools import partial, reduce
from itertools import chain, starmap
import json
from tqdm import tqdm


def compute_constellation_map(
    Y: np.ndarray, dist_freq: int = 7, dist_time: int = 7, thresh_ratio: float = 0.01
) -> Tuple[np.ndarray, ...]:
    thresh = np.max(Y) * thresh_ratio
    result = maximum_filter(
        Y, size=[2 * dist_freq + 1, 2 * dist_time + 1], mode="constant"
    )
    return np.nonzero((Y == result) & (Y > thresh))


def wav2spec(path: PathLike, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr)
    return np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))


def wang_peak_pairs(
    peak_freq_indexes: np.ndarray,
    peak_time_indexes: np.ndarray,
    zone_dist_time: int = 25,
    zone_dist_freq: int = 15,
    zone_time_offset: int = 20,
):
    target_zone_ratio = zone_dist_time / zone_dist_freq
    kdtree_freq = peak_freq_indexes * target_zone_ratio
    kdtree_time = peak_time_indexes - zone_time_offset - zone_dist_time
    radius = zone_dist_time

    tree = KDTree(np.column_stack([kdtree_time, kdtree_freq]), metric="manhattan")
    query_time = peak_time_indexes
    query_freq = kdtree_freq

    paired_peaks = tree.query_radius(
        np.column_stack([query_time, query_freq]), r=radius
    )
    return paired_peaks


def delaunay_peak_pairs(
    peak_freq_indexes: np.ndarray,
    peak_time_indexes: np.ndarray,
):
    tri = Delaunay(np.column_stack([peak_time_indexes, peak_freq_indexes]))
    edges = np.concatenate(
        [tri.simplices[:, :2], tri.simplices[:, 1:], tri.simplices[:, ::2]],
        axis=0,
    )

    mask = peak_time_indexes[edges[:, 1]] < peak_time_indexes[edges[:, 0]]
    edges = np.where(mask[:, None], edges[:, ::-1], edges)
    edges = np.unique(edges, axis=0)
    results = [[] for _ in range(len(peak_time_indexes))]
    for a, b in edges:
        results[a].append(b)
    return [np.array(x) for x in results]


def build_hash_table(
    peak_freq_indexes: List[int],
    peak_time_indexes: List[int],
    paired_peaks_indexes: Iterable[np.ndarray],
) -> Iterable[Tuple[Tuple[int, int, int], int]]:
    anchor2hashes = lambda anchor_idx, paired_indexes: [
        (
            (
                peak_freq_indexes[anchor_idx],
                peak_freq_indexes[paired_idx],
                peak_time_indexes[paired_idx] - peak_time_indexes[anchor_idx],
            ),
            peak_time_indexes[anchor_idx],
        )
        for paired_idx in paired_indexes
    ]

    hashes = chain.from_iterable(
        map(anchor2hashes, range(len(peak_time_indexes)), paired_peaks_indexes)
    )
    return hashes


# mongodb _id is 24 characters long
hashkey2bytes = lambda f1, f2, dt: f"{f1:08x}{f2:08x}{dt:08x}"


def hashes2hash_table(
    hashes: Iterable[Tuple[Tuple[int, int, int], int]], song_id: str
) -> dict:
    hash_table = defaultdict(list)
    for k, t in starmap(lambda k, t: (hashkey2bytes(*k), t), hashes):
        hash_table[k].append((song_id, t))
    return hash_table


def path2hashes(
    path: PathLike,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    dist_freq: int = 7,
    dist_time: int = 15,
    thresh_ratio: float = 0.01,
    peak_pair_func: Callable[
        [np.ndarray, np.ndarray], Iterable[np.ndarray]
    ] = delaunay_peak_pairs,
):
    spec = wav2spec(path, sr, n_fft, hop_length)
    peak_freq_indexes, peak_time_indexes = compute_constellation_map(
        spec, dist_freq, dist_time, thresh_ratio
    )

    peak_pairs = peak_pair_func(peak_freq_indexes, peak_time_indexes)

    peak_freq_indexes = peak_freq_indexes.tolist()
    peak_time_indexes = peak_time_indexes.tolist()

    hashes = build_hash_table(peak_freq_indexes, peak_time_indexes, peak_pairs)
    return hashes


def fingerprintBuilder(
    database_folder: str,
    fingerprints_path: str,
    hash_func: Callable[[PathLike], Iterable[Tuple[Tuple[int, int, int], int]]],
):
    files = list(Path(database_folder).glob("*.wav"))
    total_files = len(files)
    hashes = map(
        hash_func,
        files,
    )

    hash_tables = map(
        hashes2hash_table,
        hashes,
        map(lambda f: f.stem, files),
    )

    def inplace_extend(x, y):
        for k, v in y.items():
            tmp = x.get(k, [])
            tmp.extend(v)
            x[k] = tmp
        return x

    merged_database_hash_table = reduce(
        inplace_extend,
        tqdm(hash_tables, total=total_files, desc="Building fingerprint database"),
    )

    with open(Path(fingerprints_path) / "fingerprints.json", "w") as f:
        f.write(json.dumps(merged_database_hash_table))
