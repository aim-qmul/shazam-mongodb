from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, List
import json
from tqdm import tqdm
from collections import Counter

from fp import hashes2hash_table


def get_time_differences(
    query_table: Dict[str, List[Tuple[str, int]]],
    database_table: Dict[str, List[Tuple[str, int]]],
) -> List[Tuple[str, int]]:
    x = [
        (song_id, db_t - q_t)
        for hash, v in query_table.items()
        for song_id, db_t in database_table.get(hash, [])
        for _, q_t in v
    ]
    return x


def audioIdentification(
    query_path: str,
    fingerprints_path: str,
    output_file_path: str,
    hash_func: Callable[[PathLike], Iterable[Tuple[Tuple[int, int, int], int]]],
):
    files = list(Path(query_path).glob("*.wav"))
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

    with open(Path(fingerprints_path) / "fingerprints.json", "r") as f:
        database_table = json.load(f)

    top3_results = map(
        lambda query_table: Counter(
            get_time_differences(query_table, database_table)
        ).most_common(3),
        tqdm(hash_tables, total=total_files),
    )

    with open(output_file_path, "w") as f:
        for filename, res in zip(files, top3_results):
            f.write("\t".join([filename.stem] + [r[0] for r, _ in res]) + "\n")
