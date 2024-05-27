import argparse
from pymongo import MongoClient
from functools import reduce, partial
from pymongo.collection import Collection
from pathlib import Path
from os import PathLike
from tqdm import tqdm
from bson.objectid import ObjectId
from collections import Counter
from typing import List, Callable, Tuple, Iterable
import time

from fp import path2hashes, hashes2hash_table, wang_peak_pairs, delaunay_peak_pairs
from ai import get_time_differences


def fingerprintBuilder(
    collection: Collection,
    database_folder: str,
    path2hash_func: Callable[[PathLike], Iterable[Tuple[Tuple[int, int, int], int]]],
):
    files = list(Path(database_folder).glob("*.wav"))
    total_files = len(files)
    hashes = map(
        path2hash_func,
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

    restuls = collection.insert_many(
        [
            {"_id": ObjectId(k), "values": v}
            for k, v in merged_database_hash_table.items()
        ]
    )
    print(f"Number of inserted documents: {len(restuls.inserted_ids)}")


def audioIdentification(
    collection: Collection,
    files: List[Path],
    path2hash_func: Callable[[PathLike], Iterable[Tuple[Tuple[int, int, int], int]]],
) -> List[str]:
    total_files = len(files)
    hashes = map(
        path2hash_func,
        files,
    )

    hash_tables = map(
        hashes2hash_table,
        hashes,
        map(lambda f: f.stem, files),
    )

    restuls = []
    for query_table in tqdm(
        hash_tables, total=total_files, desc="Audio Identification"
    ):

        fp_dict = reduce(
            lambda x, y: x | y,
            [
                {str(x["_id"]): [tuple(y) for y in x["values"]]}
                for x in collection.find(
                    {"$or": [{"_id": ObjectId(k)} for k in query_table.keys()]}
                )
            ],
        )
        time_diffs = get_time_differences(query_table, fp_dict)

        top_song_id = Counter(time_diffs).most_common(1)[0][0][0]
        restuls.append(top_song_id)

    return restuls


def main(db_path: str, q_path: str, port: int, map_type: str):
    client = MongoClient("localhost", port)
    db = client["test"]
    collection = db["fingerprints"]

    # Erase the collection
    collection.delete_many({})

    match map_type:
        case "wang":
            peak_pair_func = wang_peak_pairs
        case "delaunay":
            peak_pair_func = delaunay_peak_pairs
        case _:
            raise ValueError("Invalid map type")

    path2hash_func = partial(
        path2hashes,
        peak_pair_func=peak_pair_func,
    )

    fingerprintBuilder(collection, db_path, path2hash_func)

    q_files = list(Path(q_path).glob("*.wav"))
    start = time.time()
    results = audioIdentification(collection, q_files, path2hash_func)
    elapsed = time.time() - start
    print(f"Averaged {elapsed / len(q_files) * 1000:.2f} ms per query")
    labels = [f.stem for f in q_files]

    # evaluation
    hit = [r.find(l) + 1 for r, l in zip(labels, results)]

    print("Hit rate: ", sum(hit) / len(hit))

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=str)
    parser.add_argument("q_path", type=str)
    parser.add_argument("--port", type=int, default=28000)
    parser.add_argument("--map", type=str, choices=["wang", "delaunay"], default="wang")
    args = parser.parse_args()

    main(args.db_path, args.q_path, args.port, args.map)
