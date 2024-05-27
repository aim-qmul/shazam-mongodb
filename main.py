import argparse
from pymongo import MongoClient
from functools import reduce
from pymongo.collection import Collection
from pathlib import Path
from os import PathLike
from tqdm import tqdm
from bson.objectid import ObjectId
from collections import Counter
from typing import List

from fp import path2hashes, hashes2hash_table
from ai import get_time_differences


def fingerprintBuilder(
    collection: Collection,
    database_folder: str,
):
    files = list(Path(database_folder).glob("*.wav"))
    total_files = len(files)
    hashes = map(
        path2hashes,
        files,
    )

    hash_tables = map(
        hashes2hash_table,
        hashes,
        map(lambda f: f.stem, files),
    )

    def extend_data(col: Collection, data: dict):
        for k, v in data.items():
            # k is the hash (_id)
            # v is the list of tuples (song_id, time)
            col.update_one(
                {"_id": ObjectId(k)}, {"$push": {"values": {"$each": v}}}, upsert=True
            )
        return col

    # reduce(
    #     extend_data,
    #     tqdm(hash_tables, total=total_files, desc="Building fingerprint database"),
    #     collection,
    # )

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
    print(len(restuls.inserted_ids))

    # for k, v in tqdm(merged_database_hash_table.items()):
    #     collection.insert_one({"_id": ObjectId(k), "values": v})


def audioIdentification(
    collection: Collection,
    files: List[Path],
) -> List[str]:
    total_files = len(files)
    hashes = map(
        path2hashes,
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


def main(db_path: str, q_path: str):
    client = MongoClient("localhost", 28000)
    db = client["test"]
    collection = db["fingerprints"]

    # Erase the collection
    collection.delete_many({})
    fingerprintBuilder(collection, db_path)

    q_files = list(Path(q_path).glob("*.wav"))
    results = audioIdentification(collection, q_files)
    labels = [f.stem for f in q_files]

    # evaluation
    hit = [r.find(l) + 1 for r, l in zip(labels, results)]

    print("Hit rate: ", sum(hit) / len(hit))

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=str)
    parser.add_argument("q_path", type=str)
    args = parser.parse_args()

    main(args.db_path, args.q_path)
