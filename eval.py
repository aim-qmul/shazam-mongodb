import os
from fp import fingerprintBuilder, path2hashes
from ai import audioIdentification
# from audio_identification import audioIdentification

# Initialize
import numpy as np
from itertools import takewhile


def cw2_eval(outputFilename):
    """
    cw2_eval function: read output .txt file and compute
    audio identification evaluation metrics

    e.g. cw2_eval('sample-output.txt')
    """

    rank = 3

    # Open output file and read each line
    with open(outputFilename, "r") as f:
        fl = takewhile(lambda x: x != "\n", f.readlines())

        results = [
            tuple(
                filter(
                    len,
                    (lambda x: x.split("\t") if "\t" in x else x.split(" "))(
                        s.replace("\n", "", 1).replace(".wav", ""),
                    ),
                )
            )
            + ("fake",) * rank
            for s in fl
        ]
        results = [x[:4] for x in results]
        queries, *database_items = zip(*results)
        database_items = zip(*database_items)
        relevant_items = np.asarray(
            [
                np.asarray([q.find(d) for d in db]) + 1
                for q, db in zip(queries, database_items)
            ]
        )
        cumsum = np.cumsum(relevant_items, axis=1)
        pre = cumsum / np.arange(1, rank + 1)
        rec = cumsum

    avg_pre = pre.mean(0)
    avg_rec = rec.mean(0)
    print("Average precision at ranks 1-3: ", avg_pre)
    print("Average recall at ranks 1-3: ", avg_rec)

    return


db_path = "/homes/cy006/work/ecs7006-hw2-marks/database_recordings/"
q_path = "/homes/cy006/work/ecs7006-hw2-marks/query_recordings/"
fp_path = "fingerprints/"

os.makedirs(fp_path, exist_ok=True)

fingerprintBuilder(db_path, fp_path, path2hashes)
audioIdentification(q_path, fp_path, "output.txt", path2hashes)
cw2_eval("output.txt")
