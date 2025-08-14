import time
import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import covering


def test_earth_mover_distance():
    tssb = load_tssb_dataset()
    scores = []
    runtime = time.process_time()

    print()
    for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
        if idx != 1: continue
        print("At idx:", idx)
        clasp = BinaryClaSPSegmentation(distance="earth_movers_distance", validation=None)
        found_cps = clasp.fit_predict(time_series)
        score = np.round(covering({0: cps}, found_cps, time_series.shape[0]), 2)
        scores.append(score)
        
        if len(found_cps) > 0:
            print("cps:", cps)
            print("pred:", found_cps)
        break

    runtime = np.round(time.process_time() - runtime, 3)
    score = np.mean(scores)

    print(score)

if __name__ == "__main__":
    test_earth_mover_distance()