import time
import unittest

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import covering

class EMDTest(unittest.TestCase):

    def test_earth_mover_distance(self):
        tssb = load_tssb_dataset()
        scores = []
        runtime = time.process_time()

        idx, (dataset, window_size, cps, time_series) = list(tssb.iterrows())[1]
        print()

        print(list(tssb.iterrows())[1])
        print(len(time_series))
        
        print("At idx:", idx)
        clasp = BinaryClaSPSegmentation(distance="earth_movers_distance", validation=None)
        print(1)
        found_cps = clasp.fit_predict(time_series)
        print(2)
        score = np.round(covering({0: cps}, found_cps, time_series.shape[0]), 2)
        print(3)
        scores.append(score)
        print(4)
        
        if len(found_cps) > 0:
            print("cps:", cps)
            print("pred:", found_cps)

        print(5)

        runtime = np.round(time.process_time() - runtime, 3)
        score = np.mean(scores)

        print(score)
