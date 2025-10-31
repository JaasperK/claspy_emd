import time
import unittest

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import covering

class EMDTest(unittest.TestCase):

    def test_earth_mover_distance(self):
        tssb = load_tssb_dataset(names=["Chinatown"])
        scores = []
        runtime = time.process_time()

        idx, (dataset, window_size, cps, time_series) = list(tssb.iterrows())[0]

        print(f"{dataset} with {len(time_series)} values.")
        
        clasp = BinaryClaSPSegmentation(distance="earth_movers_distance", validation=None)
        found_cps = clasp.fit_predict(time_series)
        score = np.round(covering({0: cps}, found_cps, time_series.shape[0]), 2)
        scores.append(score)
        
        runtime = np.round(time.process_time() - runtime, 3)
        score = np.mean(scores)

        print(f"Covering is: {score}")
        assert score >= 0.0
