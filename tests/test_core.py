import unittest
import numpy as np
from core import population_risk, empirical_risks, uniform_bound

class RiskTests(unittest.TestCase):
    def test_exact_noiseless_threshold(self):
        np.testing.assert_allclose(population_risk([0, .5, 1], 0), [.5, 0, .5])
    def test_random_labels_have_half_risk(self):
        np.testing.assert_allclose(population_risk(np.linspace(0, 1, 9), .5), .5)
    def test_empirical_classifier(self):
        np.testing.assert_allclose(empirical_risks([.1, .3, .6, .9], [0, 0, 1, 1], [0, .5, 1]), [.5, 0, .5])
    def test_bound_decreases_with_data(self):
        self.assertLess(uniform_bound(200, 41), uniform_bound(100, 41))
    def test_empty_sample_rejected(self):
        with self.assertRaises(ValueError): empirical_risks([], [], [.5])
    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError): uniform_bound(32, 41, delta=1)
