#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:56:12 2021

@author: john
"""

import unittest
import numpy as np

from sklearn import datasets
from popfnn.kohonen import self_organizing_learning

class test_kohonen(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_kohonen, self).__init__(*args, **kwargs)
        self.dataset = datasets.load_diabetes()

    def test_self_organizing_learning(self):
        
        centers, widths = self_organizing_learning(self.dataset.data, [3]*self.dataset.data.shape[1])
        print(centers)
        print(widths)
        # self.assertTrue(results == 3)
        # self.fail('Test not implemented yet.')

if __name__ == '__main__':
    unittest.main()