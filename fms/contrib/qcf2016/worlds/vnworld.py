#!/usr/bin/env python
"""
A minimal and over simplistic world class
"""

import numpy as np
from fms import worlds


class VNWorld(worlds.World):
    """
    Minimal world class
    """
    def __init__(self, parameters=None):
        worlds.World.__init__(self)
        self.days = parameters['engines'][0]['daylength']
        self.forecast_information = np.random.normal(scale=0.03, size=self.days)
        # VN regulation
        self.limit_change = 0.07

    def state(self):
        """
        Nullworld only returns last market info (dict)
        """
        return self.lastmarketinfo


if __name__ == '__main__':
    print NullWorld()
