#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Module defining ZeroIntelligenceTrader agent class.
"""

import random

import numpy as np
from fms import agents
from fms.utils import BUY, SELL
from fms.utils.exceptions import MissingParameter


class ThresholdTrader(agents.Agent):
    """
    Simulate an agent taking threshold behaviour

    This agent subclass should have two keys in the
    args dict :
    - ref_price : the reference price (float)
    """

    def __init__(self, params, offset=0):
        agents.Agent.__init__(self, params, offset)

        self.ref_price = self.args[0]
        self.traders = self.args[1]
        self.thetas = np.random.uniform(low=0.0001,
                                        high=0.07,
                                        size=self.traders)
        self.ss = np.random.uniform(low=0.0001,
                                    high=0.07,
                                    size=self.traders)
        # _lambda will be 100 if traders = 1000
        self._lambda = self.traders / 20

    def act(self, world=None, market=None):
        if self.stocks <= 0 or self.money <= 0:
            return {'direction': 0, 'price': 0, 'quantity': 0}
        # comute quanity
        quantity = 0
        for i in range(self.traders):
            # update threshold
            if i != 0 and market.lastprice is not None:
                s = self.ss[i]
                u = np.random.uniform(low=0, high=1)
                if u < s:
                    self.thetas[i] = np.abs(market.last_return)
                else:
                    self.thetas[i] = self.thetas[i-1]
            #
            if world.forecast_information[world.tick] > self.thetas[i]:
                quantity += 1
            elif world.forecast_information[world.tick] < -self.thetas[i]:
                quantity -= 1
            else:
                pass
        # no one want to trade
        if quantity == 0:
            return {'direction': 0, 'price': 0, 'quantity': 0}
        #
        sign = 1 if quantity > 0 else -1
        direction = BUY if quantity > 0 else SELL
        quantity = np.abs(quantity)
        delta_return = sign * float(np.abs(quantity)) / (self._lambda * 100)
        delta_return = min(delta_return, world.limit_change)
        delta_return = max(delta_return, -world.limit_change)
        price = self.ref_price * (1 + delta_return)
        #
        return {'direction': direction, 'price': price, 'quantity': quantity}


def _test():
    """
    Run tests in docstrings
    """
    import doctest
    doctest.testmod(optionflags=+doctest.ELLIPSIS)


if __name__ == '__main__':
    _test()
