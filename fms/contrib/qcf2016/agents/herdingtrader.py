#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Module defining ZeroIntelligenceTrader agent class.
"""

import random

from fms import agents
from fms.utils import BUY, SELL
from fms.utils.exceptions import MissingParameter


class HerdingTrader(agents.Agent):
    """
    Simulate an agent taking herding behaviour

    This agent subclass should have two keys in the
    args dict :
    - ref_price : the reference price (float)

    """

    def __init__(self, params, offset=0):
        agents.Agent.__init__(self, params, offset)
        #
        self.ref_price = self.args[0]
        self.traders = self.args[1]
        # _lambda will be 100 if traders = 1000
        self._lambda = self.traders / 20

    def act(self, world=None, market=None):
        if self.stocks > 0 and self.money > 0:
            direction = random.choice((BUY, SELL))
        elif self.stocks <= 0:
            # Short selling is forbidden
            direction = BUY
        else:
            # money<=0, levering is discouraged
            direction = SELL
        #
        sign = 1 if direction is BUY else -1
        quantity = random.randint(1, self.traders)
        delta_return = sign * float(quantity) / (self._lambda * 100)
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
