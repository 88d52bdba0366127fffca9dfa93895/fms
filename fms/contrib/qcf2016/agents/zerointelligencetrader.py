#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Module defining ZeroIntelligenceTrader agent class.
"""

import random

import numpy as np
import pandas as pd
from fms import agents
from fms.utils import BUY, SELL
from fms.utils.exceptions import MissingParameter


class ZeroIntelligenceTrader(agents.Agent):
    """
    Simulate an agent taking random decisions with a given constrain, which we
    call the constrained zero-intelligence trader

    This agent subclass should have two keys in the
    args dict :
    - ref_price : the reference price (float)
    """

    def __init__(self, params, offset=0):
        agents.Agent.__init__(self, params, offset)
        self.ref_price = self.args[0]
        self.constrain = 0.1

    def act(self, world=None, market=None):
        """
        Return random order as a dict with keys in (direction, price, quantity)

        To avoid short selling as far as possible, if # of stocks
        is zero or negative, force BUY direction.

        To avoid levering up as far as possible, if money
        is zero or negative, force SELL.
        """
        if self.stocks > 0 and self.money > 0:
            direction = random.choice((BUY, SELL))
        elif self.stocks <= 0:
            # Short selling is forbidden
            direction = BUY
        else:
            # money<=0, levering is discouraged
            direction = SELL

        # return & price uniformly
        # random_return = random.uniform(-self.constrain, self.constrain)
        # return & price normal distribution with vnindex distribution
        # random_return = np.random.normal(loc=0.0005, scale=0.0154)
        # return & price normal distribution with HOSE distribution
        random_return = np.random.normal(loc=0.0003, scale=0.0269)
        random_return = min(random_return, world.limit_change)
        random_return = max(random_return, -world.limit_change)
        price = self.ref_price * (1 + random_return)

        # adjust the price as VN regulation
        # if self.ref_price < 10000:
        #     price = round(price, -1)
        # elif self.ref_price <= 49950:
        #     price = price - (price % 50)
        # else:
        #     price = round(price, -2)

        # sometime price is 0 b/c it's random_return
        if price != 0:
            max_buy = int(self.money / (float(price) * (1 + 0.15 / 100)))
        else:
            return {'direction': BUY, 'price': 0, 'quantity': 0}
        #
        # # when we can only buy some stocks, let short
        # if max_buy < 1:
        #     direction = SELL

        #
        if direction is SELL or max_buy <= 0:
            direction = SELL
            quantity = random.randint(1, self.stocks)
        else:
            quantity = random.randint(1, max_buy)
        #
        # volatilities = pd.rolling_std(market.exec_prices, window=5) * np.sqrt(5)
        # if len(market.exec_prices) != 0 and not np.isnan(volatilities[-1]):
        #     quantity = quantity * 10 * volatilities[-1]

        return {'direction': direction, 'price': price, 'quantity': quantity}


def _test():
    """
    Run tests in docstrings
    """
    import doctest
    doctest.testmod(optionflags=+doctest.ELLIPSIS)


if __name__ == '__main__':
    _test()
