#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Module defining RandomTrader agent class.
"""

import random
import logging

from fms import agents
from fms.utils import BUY, SELL
from fms.utils.exceptions import MissingParameter

logger = logging.getLogger('fms.agents.followtrader')

class FollowTrader(agents.Agent):
    """
    Simulate an agent pricing the security the same as the last market price
    See RandomTrader for further documentation of errors and exceptions raised

    """
    
    def __init__(self, params, offset=0):
        agents.Agent.__init__(self, params, offset)
        try:
            self.avgprice = self.args[0]
        except (AttributeError, IndexError):
            raise MissingParameter, 'avgprice'
        try:
            self.maxfluct = self.args[1]
        except IndexError:
            raise MissingParameter, 'maxfluct'
        try:
            self.maxbuy = self.args[2]
        except IndexError:
            raise MissingParameter, 'maxbuy'
        del self.args

    def act(self, world=None, market=None):
        """
        Return random order as a dict with keys in (direction, price, quantity).
        """
        if self.stocks > 0:
            direction = random.choice((BUY, SELL))
        else:
            direction = BUY
        if True:
            try:
                #Just follow the last price in own pricing
                self.avgprice = market.lastprice
            except AttributeError:
                self.avgprice = 100
                logger.warning("No market, no avgprice, avgprice set to 100")
        price = random.uniform(self.avgprice*(100-self.maxfluct), 
                self.avgprice*(100+self.maxfluct))/100.
        
        
        if direction:
            maxq = self.stocks
        else:
            maxq = min(self.maxbuy, int(self.money/price))
        try:
            quantity = random.randint(1, maxq)
        except ValueError:
            quantity = 1
        return {'direction':direction, 'price':price, 'quantity':quantity}

def _test():
    """
    Run tests in docstrings
    """
    import doctest
    doctest.testmod(optionflags=+doctest.ELLIPSIS)

if __name__ == '__main__':
    _test()
