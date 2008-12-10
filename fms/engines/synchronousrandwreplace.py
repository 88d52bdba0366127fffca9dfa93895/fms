#!/usr/bin/env python
"""
Synchronous random with replace engine
"""

import random
import logging

from fms.engines import Engine

logger = logging.getLogger('fms.engines.synchronousrandwreplace')

class SynchronousRandWReplace(Engine):
    """
    Synchronous engine, random sampling of agents,
    with replacement.
    After given number of periods, calls Market clearing.
    """

    def __init__(self, parameters=None, offset=0):
        """
        Constructor. Takes parameters from config.
        Seeds ramdom engine from parameter.randomseed, if any.
        """
        Engine.__init__(self, parameters, offset)
        self.params = parameters
        self.rank = offset
        if parameters:
            random.seed(parameters['randomseed'])

    def run(self, world, agents, market):
        """
        Sample agents (with replacement) and let them speak on market.   
        As market is synchronous, do_clearing is called after 
        self.days*self.daylength periods.
        """
        market.sellbook = world.state()['sellbook']
        logger.debug("Starting with sellbook %s" % market.sellbook)
        market.buybook = world.state()['buybook']
        logger.debug("Starting with buybook %s" % market.buybook)
        now = world.time
        while world.time < now + self.days*self.daylength: 
            agt = random.randint(0, len(agents)-1)
            order = agents[agt].act()
            valid = market.is_valid(agents[agt], order)
            if valid:
                if self.params.orderslogfile:
                    print >> self.params.orderslogfile, \
                            "%(direction)s;%(price).2f;%(quantity)d" % order
                market.record_order(agents[agt], order, world.time)
                world.lastmarketinfo.update(
                        {'sellbook':market.sellbook, 'buybook':market.buybook})
            world.time +=1
        market.do_clearing(world.time)
        logger.debug("Ending with sellbook %s" % market.sellbook)
        logger.debug("Ending with buybook %s" % market.buybook)

if __name__ == '__main__':
    print SynchronousRandWReplace()