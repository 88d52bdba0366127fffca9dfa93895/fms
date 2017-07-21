#!/usr/bin/env python
"""
Asynchronous random with replace engine
"""

import random
import logging

from fms.engines import Engine

logger = logging.getLogger('fms.engines.asynchronousdeterministic')

class AsynchronousDeterministic(Engine):
    """
    Asynchronous engine, deterministic selection of agents
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
        Deterministically select agents one by one and let them speak on market.   
        As market is asynchronous, as soon as an agent speaks, do_clearing
        is called to execute any possible transaction immediately.
        """
        market.sellbook = world.state()['sellbook']
        logger.debug("Starting with sellbook %s" % market.sellbook)
        market.buybook = world.state()['buybook']
        logger.debug("Starting with buybook %s" % market.buybook)
        
        #Loop over the days
        for day in range(self.days):
            #Loop through all periods in the day, in each period we let one agent speak
            for time in range(self.daylength):
                #Select agent & make sure we never exceed bounds of agent array
                n_agents = len(agents)
                agt = time % n_agents
                #Get the order from the agent
                order = market.sanitize_order(agents[agt].speak())
                #Execute the order
                if market.is_valid(agents[agt], order):
                    #If logging is activated, log the order
                    if self.params.orderslogfile:
                        self.output_order(order)
                    #Record the order in the market
                    market.record_order(order, world.tick,
                            self.unique_by_agent)
                    #If book logging is activated, output the orderbook
                    if self.showbooks:
                        market.output_books(world.tick)
                    #Clear the order
                    market.do_clearing(world.tick)
                    #Update the market info
                    world.lastmarketinfo.update(
                            {'sellbook':market.sellbook, 'buybook':market.buybook})
                world.tick +=1
                #If specified, show the timer
                if self.params['timer']:
                    world.show_time(day, time, self.days*self.daylength)
            #At the end of the day clear the books        
            if self.clearbooksateod:
                market.clear_books()
        #Log the final books
        logger.debug("Ending with sellbook %s" % market.sellbook)
        logger.debug("Ending with buybook %s" % market.buybook)

if __name__ == '__main__':
    print AsynchronousRandWReplace()
