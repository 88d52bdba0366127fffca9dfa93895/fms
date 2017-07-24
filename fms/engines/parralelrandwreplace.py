#!/usr/bin/env python
"""
Asynchronous random with replace engine
Using paralel computation of orders.
This engine is intended to use with agents which require heavy computation to get their orders.
"""

import random
import logging
import numpy as np

from fms.engines import Engine

from joblib import Parallel, delayed
import multiprocessing

logger = logging.getLogger('fms.engines.parralelrandwreplace')

"""
Get order helper function has to be declared outside of class to avoid issues with joblib
See: http://qingkaikong.blogspot.de/2016/12/python-parallel-method-in-class.html
"""
def get_order(agent,world,market):
    order = market.sanitize_order(agent.speak(world = world, market = market))
    if market.is_valid(agent, order):
        return order, agent
    return None, agent

class ParralelRandWReplace(Engine):
    """
    Asynchronous engine, random sampling of agents,
    with replacement.
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
        As market is asynchronous, as soon as an agent speaks, do_clearing
        is called to execute any possible transaction immediately.
        """
        market.sellbook = world.state()['sellbook']
        logger.debug("Starting with sellbook %s" % market.sellbook)
        market.buybook = world.state()['buybook']
        logger.debug("Starting with buybook %s" % market.buybook)
        
        #Loop over days
        for day in range(self.days):

            #We want as many threads as there are cpus
            cpu_count = multiprocessing.cpu_count()
            #get the orders for all agents, in parallel
            r = Parallel(n_jobs=cpu_count)(delayed(get_order)(agent,world,market) for agent in agents)
            orders, updated_agents = zip(*r)
            agents = updated_agents
            #Sample a subset of the agents
            sample = np.random.choice(orders,size=self.daylength)
            #Now we loop over orders and execute them one by one
            for order in sample:
                #Check whether there actually is an order to execute
                if order != None:
                    #Optional logging
                    if self.params.orderslogfile:
                        self.output_order(order)
                    #Record order to market
                    market.record_order(order, world.tick,
                            self.unique_by_agent)
                    #Optional display of books
                    if self.showbooks:
                        market.output_books(world.tick)
                    #Clear all open orders on market
                    market.do_clearing(world.tick)
                    #Update the world
                    world.lastmarketinfo.update(
                            {'sellbook':market.sellbook, 'buybook':market.buybook})
                world.tick +=1
                if self.params['timer']:
                    world.show_time(day, time, self.days*self.daylength)
            if self.clearbooksateod:
                market.clear_books()
        logger.debug("Ending with sellbook %s" % market.sellbook)
        logger.debug("Ending with buybook %s" % market.buybook)
    

if __name__ == '__main__':
    print AsynchronousRandWReplace()
