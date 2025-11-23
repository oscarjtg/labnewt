"""Defines Callback class, which is called during Simulation.run()"""


class Callback:
    def __init__(self, func, interval, on_init):
        self.func = func
        self.interval = interval
        self.on_init = on_init
