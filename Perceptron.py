import numpy as np

class Perceptron:

    def __init__(self, l):
        self.h      = 0
        self.v      = 0
        self.d      = 0
        self.old_dw = 0
        # Layer
        self.l      = l
    

    def __str__(self):
        return "[ h: " +str(self.h) + " " + ", v: " +str(self.v) + " "+ ", d: " +str(self.d) + " " + ", l: " +str(self.l) + " ]"