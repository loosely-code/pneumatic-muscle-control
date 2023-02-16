import numpy as np

class signal_sin:
    def __init__(self, Amp, Period, Center, T_sample ):
        self.Amp = Amp
        self.Period = Period
        self.Center = Center
        self.W = (2 * np.pi) / self.Period
        self.T_sample = T_sample
        self.out = np.zeros(2,dtype=float)
        self.t_global = 0.0
        self.reset()

    def search(self,Time):
        result = self.Center + self.Amp * (np.sin(self.W * Time))
        result_dot = self.Amp * self.W * np.cos(self.W * Time)
        return result, result_dot
    
    def reset(self):
        self.out[0] = self.search(0)[0]
        self.out[1] = self.search(0)[1]
        self.t_global = 0.0
        return self.out

    def step(self):
        self.t_global += self.T_sample
        self.out[0],self.out[1] = self.search(self.t_global)
        return self.out

    
