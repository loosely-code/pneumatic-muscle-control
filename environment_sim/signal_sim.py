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

class signal_square_wave:
    def __init__(self,Amp,Duty_circle,Period,Base,T_sample) -> None:
        self.Amp = Amp
        self.Duty_circle = Duty_circle
        self.Period = Period
        self.Base = Base
        self.T_sample = T_sample
        self.out = np.zeros(2,dtype=float)
        self.reset()    

    def reset(self):
        self.out = self.search(0)
        self.t_global = 0.0
        return self.out

    def step(self):
        self.t_global += self.T_sample
        self.out = self.search(self.t_global)
        return self.out

    def search(self,Time):
        t = Time % self.Period
        if t >= self.Period * (1-self.Duty_circle):
            result = self.Amp +self.Base
        else:
            result = self.Base
        result_vec=np.array([result,0])
        return result_vec

if __name__ == "__main__":

    T_total =10
    T_sample=0.01
    N_iter = int(T_total // T_sample)
    
    signal = signal_square_wave(
        Amp=1,
        Duty_circle=0.5,
        Period =5,
        Base=0.0,
        T_sample= T_sample
    )
    """
    signal = signal_sin(
        Amp=1,
        Period=5,
        Center=0,
        T_sample=T_sample)
    """
    # plot 
    plot_data_time = np.zeros(N_iter)
    plot_data_signal = np.zeros(N_iter)

    import matplotlib.pyplot as plt
    signal.reset()
    t_global = 0
    for iter in range(N_iter):
        t_global += signal.T_sample
        plot_data_time[iter] =t_global
        plot_data_signal[iter] = signal.step()[0]

    fig,ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.ion()
    ax.set_title("signal")
    ax.set_xlabel("t - s")
    ax.set_ylabel("s - m")
    ax.grid(True)
    ax.plot(plot_data_time,plot_data_signal)

    plt.ioff()
    plt.show()
    
