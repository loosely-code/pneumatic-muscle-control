# 将项目根目录加入python查找目录,保证自定模块运行正常
import sys 
import os
current_dir = os.path.dirname(__file__)
root_dir = os.path.split(current_dir)[0] # 获取当前父目录路径
sys.path.append(root_dir) #加入python查找目录

import numpy as np
from controller_sim.Controller import controller_1i1o

class PID_position(controller_1i1o):
    def __init__(self,K_p,K_i,K_d) -> None:
        super().__init__()
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.integral = 0.0

    def set_param(self,param):
        self.K_p = param[0]
        self.K_i = param[1]
        self.K_d = param[2]
        self.parameter = np.array([self.K_p,self.K_i,self.K_d])

    def reset(self, Initial_state, Initial_target):
        super().reset(Initial_state, Initial_target)
        self.integral = 0.0

    def step(self, target, state):
        super().step(target, state)

        self.integral += self.error_batch[0]
        self.output_batch[0] = self.K_p * self.error_batch[0] + self.K_i * self.integral + self.K_d * (self.error_batch[0] - self.error_batch[1])
        return self.output_batch[0]

class PID_increment(controller_1i1o):
    def __init__(self,K_p,K_i,K_d) -> None:
        super().__init__()
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.parameter = np.array([self.K_p,self.K_i,self.K_d])
        self.T_matrix = np.array(
            [[1,-1,0],
            [1,0,0],
            [1,-2,1]])
        self.increment = 0

    def set_param(self,param):
        self.K_p = param[0]
        self.K_i = param[1]
        self.K_d = param[2]
        self.parameter = np.array([self.K_p,self.K_i,self.K_d])
    
    def reset(self, Initial_state, Initial_target):
        super().reset(Initial_state, Initial_target)
        self.increment = 0.0

    def step(self, target, state):
        super().step(target, state)
        increment = self.parameter.dot(self.T_matrix)
        self.increment = increment.dot(self.error_batch.T)
        self.output_batch[0] += self.increment
        return self.output_batch[0] 
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    # define the simulation parameters
    from environment_sim.MSD_env import MSD_env
    T_total = 10
    T_sample = 0.010
    N_iter = int(T_total // T_sample)

    # define parameters of the environment
    from environment_sim.MSD_env import MSD_env
    M = 1.0
    D = 2.0
    K = 3/4
    env = MSD_env(
        M = M,
        K = K,
        D = D,
        T_sample= T_sample
    )
    # define the parameters of the PID controller
    K_p = 80
    K_i = 0.5
    K_d = 80
    controller = PID_increment(
        K_p=K_p,
        K_i=K_i,
        K_d=K_d)
    
    # define target signal 
    from environment_sim.signal_sim import signal_sin
    Amp =5
    Period = 5
    Center = 0
    signal_target = signal_sin(
        Amp=Amp,
        Period=Period,
        Center=Center,
        T_sample=T_sample)
    # define plot_batch
    plot_data_time = np.zeros(N_iter)
    plot_data_pos = np.zeros(N_iter)
    plot_data_vec = np.zeros(N_iter)
    plot_data_target = np.zeros(N_iter)
    plot_data_target_dot = np.zeros(N_iter)
    plot_data_error = np.zeros(N_iter)
    plot_data_control = np.zeros(N_iter)

    # initialization of the simulation
    state = env.reset()
    target = signal_target.reset()
    controller.reset(Initial_state =state[0],Initial_target = target[0])
    error = controller.error_batch[0]
    action = controller.output_batch[0]

    for iter in range(N_iter):
        # save data in k squence
        plot_data_time[iter] = env.get_t_global()
        plot_data_pos[iter] = state[0]
        plot_data_vec[iter] = state[1]
        plot_data_target[iter] = target[0]
        plot_data_target_dot[iter] = target[1]
        plot_data_error[iter] = error
        plot_data_control[iter] = action
        # calcualte data in k+1 squence
        target = signal_target.step()
        action =controller.step(target=target[0],state=state[0])
        error = controller.error_batch[0]
        state = env.step(Load=action)

        
    # plot
    fig,ax = plt.subplots(4, 1)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.ion()

    ax[0].set_title("position")
    ax[0].set_xlabel("t - s")
    ax[0].set_ylabel("pos - m")
    ax[0].grid(True)
    ax[0].plot(plot_data_time,plot_data_target,"--")
    ax[0].plot(plot_data_time,plot_data_pos)
    
    ax[1].set_title("velocity")
    ax[1].set_xlabel("t - s")
    ax[1].set_ylabel("v - m/s")
    ax[1].grid(True)
    ax[1].plot(plot_data_time,plot_data_target_dot,"--")
    ax[1].plot(plot_data_time,plot_data_vec)

    ax[2].set_title("tracking error")
    ax[2].set_xlabel("t - s")
    ax[2].set_ylabel("error - m")
    ax[2].grid(True)
    ax[2].plot(plot_data_time,plot_data_error)

    ax[3].set_title("control torque")
    ax[3].set_xlabel("t - s")
    ax[3].set_ylabel("N - N")
    ax[3].grid(True)
    ax[3].plot(plot_data_time,plot_data_control)
    
    plt.ioff()
    plt.show()


