# 将项目根目录加入python查找目录,保证自定模块运行正常
import sys 
import os
current_dir = os.path.dirname(__file__)
root_dir = os.path.split(current_dir)[0] # 获取当前父目录路径
sys.path.append(root_dir) #加入python查找目录

import numpy as np
from controller_sim.Controller import controller_1i1o

"""
class template_controller_1i1o(controller_1i1o):
    def __init__(self) -> None:
        super().__init__()
    
    def step(self,target,state):
        # in output_batch [0:k+1,1:k,2:k-1]
        super().step(target,state)
        self.output_batch[0] += 1
        return 
"""

class SMC_MSD(controller_1i1o):
    def __init__(self,M,K,D,C,Ro) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.D = D
        self.C = C
        self.Ro = Ro
        self.s = 0.0
        self.tracking_state = np.zeros(2,dtype=float)
        self.state = np.zeros(2,dtype=float)
        self.target_ddot = 0
        self.reset(Initial_state=0,Initial_target=0)

    def reset(self, Initial_state: float, Initial_target: float):
        super().reset(Initial_state, Initial_target)
        self.state[0] = self.state_batch[0]
        self.state[1] = self.state_batch[0] - self.state_batch[1] 
        self.tracking_state[0] =self.error_batch[0]
        self.tracking_state[1] =self.error_batch[0]-self.error_batch[1]
        self.target_ddot = self.target_batch[0]-2*self.target_batch[1]+self.target_batch[2]
        self.update_s()
        return self.output_batch[0]

    def update_s(self):
        self.s = self.C * self.tracking_state[0]+self.tracking_state[1]
    
    def step(self, target, state):
        super().step(target, state)
        self.state[0] = self.state_batch[0]
        self.state[1] = self.state_batch[0] - self.state_batch[1] 
        self.tracking_state[0] =self.error_batch[0]
        self.tracking_state[1] =self.error_batch[0]-self.error_batch[1]
        self.target_ddot = self.target_batch[0]-2*self.target_batch[1]+self.target_batch[2]
        self.update_s()
        self.output_batch[0] = self.M*(self.C*self.tracking_state[1]+ self.target_ddot+self.Ro*np.sign(self.s)) + self.D*self.state[1]+self.K*self.state[0]
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

    # define the parameters of the SMC controller
    controller = SMC_MSD(
        M=M,
        K=K,
        D=D,
        C=2000,
        Ro=40
    )

    from environment_sim.signal_sim import signal_square_wave
    
    signal_target = signal_square_wave(
        Amp=5,
        Duty_circle=0.8,
        Period =10,
        Base=0.0,
        T_sample= T_sample
    )

    # define plot_batch
    plot_data_time = np.zeros(N_iter)
    plot_data_pos = np.zeros(N_iter)
    plot_data_vec = np.zeros(N_iter)
    plot_data_target = np.zeros(N_iter)
    plot_data_target_dot = np.zeros(N_iter)
    plot_data_error = np.zeros(N_iter)
    plot_data_s = np.zeros(N_iter)
    plot_data_control = np.zeros(N_iter)

    # initialization of the simulation
    state = env.reset()
    target = signal_target.reset()
    controller.reset(Initial_state =state[0],Initial_target = target[0])
    error = controller.error_batch[0]
    action = controller.output_batch[0]
    s = controller.s

    for iter in range(N_iter):
        # save data in k squence
        plot_data_time[iter] = env.get_t_global()
        plot_data_pos[iter] = state[0]
        plot_data_vec[iter] = state[1]
        plot_data_target[iter] = target[0]
        plot_data_target_dot[iter] = target[1]
        plot_data_s[iter] = s
        plot_data_error[iter] = error
        plot_data_control[iter] = action

        # calcualte data in k+1 squence
        target = signal_target.step()
        action =controller.step(target=target[0],state=state[0])
        error = controller.error_batch[0]
        s = controller.s 
        state = env.step(Load=action)

    # plot
    fig,ax = plt.subplots(5, 1)
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
    ax[3].set_ylabel("L - N")
    ax[3].grid(True)
    ax[3].plot(plot_data_time,plot_data_control)

    ax[4].set_title("sliding varible")
    ax[4].set_xlabel("t - s")
    ax[4].set_ylabel("s")
    ax[4].grid(True)
    ax[4].plot(plot_data_time,plot_data_control)
    
    plt.ioff()
    plt.show()

        

