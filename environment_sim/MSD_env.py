import numpy as np
class MSD_env:
    """
    1D mass-spring-damping enviroment 

    Attributes:
    M: mass
    K: spring
    D: damping
    T_sample: sample time
    load: external load apply on the system
    state: state vector of the system [x1=x,x2=\dot x]^T
    t_global: global time 
    """
    def __init__(self,M,K,D,T_sample) -> None:
        # set constant parameters
        self.T_sample = T_sample
        self.M = M
        self.K = K 
        self.D = D
        # define transform matrics
        # A matrix in discrete time detail derivation in readme
        self.A_d = np.array(
            [[1,T_sample],
            [(-self.K*self.T_sample)/self.M,(-self.D*self.T_sample)/self.M +1]])
        # B matrix in discrete time detail derivation in readme
        self.B_d = np.array(
            [[0],
             [self.T_sample/self.M]]
        )
        # initialize variables
        self.reset()
        
    def reset(self) -> None:
        """
        reset the simulator to initial state

        reset all varibles
        """
        self.load = 0.0
        self.state = np.zeros([2,1],dtype=float)
        self.t_global =0
        return self.state[0,0], self.state[1,0]

    def step(self,Load):
        """
        update the system state by one step of sampling time

        Args:
            Load: external load (system input)

        Returns:
            self.state: updated state vector 
        """

        # update system input
        self.load = Load
        #calculate system state vector
        self.state = self.A_d.dot(self.state) + self.B_d.dot(self.load)
        #update t_global
        self.t_global += self.T_sample

        return self.state[0,0], self.state[1,0]
    
    def get_t_global(self):
        return self.t_global
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # define the simulation parameters
    T_total = 10
    T_sample = 0.010
    N_iter = int(T_total // T_sample)

    # define parameters of the environment
    Load_0 = 1
    M = 1.0
    D = 2.0
    K = 3/4
    env = MSD_env(
        M = M,
        K = K,
        D = D,
        T_sample= T_sample
    )

    # define plot_batch
    plot_batch_time = np.zeros(N_iter)
    plot_batch_pos = np.zeros(N_iter)
    plot_batch_pos_a = np.zeros(N_iter)
    plot_batch_model_error = np.zeros(N_iter)
    
    for iter in range(N_iter):
        state = env.step(Load=Load_0)
        plot_batch_pos[iter] = state[0]
        plot_batch_time[iter] = env.get_t_global()
        # analytical result, mathematical derivation in readme
        plot_batch_pos_a[iter] = 4/3 - 2*np.exp(-1/2 *env.get_t_global()) +2/3 *np.exp(-3/2 * env.get_t_global())
        plot_batch_model_error[iter] = plot_batch_pos_a[iter] - plot_batch_pos[iter]

    # plot
    fig,ax = plt.subplots(3, 1)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.ion()

    ax[0].set_title("position-discrete time simulation")
    ax[0].set_xlabel("t - s")
    ax[0].set_ylabel("pos - m")
    ax[0].grid(True)
    ax[0].plot(plot_batch_time,plot_batch_pos)

    ax[1].set_title("position - analytical result")
    ax[1].set_xlabel("t - s")
    ax[1].set_ylabel("pos - m")
    ax[1].grid(True)
    ax[1].plot(plot_batch_time,plot_batch_pos_a)

    ax[2].set_title("model error")
    ax[2].set_xlabel("t - s")
    ax[2].set_ylabel("error - m")
    ax[2].grid(True)
    ax[2].plot(plot_batch_time,plot_batch_model_error)
    
    plt.ioff()
    plt.show()



