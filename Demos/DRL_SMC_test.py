if __name__ == "__main__":
    # 将项目根目录加入python查找目录,保证自定模块运行正常
    import sys 
    import os
    current_dir = os.path.dirname(__file__)
    root_dir = os.path.split(current_dir)[0] # 获取当前父目录路径
    sys.path.append(root_dir) #加入python查找目录

    import numpy as np
    from controller_sim.sliding_mode_MSD import SMC_MSD

    import matplotlib.pyplot as plt
    
    # define the simulation parameters
    from environment_sim.MSD_env import MSD_env
    T_total = 10
    T_sample = 0.005
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
        Ro=20
    )

    from environment_sim.signal_sim import signal_square_wave
    from environment_sim.signal_sim import signal_sin
    """
    signal_target = signal_square_wave(
        Amp=0.5,
        Duty_circle=0.5,
        Period =5,
        Base=0.0,
        T_sample= T_sample
    )
    """

    signal_target = signal_sin(
        Amp=5,
        Period= 5,
        Center= 0,
        T_sample= T_sample
    )

    signal_d = signal_sin(
        Amp=5,
        Period= 1,
        Center= 0,
        T_sample= T_sample
    )

    from controller_sim.reinforcement_learning import DDPG
    ddpg = DDPG(state_dim=2,
                action_dim=1,
                replacement=dict(name='soft', tau=0.005),
                memory_size=2000)
    VAR =2
    Max_episode =10
    
    # define plot_batch
    plot_data_epreward = np.zeros(Max_episode)
    plot_data_time = np.zeros(N_iter)
    plot_data_pos = np.zeros(N_iter)
    plot_data_vec = np.zeros(N_iter)
    plot_data_target = np.zeros(N_iter)
    plot_data_target_dot = np.zeros(N_iter)
    plot_data_error = np.zeros(N_iter)
    plot_data_s = np.zeros(N_iter)
    plot_data_control = np.zeros(N_iter)
    plot_data_a = np.zeros(N_iter)

    

    fig, ax = plt.subplots(5, 1)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.ion()

    for i_ep in range(Max_episode):
        ep_reward =0
        # initialization of the simulation
        state = env.reset()
        target = signal_target.reset()
        controller.reset(Initial_state =state[0],Initial_target = target[0])
        error = controller.error_batch[0]
        action = controller.output_batch[0]
        s = controller.s
        d = 0
        a=0

        for iter in range(N_iter):
            # save data in k squence
            plot_data_time[iter] = env.get_t_global()
            plot_data_pos[iter] = state[0]
            plot_data_vec[iter] = state[1]
            plot_data_target[iter] = target[0]
            plot_data_target_dot[iter] = target[1]
            plot_data_s[iter] = s
            plot_data_error[iter] = error
            plot_data_control[iter] = 10*a
            plot_data_a[iter] = a

            target = signal_target.step()
            d = signal_d.step()[0]
            action =controller.step(target=target[0],state=state[0])
            error = controller.error_batch[0]
            s = controller.s 
            a = ddpg.choose_action(controller.tracking_state)[0]
            a = np.random.normal(a,VAR)
            state = env.step(Load=d+10*a)

            reward = -(0.5* controller.error_batch[0]**2)

            ddpg.store_memory(controller.state,state,a,reward/100)

            if ddpg.count_iter > ddpg.memory_size:
                VAR *= 0.95
                ddpg.learn()

            ep_reward+=reward

            if iter ==N_iter-1:
                plot_data_epreward[i_ep] =ep_reward

                print('Episode:', i_ep, ' Episode reward: %i' % int(ep_reward), 'VAR: %.2f' % VAR, )

                ax[0].cla()
                ax[0].set_title("position")
                ax[0].set_xlabel("t - s")
                ax[0].set_ylabel("pos - m")
                ax[0].grid(True)
                ax[0].plot(plot_data_time,plot_data_target,"--")
                ax[0].plot(plot_data_time,plot_data_pos)

                ax[1].cla()
                ax[1].set_title("velocity")
                ax[1].set_xlabel("t - s")
                ax[1].set_ylabel("v - m/s")
                ax[1].grid(True)
                ax[1].plot(plot_data_time,plot_data_target_dot,"--")
                ax[1].plot(plot_data_time,plot_data_vec)

                ax[2].cla()
                ax[2].set_title("tracking error")
                ax[2].set_xlabel("t - s")
                ax[2].set_ylabel("error - m")
                ax[2].grid(True)
                ax[2].plot(plot_data_time,plot_data_error)

                ax[3].cla()
                ax[3].set_title("control torque")
                ax[3].set_xlabel("t - s")
                ax[3].set_ylabel("L - N")
                ax[3].grid(True)
                ax[3].plot(plot_data_time,plot_data_control)

                ax[4].cla()
                ax[4].set_title("sliding varible")
                ax[4].set_xlabel("t - s")
                ax[4].set_ylabel("s")
                ax[4].grid(True)
                ax[4].plot(plot_data_time,plot_data_control)
                ax[4].plot(plot_data_time,plot_data_a)
                plt.pause(0.1)


    plt.ioff()
    plt.show()






    

