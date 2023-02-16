import numpy as np
class controller_1i1o:
    """
    basic controller frame for 1 input 1 output system

    Args:
        state_batch: [0:k,1:k-1,2:k-2]
        target_batch: [0:k,1:k-1,2:k-2]
        error_batch: [0:k,1:k-1,2:k-2]
        output_batch: [0:k+1,1:k,2:k-1]
    """
    def __init__(self) -> None:
        self.state_batch = np.zeros(3,dtype=float)
        self.target_batch = np.zeros(3,dtype=float)
        self.error_batch = np.zeros(3,dtype=float)
        self.output_batch = np.zeros(3,dtype=float)
        
    def reset(self,Initial_state,Initial_target):
        for index in range(3):
            self.state_batch[index] = Initial_state
            self.target_batch[index] = Initial_target
            self.error_batch[index] = Initial_target - Initial_state
            self.output_batch[index] = 0


        
    def observe(self,target,state):
        # observe data in k squence output data in k+1 squence
        # in observe state_batch [0:k,1:k-1,2:k-2]
        self.state_batch[2] = self.state_batch[1]
        self.state_batch[1] = self.state_batch[0]
        self.state_batch[0] = state

        self.target_batch[2] = self.target_batch[1]
        self.target_batch[1] = self.target_batch[0]
        self.target_batch[0] = target

        self.error_batch[2] = self.error_batch[1] 
        self.error_batch[1] = self.error_batch[0]
        self.error_batch[0] = target -state
        

    def step(self,target,state):
        self.observe(target,state)
        self.output_batch[2] =self.output_batch[1]
        self.output_batch[1] =self.output_batch[0]

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
        
        



