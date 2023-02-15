# pneumatic-muscle-control

## File arrangement

### controller_sim

- `controller.py` : basic framework of controllers
  - class `single_controller` : controller framework of 1 input and 1 output
  - class `multi_controller` : controller framework of multi input and multi output

### environment_sim

- MSD_env.py
  - class `MSD_env 1D` : test environment of a mass-spring-damping system

### demos

## Mathematical derivation

### Dynamics of the mass-spring-damping system

state vector of the system (the initial point x(0) is defined at when spring is fully relax):

$$
\mathbf{x}(t) = 
\begin{bmatrix}
x_1(t)\\
x_2(t)
\end{bmatrix} = 
\begin{bmatrix}
x(t)\\
\dot x(t)
\end{bmatrix}
$$

```math
\mathbf{x}(t) = 
\begin{bmatrix}
x_1(t)\\
x_2(t)
\end{bmatrix} = 
\begin{bmatrix}
x(t)\\
\dot x(t)
\end{bmatrix}
```

ODE of the system:

$$
M \ddot x(t) + K x(t) + D \dot x(t) - L = 0 
$$

Let:

$$
u(t) = L 
$$

the state space equation of the system is given as:

$$
\dot {\mathbf x} = 
\begin{bmatrix}
\dot x \\ \ddot x
\end{bmatrix} = 
\begin{bmatrix}
\dot x_1 \\ \dot x_2
\end{bmatrix} = 
\begin{bmatrix}
0 & 1 \\ 
- \frac{K}{M} & -\frac{D}{M}
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix} + 
\begin{bmatrix}
0 \\ \frac{1}{M} 
\end{bmatrix} u 
$$

which can also be noted as

$$
\dot {\mathbf x} = \mathbf A \mathbf x + \mathbf B \mathbf u
$$

Discretize the system with:

$$
\dot {\mathbf x}(k) = \frac{\mathbf x(k+1)-\mathbf x(k)}{T}
$$

Thus the discrete form of the system dynamics is given as:

$$

\frac{\mathbf x(k+1)-\mathbf x(k)}{T} = \mathbf A \mathbf x(k) + \mathbf B \mathbf u(k) 
$$

then

$$
\mathbf x(k+1) = (\mathbf I + T \mathbf A)\mathbf x(k) + T \mathbf B \mathbf u(k)
$$

Finally, 

$$

\mathbf x(k+1) = 
\begin{bmatrix}
1 & T \\ 
- \frac{KT}{M} & -\frac{DT}{M}+1
\end{bmatrix} 
\mathbf x(k) + 
\begin{bmatrix}
0 \\ \frac{T}{M} 
\end{bmatrix} \mathbf u

$$

the ouput equation of the system is 

$$
\mathbf y(k) = \begin{bmatrix} 1 & 0\end{bmatrix} \mathbf x(k) 
$$