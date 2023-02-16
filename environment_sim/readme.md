# Simulation environment

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
-\frac{K}{M} & -\frac{D}{M}
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

To verify the result, set $M=1$ , $D=2$, $K=3/4$, Given a constant load input $L_0 = 1$, and a initial condition of $x(0) = 0$, the analitical solution of the ODE is 

$$
 x(s) = \frac {1}{s(s+\frac12)(s+\frac32)} =\frac{4}{3}\cdot \frac{1}{s} -2 \cdot \frac{1}{s+\frac12} + \frac23 \cdot \frac{1}{  s+\frac32}
$$

thus 

$$
x(t) = \frac{4}{3} -2 \cdot e^{-\frac12t} +\frac23 \cdot e^{-\frac32t}
$$