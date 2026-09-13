# Finite Element Methods for Obstacle Problems and American Optimal Stopping

Numerical Practicum project at the University of Zurich, supervised by Dr. Benedikt Grässle.

This repository develops finite element methods for one-dimensional obstacle problems and explores their connection to optimal stopping in mathematical finance.

The core project starts from a variational inequality with a pointwise obstacle constraint, discretizes it using piecewise-linear P1 finite elements, and studies the resulting finite-dimensional complementarity problem.

As a self-directed extension, the same obstacle-problem framework is applied to American put option pricing under the Black-Scholes model. The option payoff acts as the obstacle, while the free boundary corresponds to the optimal early-exercise boundary.

## Project overview

The repository currently includes:

- P1 finite element assembly for one-dimensional obstacle problems,
- bound-constrained quadratic optimization and complementarity checks,
- validation against an exact obstacle-problem benchmark,
- L2 and H1-seminorm convergence experiments,
- American put pricing using P1 finite elements and implicit time stepping,
- numerical recovery of the American early-exercise boundary.

Current numerical extensions include a primal-dual active-set solver and higher-order P2 finite elements.

## American option pricing

The American-option extension applies the same obstacle-problem framework to optimal stopping.

For an American put with payoff $g(S)=\max(K-S,0)$, the option value satisfies the obstacle condition $V(t,S)\geq g(S)$.

In the continuation region, where $V(t,S)>g(S)$, the Black-Scholes pricing PDE holds. In the exercise region, $V(t,S)=g(S)$. The interface between these regions is the free boundary and represents the optimal early-exercise boundary.

The implementation transforms the Black-Scholes problem to log-price coordinates, applies P1 finite elements in space and implicit Euler in time, and solves the resulting bound-constrained quadratic problem at each time step.

For the benchmark parameters $S_0=100$, $K=100$, $r=0.05$, $\sigma=0.20$, and $T=1$, the current implementation produces an American put value of approximately $6.09$, compared with a European Black-Scholes value of approximately $5.57$.

See [`american_options/README.md`](american_options/README.md) for the mathematical formulation and implementation details.

### Numerical results

| American put value and payoff | Early-exercise boundary |
| --- | --- |
| ![American put value](figures/american_put_value.png) | ![American exercise boundary](figures/american_put_boundary.png) |

The American option value stays above the immediate-exercise payoff. The recovered free boundary separates the exercise and continuation regions.

## Continuous problem

Let $[a,b]$ be a bounded interval. We consider the energy functional

$$
J(v) = \frac{1}{2}\int_a^b |v'(x)|^2\,dx - \int_a^b f(x)v(x)\,dx.
$$

The admissible set is

$$
K = \{v \in H_0^1(a,b) \mid v \geq \psi\}.
$$

The obstacle problem is to find the minimizer of the energy over this admissible set:

$$
u = \mathop{\mathrm{argmin}}_{v \in K} J(v).
$$

The corresponding variational inequality is

$$
\int_a^b u'(x)(v-u)'(x)\,dx
\geq
\int_a^b f(x)(v-u)(x)\,dx
\quad \text{for all } v \in K.
$$

Formally, the obstacle problem satisfies the complementarity conditions

$$
u-\psi \geq 0,
\qquad
-u''-f \geq 0,
\qquad
(u-\psi)(-u''-f)=0.
$$

In the free region, where $u>\psi$,

$$
-u''=f.
$$

In the contact region, the obstacle constraint is active and

$$
u=\psi.
$$

## P1 finite element discretization

Let

$$
a=x_0<x_1<\cdots<x_N=b
$$

be a partition of the interval into $N$ finite elements.

We approximate the continuous solution using the space of continuous piecewise-linear functions that vanish at the boundary.

The nodal hat functions associated with the interior nodes are denoted by

$$
\lambda_1,\ldots,\lambda_{N-1}.
$$

A discrete function can be written as

$$
v_h(x)=\sum_{i=1}^{N-1}V_i\lambda_i(x).
$$

Because the hat functions satisfy

$$
\lambda_i(x_j)=\delta_{ij},
$$

the coefficients are exactly the values at the interior mesh nodes:

$$
V_i=v_h(x_i).
$$

## Discrete obstacle

The obstacle is approximated by its piecewise-linear nodal interpolant

$$
\psi_h=I_h\psi.
$$

At the interior nodes, define

$$
\Psi_i=\psi(x_i).
$$

The obstacle constraint becomes

$$
V_i\geq\Psi_i,
\qquad
i=1,\ldots,N-1.
$$

Thus the discrete admissible set consists of all piecewise-linear finite element functions that lie above the interpolated obstacle.

## Matrix formulation

Substituting

$$
v_h(x)=\sum_{i=1}^{N-1}V_i\lambda_i(x)
$$

into the energy functional gives

$$
J(v_h)=\frac{1}{2}V^{T}AV-F^{T}V.
$$

The stiffness matrix is defined by
$A_{ij}=\int_a^b \lambda_i'(x)\lambda_j'(x)\,dx$.

The load vector is defined by
$F_i=\int_a^b f(x)\lambda_i(x)\,dx$.

For a uniform mesh with mesh size

$$
h=\frac{b-a}{N},
$$

the stiffness matrix is tridiagonal with

$$
A_{ii}=\frac{2}{h}
$$

and

$$
A_{i,i+1}=A_{i+1,i}=-\frac{1}{h}.
$$

All other entries are zero.

The discrete obstacle problem is therefore

$$
\min_{V\geq\Psi}
\left(
\frac{1}{2}V^{T}AV-F^{T}V
\right).
$$

Its discrete complementarity conditions are

$$
V-\Psi\geq0,
$$

$$
AV-F\geq0,
$$

and, for every interior node $i$,

$$
(V_i-\Psi_i)(AV-F)_i=0.
$$

The vector $AV-F$ is the discrete analogue of the continuous reaction term $-u''-f$. At a free node the residual vanishes, while at a contact node it may be positive.

## Benchmark problem

The implementation is tested on the example discussed in the Numerical Practicum.

The domain is $[0,1]$ and the load is

$$
f(x)=-1.
$$

For a parameter

$$
0<\alpha<\frac{1}{2},
$$

the obstacle is

$$
\psi(x)=x(1-x)-\frac{3}{2}\alpha^2.
$$

The exact solution is defined piecewise.

For $0\leq x<\alpha$,

$$
u(x)=\frac{x^2}{2}+(1-3\alpha)x.
$$

For $\alpha\leq x\leq1-\alpha$,

$$
u(x)=\psi(x).
$$

For $1-\alpha<x\leq1$,

$$
u(x)=\frac{(1-x)^2}{2}+(1-3\alpha)(1-x).
$$

The exact contact region is therefore

$$
[\alpha,1-\alpha].
$$

Because an exact solution is available, the finite element approximation can be validated directly.

## Numerical experiments

The repository contains two main numerical experiments.

`experiments/run_example.py` solves the benchmark problem for a fixed mesh and compares:

- the exact solution,
- the P1 finite element solution,
- the obstacle.

`experiments/convergence.py` repeats the computation for increasingly fine meshes and evaluates the numerical error.

For the benchmark with $\alpha=0.2$, the numerical experiments give approximately

$$
\|u-u_h\|_{L^2}=O(h^2)
$$

and

$$
|u-u_h|_{H^1}=O(h).
$$

## Project structure

```text
obstacle-problem-fem/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── fem.py
│   ├── problem.py
│   └── solver.py
├── experiments/
│   ├── run_example.py
│   └── convergence.py
└── figures/
```

The main files have the following roles:

- `src/problem.py` defines the load, obstacle, and exact benchmark solution.
- `src/fem.py` constructs the mesh and assembles the P1 finite element system.
- `src/solver.py` solves the bound-constrained quadratic optimization problem.
- `experiments/run_example.py` runs and visualizes one benchmark solution.
- `experiments/convergence.py` performs the mesh-refinement and convergence study.

## Running the code

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the benchmark problem:

```bash
python experiments/run_example.py
```

Run the convergence study:

```bash
python experiments/convergence.py
```

## Requirements

- Python
- NumPy
- SciPy
- Matplotlib