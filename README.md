# Finite Element Discretization of a 1D Obstacle Problem

Numerical Practicum project at the University of Zurich.

This repository implements a piecewise-linear finite element discretization of a one-dimensional obstacle problem.

## Continuous problem

We consider the energy functional

$$ J(v) = \frac{1}{2}\int_0^1 |v'(x)|^2\,dx - \int_0^1 f(x)v(x)\,dx. $$

The admissible set is

$$ K = \left\{v\in H_0^1(0,1) : v\geq\psi\right\}. $$

The obstacle problem consists of finding

$$ u = \operatorname*{arg\,min}_{v\in K} J(v). $$

The corresponding variational inequality is

$$ \int_0^1 u'(x)(v-u)'(x)\,dx \geq \int_0^1 f(x)(v-u)(x)\,dx, \qquad \forall v\in K. $$

Formally, the obstacle problem can also be expressed through the complementarity conditions

$$ u-\psi\geq 0, \qquad -u''-f\geq 0, \qquad (u-\psi)(-u''-f)=0. $$

Thus, in the free region where $u>\psi$, the solution satisfies

$$ -u''=f, $$

while in the contact region the constraint $u=\psi$ is active.

## P1 finite element discretization

Let

$$ 0=x_0<x_1<\dots<x_N=1 $$

be a partition of the interval into $N$ finite elements.

The continuous space $H_0^1(0,1)$ is approximated by the finite-dimensional space of continuous piecewise-linear functions

$$ S_0^1(\Delta)=\operatorname{span}\{\lambda_1,\dots,\lambda_{N-1}\}, $$

where $\lambda_i$ denotes the standard nodal hat basis function.

Every discrete function can therefore be written as

$$ v_h(x)=\sum_{i=1}^{N-1}V_i\lambda_i(x). $$

The coefficients satisfy

$$ V_i=v_h(x_i), $$

so a discrete function is completely determined by its values at the interior mesh nodes.

### Discrete obstacle

The obstacle is approximated by its piecewise-linear nodal interpolant

$$ \psi_h=I_h\psi. $$

Writing

$$ \Psi_i=\psi(x_i), $$

the discrete obstacle constraint becomes

$$ V_i\geq\Psi_i, \qquad i=1,\dots,N-1. $$

The discrete admissible set is therefore

$$ K_h=\left\{v_h\in S_0^1(\Delta):v_h\geq\psi_h\right\}. $$

## Matrix formulation

Substituting

$$ v_h(x)=\sum_{i=1}^{N-1}V_i\lambda_i(x) $$

into the energy functional gives the finite-dimensional quadratic functional

$$ J(v_h)=\frac{1}{2}V^\top A V-F^\top V. $$

The stiffness matrix $A$ is defined by

$$ A_{ij}=\int_0^1\lambda_i'(x)\lambda_j'(x)\,dx, $$

and the load vector $F$ by

$$ F_i=\int_0^1f(x)\lambda_i(x)\,dx. $$

For a uniform mesh with mesh size $h$, the stiffness matrix has the tridiagonal form

$$ A=\frac{1}{h}\begin{pmatrix}2&-1&0&\cdots&0\\-1&2&-1&\ddots&\vdots\\0&-1&2&\ddots&0\\\vdots&\ddots&\ddots&\ddots&-1\\0&\cdots&0&-1&2\end{pmatrix}. $$

The discrete obstacle problem is therefore the convex quadratic optimization problem

$$ \min_{V\geq\Psi}\left(\frac{1}{2}V^\top A V-F^\top V\right). $$

The corresponding discrete complementarity conditions are

$$ V-\Psi\geq0, \qquad AV-F\geq0, \qquad (V_i-\Psi_i)(AV-F)_i=0. $$

## Benchmark problem

The implementation is tested on the example discussed in the Numerical Practicum.

The domain is $[0,1]$ and the load is

$$ f(x)=-1. $$

For a parameter $0<\alpha<\frac{1}{2}$, the obstacle is

$$ \psi(x)=x(1-x)-\frac{3}{2}\alpha^2. $$

The exact solution is

$$
u(x)=
\begin{cases}
\frac{x^2}{2}+(1-3\alpha)x, & 0\leq x<\alpha,\\
\psi(x), & \alpha\leq x\leq 1-\alpha,\\
\frac{(1-x)^2}{2}+(1-3\alpha)(1-x), & 1-\alpha<x\leq1.
\end{cases}
$$

Hence, the exact contact region is

$$ [\alpha,1-\alpha]. $$

The availability of an exact solution allows the finite element approximation to be validated directly and its convergence under mesh refinement to be studied.

## Numerical experiments

The repository contains two main experiments.

`experiments/run_example.py` solves the benchmark problem for a fixed mesh and compares the P1 finite element solution with the exact solution and obstacle.

`experiments/convergence.py` repeats the computation for increasingly fine meshes and evaluates the numerical error.

For the benchmark with $\alpha=0.2$, the observed convergence rates are approximately

$$ \|u-u_h\|_{L^2}=O(h^2) $$

and

$$ |u-u_h|_{H^1}=O(h). $$

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
- `src/solver.py` solves the resulting bound-constrained quadratic optimization problem.
- `experiments/run_example.py` runs and visualizes one benchmark solution.
- `experiments/convergence.py` performs the mesh-refinement and convergence study.

## Running the code

Install the required Python packages with

```bash
pip install -r requirements.txt
```

Run the benchmark example with

```bash
python experiments/run_example.py
```

Run the convergence study with

```bash
python experiments/convergence.py
```

## Requirements

- Python
- NumPy
- SciPy
- Matplotlib