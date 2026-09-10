# Finite Element Discretization of a 1D Obstacle Problem

Numerical Practicum project at the University of Zurich.

This repository implements a piecewise-linear finite element discretization of a one-dimensional obstacle problem.

## Continuous problem

We consider the energy functional

$$
J(v)
=
\frac{1}{2}\int_0^1 |v'(x)|^2\,dx
-
\int_0^1 f(x)v(x)\,dx.
$$

The obstacle problem consists of minimizing this functional over the admissible set

$$
K
=
\left\{
v\in H_0^1(0,1)
:
v\geq\psi
\right\}.
$$

Thus, the solution satisfies

$$
u
=
\operatorname*{arg\,min}_{v\in K} J(v).
$$

The corresponding variational inequality is

$$
\int_0^1 u'(x)(v-u)'(x)\,dx
\geq
\int_0^1 f(x)(v-u)(x)\,dx,
\qquad
\forall v\in K.
$$

## P1 finite element discretization

Let

$$
0=x_0<x_1<\dots<x_N=1
$$

be a mesh of the interval.

We approximate the solution using continuous piecewise-linear finite elements. With the nodal hat basis
$\{\lambda_i\}_{i=1}^{N-1}$, the discrete solution can be written as

$$
u_h(x)
=
\sum_{i=1}^{N-1} U_i\lambda_i(x).
$$

The discrete energy minimization problem becomes

$$
\min_{U\geq\Psi}
\left(
\frac{1}{2}U^\top A U-F^\top U
\right),
$$

where the stiffness matrix is given by

$$
A_{ij}
=
\int_0^1
\lambda_i'(x)\lambda_j'(x)\,dx,
$$

and the load vector by

$$
F_i
=
\int_0^1
f(x)\lambda_i(x)\,dx.
$$

The obstacle is interpolated at the mesh nodes, giving the nodal constraints

$$
U_i\geq\Psi_i=\psi(x_i).
$$

## Benchmark problem

The implementation is tested on the example

$$
f(x)=-1
$$

on the interval $[0,1]$, with obstacle

$$
\psi(x)
=
x(1-x)-\frac{3}{2}\alpha^2,
\qquad
0<\alpha<\frac{1}{2}.
$$

For this problem, an exact solution is available, allowing the numerical finite element solution to be compared directly with the analytical solution and its convergence to be studied.

## Running

Run the benchmark example:

```bash
python experiments/run_example.py
```

Run the convergence study:

```bash
python experiments/convergence.py
```