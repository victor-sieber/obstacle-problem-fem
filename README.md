# Finite Element Discretization of a 1D Obstacle Problem

Numerical Practicum project at the University of Zurich.

This repository implements a piecewise-linear finite element
discretization of a one-dimensional obstacle problem.

## Continuous problem

We minimize

\[
J(v)
=
\frac12\int_0^1 |v'(x)|^2\,dx
-
\int_0^1 f(x)v(x)\,dx
\]

over

\[
K
=
\{v\in H_0^1(0,1):v\geq\psi\}.
\]

The corresponding variational inequality is

\[
\int_0^1u'(v-u)'\,dx
\geq
\int_0^1f(v-u)\,dx
\qquad
\forall v\in K.
\]

## P1 finite element discretization

Using continuous piecewise-linear basis functions,

\[
u_h
=
\sum_{i=1}^{N-1}U_i\lambda_i,
\]

the discrete problem becomes

\[
\min_{U\geq\Psi}
\left(
\frac12U^TAU-F^TU
\right),
\]

where

\[
A_{ij}
=
\int_0^1
\lambda_i'(x)\lambda_j'(x)\,dx
\]

and

\[
F_i
=
\int_0^1
f(x)\lambda_i(x)\,dx.
\]

## Benchmark problem

The implementation is tested using

\[
f(x)=-1
\]

and

\[
\psi(x)
=
x(1-x)-\frac32\alpha^2,
\qquad
0<\alpha<\frac12.
\]

An exact solution is available, allowing numerical convergence
to be studied directly.

## Running

Run the benchmark example:

```bash
python experiments/run_example.py