## American option pricing extension

The obstacle formulation also appears naturally in mathematical finance through optimal stopping.

For an American put with strike $K$, the immediate exercise payoff is

$g(S) = \max(K-S,0)$.

Because the holder may exercise at any time before maturity, the option value must satisfy

$V(t,S) \geq g(S)$.

Where $V(t,S) > g(S)$, early exercise is not optimal and the option satisfies the Black-Scholes PDE

$$
V_t
+
\frac{1}{2}\sigma^2 S^2 V_{SS}
+
rS V_S
-
rV
=
0.
$$

Where $V(t,S)=g(S)$, the option is in the exercise region. The boundary separating the continuation and exercise regions is the free boundary of the obstacle problem and represents the optimal early-exercise boundary.

### Log-price transformation

To connect the pricing problem with the finite element obstacle solver, introduce time to maturity $\tau=T-t$ and log-moneyness

$x=\log(S/K)$.

Writing $v(\tau,x)=V(T-\tau,Ke^x)$ gives

$$
v_\tau
=
\frac{1}{2}\sigma^2v_{xx}
+
\left(r-\frac{1}{2}\sigma^2\right)v_x
-
rv.
$$

An exponential transformation $v=e^{\alpha x+\beta\tau}u$ is used to remove the first-derivative and reaction terms. With suitable choices of $\alpha$ and $\beta$, the continuation equation becomes

$$
u_\tau
=
\frac{1}{2}\sigma^2u_{xx}.
$$

### Finite element and time discretization

The transformed equation is discretized in space using the same piecewise-linear P1 finite elements as the stationary obstacle problem.

The stiffness matrix is defined by
$A_{ij}=\int \lambda_i'(x)\lambda_j'(x)\,dx$.

Because the problem is time dependent, a mass matrix is also required:

$M_{ij}=\int \lambda_i(x)\lambda_j(x)\,dx$.

Using implicit Euler with time step $\Delta\tau$ gives the system matrix

$$
B
=
\frac{M}{\Delta\tau}
+
\frac{1}{2}\sigma^2 A.
$$

At each time step, the American exercise constraint produces a convex quadratic obstacle problem of the form

$$
\min_{U\geq\Psi}
\left(
\frac{1}{2}U^TBU-b^TU
\right).
$$

The current implementation solves this bound-constrained problem with SciPy's L-BFGS-B optimizer.

The same complementarity structure as in the stationary obstacle problem therefore appears at every time step.

### Numerical example

The example uses the parameters

- spot price $S_0=100$,
- strike $K=100$,
- risk-free rate $r=0.05$,
- volatility $\sigma=0.20$,
- maturity $T=1$ year.

With 200 P1 elements and 200 implicit-Euler time steps, the numerical American put value is approximately $6.09$.

For comparison, the corresponding European Black-Scholes put value is approximately $5.57$.

The American value is higher because the holder has the additional right to exercise before maturity.

The implementation also estimates the free boundary separating the continuation and early-exercise regions.

### Outlook

The current financial extension intentionally uses the same generic L-BFGS-B optimizer as the first obstacle-problem implementation.

Planned numerical extensions include:

- replacing L-BFGS-B with a primal-dual active-set method,
- extending the spatial discretization from P1 to P2 finite elements,
- comparing convergence and computational efficiency,
- studying multidimensional obstacle problems arising from multi-asset or stochastic-volatility option models.