"""No-argument regression check for stable sine-perturbed P1/P2 obstacle FEM.

Compares with original PDAS+indicator at safely resolved mesh sizes.
This test does not change any CSV files or other experiment outputs.
"""
import numpy as np
from experiments.key.sine_dirac.problem import (
    solve_on_mesh, compute_errors, sine, _sine_p2_stiffness_action,
)
from src.fem import assemble_load_vector as assemble_load_p1
from src.fem_p2 import assemble_load_vector as assemble_load_p2, create_p2_nodes
from src.error_indicators import (
    compute_p1_residual_indicator, compute_p2_residual_indicator,
)
from src.multiplier_l2_transfer import p1_mass_action, p2_mass_action
from src.p1_nested_large import (
    assemble_p1_dirac, matvec, solve_pdas_p1,
)
from src.p2_nested_large import (
    assemble_p2_dirac, join_p2, p2_matvec, solve_pdas_p2,
)


def reference(nodes, degree, epsilon):
    regular = lambda x: -6.0 + epsilon * np.pi**2 * np.sin(np.pi*x)
    forcing = lambda x: epsilon * np.pi**2 * np.sin(np.pi*x)
    if degree == 1:
        lower, diagonal, upper, F0 = assemble_p1_dirac(nodes)
        action = lambda x: matvec(lower, diagonal, upper, x)
        G = sine(nodes[1:-1], epsilon)
        F = F0 + assemble_load_p1(nodes, forcing, quadrature_order=8)
        W, _ = solve_pdas_p1(
            nodes, lower, diagonal, upper, F-action(G), maxiter=2000,
        )
        U = W + G
        indicator = compute_p1_residual_indicator(
            nodes, U, action, F, regular,
        )
    else:
        F0v, F0m = assemble_p2_dirac(nodes)
        F0 = join_p2(F0v, F0m)
        G = sine(create_p2_nodes(nodes)[1:-1], epsilon)
        action = lambda x: p2_matvec(nodes, x)
        F = F0 + assemble_load_p2(nodes, forcing, quadrature_order=8)
        Fshift = F-action(G)
        m = len(nodes)-1
        Fvshift = np.zeros(m+1)
        Fvshift[1:-1] = Fshift[1::2]
        W, _ = solve_pdas_p2(
            nodes, Fvshift, Fshift[0::2], maxiter=2000,
        )
        U = W + G
        indicator = compute_p2_residual_indicator(
            nodes, U, action, F, regular,
        )
    return U, indicator, F, action


for epsilon in (0.0, 1.0):
    for degree in (1, 2):
        for n in (20, 40, 80):
            nodes = np.linspace(0.0, 1.0, n+1)
            r = solve_on_mesh(nodes, degree, epsilon=epsilon)
            Uref, iref, F, action = reference(nodes, degree, epsilon)
            u_diff = float(np.max(np.abs(r['U']-Uref)))
            eta = float(np.linalg.norm(r['indicator']['eta']))
            eta_ref = float(np.linalg.norm(iref['eta']))
            assert u_diff < 1e-9, (epsilon, degree, n, 'solution', u_diff)
            assert abs(eta-eta_ref) < 1e-8, (epsilon, degree, n, 'eta', eta-eta_ref)
            direct_dual = action(r['U'])-F
            dual_diff = float(np.max(np.abs(direct_dual-r['Lambda'])))
            assert dual_diff < 1e-9, (epsilon, degree, n, 'dual', dual_diff)
            Maction = p1_mass_action if degree==1 else p2_mass_action
            mass_error = float(np.max(np.abs(
                Maction(nodes, r['indicator']['mu']) - r['Lambda']
            )))
            assert mass_error < 1e-10, (epsilon, degree, n, 'mass', mass_error)
            info = r['info']
            assert info['min_gap_physical'] >= -1e-9
            assert info['min_multiplier_physical'] >= -1e-9
            assert info['max_complementarity_physical'] <= 1e-9
            assert np.all(np.isfinite(compute_errors(r)))
            print(f'epsilon={epsilon:g} P{degree} n={n}: '
                  f'|dU|={u_diff:.2e} |deta|={abs(eta-eta_ref):.2e} '
                  f'|ddual|={dual_diff:.2e} |Mmu-Lambda|={mass_error:.2e} PASS')

for n in (20,40,80):
    nodes = np.linspace(0,1,n+1)
    G=sine(create_p2_nodes(nodes)[1:-1],1.0)
    diff = np.max(np.abs(_sine_p2_stiffness_action(nodes,1.0)-p2_matvec(nodes,G)))
    assert diff < 1e-10, ('sine action',n,diff)
    print(f'P2 stable sine stiffness n={n}: difference={diff:.2e} PASS')

print('All sine gap integration regression checks PASSED.')
