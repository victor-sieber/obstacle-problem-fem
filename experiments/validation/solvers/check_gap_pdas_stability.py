"""Regression and strongly graded-mesh checks for the optional gap PDAS."""
import numpy as np
from src.p1_nested_large import (
    assemble_p1_dirac, matvec, solve_pdas_p1, solve_pdas_p1_gap,
    prolongate_p1,
)
from src.p2_nested_large import (
    assemble_p2_dirac, join_p2, p2_matvec,
    solve_pdas_p2, solve_pdas_p2_gap, prolongate_p2,
)
from src.multiplier_l2_transfer import (
    transfer_l2_multiplier_p1, transfer_l2_multiplier_p2,
)
from src.fem import assemble_load_vector as assemble_p1_load
from src.fem_p2 import assemble_load_vector as assemble_p2_load

for n in (20, 60, 200):
    nodes = np.linspace(0, 1, n + 1)
    lower, diagonal, upper, F = assemble_p1_dirac(nodes)
    old, _ = solve_pdas_p1(nodes, lower, diagonal, upper, F, maxiter=1000)
    new, info = solve_pdas_p1_gap(nodes, lower, diagonal, upper, F, maxiter=1000)
    diff = np.max(np.abs(old - new))
    assert diff < 1e-9, ("P1", n, diff)
    assert info["min_gap"] >= -1e-8
    assert info["min_multiplier"] >= -1e-8
    print(f"P1 Dirac n={n}: max solution difference={diff:.3e}, PASS")

    Fv, Fm = assemble_p2_dirac(nodes)
    old, _ = solve_pdas_p2(nodes, Fv, Fm, maxiter=1000)
    new, info = solve_pdas_p2_gap(nodes, Fv, Fm, maxiter=1000)
    diff = np.max(np.abs(old - new))
    assert diff < 1e-9, ("P2", n, diff)
    assert info["min_gap"] >= -1e-8
    assert info["min_multiplier"] >= -1e-8
    print(f"P2 Dirac n={n}: max solution difference={diff:.3e}, PASS")

gamma = 1 - np.sqrt(2) / 3
forcing = lambda x: np.pi**2 * np.sin(np.pi * x)

for degree in (1, 2):
    nodes = np.linspace(0, 1, 21)
    previous = None
    for level in range(25):
        M = len(nodes)-1
        if degree == 1:
            lower, diagonal, upper, F0 = assemble_p1_dirac(nodes)
            F = F0 + assemble_p1_load(nodes, forcing, quadrature_order=8)
            g = np.sin(np.pi * nodes[1:-1])
            Fshift = F - matvec(lower, diagonal, upper, g)
            args = (lower, diagonal, upper, Fshift)
            solve = solve_pdas_p1_gap
            prolong = prolongate_p1
            transfer = transfer_l2_multiplier_p1
        else:
            Fv, Fm = assemble_p2_dirac(nodes)
            F = join_p2(Fv, Fm) + assemble_p2_load(
                nodes, forcing, quadrature_order=8
            )
            x = np.empty(2*M-1)
            x[0::2] = (nodes[:-1] + nodes[1:]) / 2
            x[1::2] = nodes[1:-1]
            g = np.sin(np.pi * x)
            Fshift = F - p2_matvec(nodes, g)
            Fvshift = np.zeros(M+1)
            Fvshift[1:-1] = Fshift[1::2]
            args = (Fvshift, Fshift[0::2])
            solve = solve_pdas_p2_gap
            prolong = prolongate_p2
            transfer = transfer_l2_multiplier_p2

        if previous is None:
            warm = {}
        else:
            oldnodes, oldU, olddual = previous
            warm = dict(
                initial_solution=prolong(oldnodes, oldU, nodes),
                initial_multiplier=transfer(oldnodes, olddual, nodes),
            )
        U, info = solve(nodes, *args, maxiter=100, **warm)
        assert info["min_gap"] >= -1e-8
        assert info["min_multiplier"] >= -1e-8
        assert info["max_complementarity"] <= 1e-8
        previous = (nodes, U, info["dual_coefficients"])
        if level < 24:
            idx = np.searchsorted(nodes, gamma) - 1
            nodes = np.insert(nodes, idx+1, (nodes[idx] + nodes[idx+1])/2)
    print(
        f"P{degree} locally refined near gamma: "
        f"24 refinements, min_h={np.min(np.diff(nodes)):.3e}, "
        f"last PDAS iterations={info['iterations']}, PASS"
    )

print("All regression and graded-mesh tests PASSED.")
