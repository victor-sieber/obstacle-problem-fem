"""Independent submission regressions. No canonical CSVs are overwritten.

Checks the nested Galerkin identities, mass transfer, regular-load reference,
large P1 precision activation, and all 1280-element warm-start counts once.
Writes only results/validation/submission/checks.json.
"""
import hashlib
import json
import platform

import numpy as np
import scipy

from src import fem, fem_p2
from src.p1_nested_large import prolongate_p1, assemble_p1_dirac, solve_pdas_p1_gap
from src.p2_nested_large import prolongate_p2
from src.multiplier_l2_transfer import transfer_l2_multiplier_p1, transfer_l2_multiplier_p2
from experiments.key.final_study import ROOT, solve, reference
from experiments.key.inner_study import transferred_start


def main():
    results = {}
    coarse = np.array([0., .09, .31, .6, 1.])
    fine = np.sort(np.r_[coarse, (coarse[:-1]+coarse[1:])/2])
    for p, module, prolong, transfer in (
            (1, fem, prolongate_p1, transfer_l2_multiplier_p1),
            (2, fem_p2, prolongate_p2, transfer_l2_multiplier_p2)):
        n = p*(len(coarse)-1)-1
        P = np.column_stack([prolong(coarse, v, fine) for v in np.eye(n)])
        AH, Ah = module.assemble_stiffness_matrix(coarse), module.assemble_stiffness_matrix(fine)
        MH, Mh = module.assemble_mass_matrix(coarse), module.assemble_mass_matrix(fine)
        assert np.max(abs(AH-P.T@Ah@P)) < 1e-12
        assert np.max(abs(MH-P.T@Mh@P)) < 1e-12
        dual = np.arange(1., n+1)
        computed = transfer(coarse, dual, fine)
        assert np.max(abs(computed-Mh@P@np.linalg.solve(MH, dual))) < 1e-11
        results[f'P{p}_Galerkin_mass_transfer'] = 'PASS'
    exact, breaks = reference('no_exact')
    x = np.linspace(0, 1, 10001)
    u, du = exact(x)
    assert np.min(u) >= -1-1e-14
    assert np.max(abs(exact(np.array(breaks))[1])) < 1e-12
    assert np.max(abs(exact(np.array(breaks))[0]+1)) < 1e-12
    results['regular_contact_boundaries'] = breaks
    # Exact P1 nodal values for -u''=1 on a large uniform mesh. This checks
    # the public PDAS path, including the newly broadened precision trigger.
    nodes = np.linspace(0, 1, 16001)
    lo, d, up, _ = assemble_p1_dirac(nodes)
    F = (np.diff(nodes)[:-1]+np.diff(nodes)[1:])/2
    u, info = solve_pdas_p1_gap(nodes, lo, d, up, F)
    target = nodes[1:-1]*(1-nodes[1:-1])/2
    error = float(np.max(abs(u-target)))
    assert error < 2e-14, error
    results['large_P1_max_nodal_error'] = error
    expected = {('sine_dirac', 1): [234,177,5], ('sine_dirac', 2): [461,355,9],
                ('no_exact', 1): [243,51,5], ('no_exact', 2): [487,3,9]}
    results['warm_starts'] = []
    for (case, degree), counts in expected.items():
        coarse = solve(case, np.linspace(0,1,641), degree, 5000, None, None, inner=1e-3)
        fine = np.linspace(0,1,1281)
        cold = None
        for method, count in zip(('cold', 'primal', 'riesz'), counts):
            primal, dual = transferred_start(coarse, fine, degree, method)
            trial = solve(case, fine, degree, 5000, primal, dual, inner=1e-3)
            if cold is None:
                cold = trial['U']
            difference = float(np.max(abs(trial['U']-cold)))
            assert difference <= 1e-8
            assert trial['info']['iterations'] == count
            results['warm_starts'].append(dict(case=case, degree=degree, method=method,
                                              iterations=count, max_difference=difference))
            print(case, degree, method, count, 'PASS', flush=True)
    results['environment'] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    results['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'src').glob('*.py'))}
    out = ROOT/'results/validation/submission/checks.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2)+'\n')
    print('PASS: submission regressions; evidence:', out)


if __name__ == '__main__':
    main()
