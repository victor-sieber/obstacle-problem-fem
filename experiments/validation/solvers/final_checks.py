"""Small independent matrix checks and actual stopping-rule integration checks."""
import contextlib
import io
import tempfile
from functools import partial
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from src import fem, fem_p2
from src.p1_nested_large import thomas, assemble_p1_dirac
from src.p2_nested_large import solve_gap_p2_on_active_set, split_p2, _p2_gap_load
from src.multiplier_l2_transfer import p1_riesz_coefficients, p2_riesz_coefficients
from src.pdas_indicator import PDASIndicator, passes_nodal_kkt
from experiments.key.final_study import solve as solve_case, reference
from experiments.key.sine_dirac.convergence import run_series


def main():
    rng = np.random.default_rng(20261115)
    # Independent dense assembly is deliberately restricted to small systems.
    for p, module in ((1,fem),(2,fem_p2)):
        for case in ('sine_dirac','no_exact','split20','split40','split80'):
            for n in (20,47):
                nodes=np.linspace(0,1,n+1)**1.2
                result=solve_case(case,nodes,p,2000,None,None,measure=False)
                A=module.assemble_stiffness_matrix(nodes)
                M=module.assemble_mass_matrix(nodes)
                u, lam=result['U'],result['Lambda']
                assert np.max(abs(A@u-result['F']-lam)) < 1e-10
                mu=(p1_riesz_coefficients if p==1 else p2_riesz_coefficients)(nodes,lam)
                assert np.max(abs(M@mu-lam)) < 1e-11
                assert result['info']['min_gap_physical'] >= -1e-10
                if p==2:
                    fv,fm=split_p2(result.get('F_shift',result['F']),n)
                    fv,fm=_p2_gap_load(nodes,fv,fm)
                    rhs=np.empty(2*n-1);rhs[0::2]=fm;rhs[1::2]=fv[1:-1]
                    for trial in range(4):
                        active=rng.random(2*n-1)<.4
                        fast=solve_gap_p2_on_active_set(nodes,fv,fm,active)
                        expected=np.zeros_like(fast)
                        free=~active
                        expected[free]=solve(A[np.ix_(free,free)],rhs[free],assume_a='pos')
                        assert np.max(abs(fast-expected)) < 1e-10
    for n in (2,9,31):
        nodes=np.linspace(0,1,n+2)
        lo,d,up,_=assemble_p1_dirac(nodes)
        rhs=rng.normal(size=n)
        A=np.diag(d)+np.diag(lo[1:],-1)+np.diag(up[:-1],1)
        assert np.allclose(thomas(lo,d,up,rhs),solve(A,rhs),atol=1e-12)
    # Real refinement runs: target must stop refinement before the level cap.
    with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
        kw=dict(degree=2,method='adaptive',solver=partial(solve_case,'no_exact'),
                benchmark='check',max_levels=4,max_dofs=2000,output_dir=Path(temp))
        budget=run_series(**kw)
        stopped=run_series(**kw,mesh_rel_tol=.5)
        assert len(stopped)<len(budget)
        assert stopped[-1]['eta_ratio']<=.5
        assert stopped[-1]['stop_reason']=='relative_estimator'
        assert stopped[0]['eta_ratio']==1
        for a,b in zip(stopped,budget):
            assert a['dofs']==b['dofs'] and a['eta']==b['eta']
    # Inner denominator belongs to this monitor instance, and zero is safe.
    evaluate=lambda gap,dual: {'eta':np.asarray(gap)}
    first=PDASIndicator(evaluate,.1)
    assert not first(np.array([10.]),None)
    assert first(np.array([.5]),None)
    second=PDASIndicator(evaluate,.1)
    assert not second(np.array([.5]),None)
    zero=PDASIndicator(evaluate,.1)
    assert not zero(np.array([0.]),None) and zero.ratio is None
    assert not passes_nodal_kkt(np.array([-1.]),np.array([0.]),np.array([False]))
    # A permissive target exercises the real relative-exit branch.
    nodes=np.linspace(0,1,41)
    normal=solve_case('no_exact',nodes,1,2000,None,None)
    trial=solve_case('no_exact',nodes,1,2000,None,None,inner=.99)
    assert trial['info']['pdas_stop_reason']=='relative_indicator'
    assert np.max(abs(normal['U']-trial['U']))<1e-10
    for case in ('no_exact','split20','split40','split80'):
        exact,breaks=reference(case)
        assert np.allclose(exact(np.array([0.,1.]))[0],0,atol=1e-12)
        assert np.allclose(exact(np.asarray(breaks))[0],-1,atol=1e-12)
        assert np.allclose(exact(np.asarray(breaks))[1],0,atol=1e-12)
    print('PASS: independent assembly, P2 condensation, mass solves, Thomas, outer stopping, inner reset/guard, reference boundaries.')


if __name__=='__main__':
    main()
