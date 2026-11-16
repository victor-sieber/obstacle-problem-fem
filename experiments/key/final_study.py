"""Reproducible final study. Run from the repository root with python -m.

Writes final-study topic folders and results/validation/stopping.
All methods use the existing indicator and maximum marking theta=0.5.
"""
import argparse
import csv
import json
import hashlib
import platform
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import brentq

from experiments.key.sine_dirac.convergence import run_series
from experiments.key.sine_dirac.problem import solve_on_mesh as sine_solve, compute_errors
from experiments.key.no_exact.convergence import solve_on_mesh as load_solve
from src.p1_nested_large import prolongate_p1, assemble_p1_dirac, _p1_gap_load, matvec
from src.p2_nested_large import prolongate_p2, join_p2, split_p2, _p2_gap_load
from src.multiplier_l2_transfer import transfer_l2_multiplier_p1, transfer_l2_multiplier_p2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/key/final_study'
FIG = ROOT / 'figures/key/final_study'
VALID = ROOT / 'results/validation/stopping'
CASES = ('sine_dirac', 'no_exact', 'split20', 'split40', 'split80')
COLORS = ('#457b9d', '#1d3557', '#e09f3e', '#9e2a2b')
PAIRS = ((1, 'uniform'), (1, 'adaptive'), (2, 'uniform'), (2, 'adaptive'))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def element_gap_minimum(g, p):
    """Exact minimum of the FE gap relative to the nodal obstacle interpolant."""
    full = np.r_[1., g, 1.]
    if p == 1:
        return float(full.min())
    left, mid, right = full[:-2:2], full[1::2], full[2::2]
    a = 2 * (left - 2 * mid + right)
    b = -3 * left + 4 * mid - right
    t = np.divide(-b, 2 * a, out=np.full_like(a, -1.), where=a > 0)
    inside = (t > 0) & (t < 1)
    values = a[inside] * t[inside]**2 + b[inside] * t[inside] + left[inside]
    return float(min(full.min(), values.min(initial=np.inf)))


def check_result(r):
    """Nodal KKT plus row-scaled stationarity, evaluated in gap coordinates."""
    x, p, g, lam = r['nodes'], r['degree'], r['gap_coefficients'], r['Lambda']
    f = r.get('F_shift', r['F'])
    h = np.diff(x)
    if p == 1:
        lo, d, up, _ = assemble_p1_dirac(x)
        rhs = _p1_gap_load(lo, d, up, f)
        scale = 1 + abs(rhs) + matvec(abs(lo), d, abs(up), abs(g))
    else:
        fv, fm = split_p2(f, len(h))
        fv, fm = _p2_gap_load(x, fv, fm)
        rhs = join_p2(fv, fm)
        v, m = split_p2(abs(g), len(h))
        sv = np.zeros(len(x))
        sv[:-1] += (7*v[:-1] + 8*m + v[1:])/(3*h)
        sv[1:] += (v[:-1] + 8*m + 7*v[1:])/(3*h)
        sm = (8*v[:-1] + 16*m + 8*v[1:])/(3*h)
        scale = 1 + abs(rhs) + join_p2(sv, sm)
    active = g == 0
    free = ~active
    scaled = float(np.max(abs(lam[free])/scale[free], initial=0))
    assert np.min(g) >= -1e-9, 'Primal KKT failure'
    assert np.min(lam[active], initial=0) >= -1e-8, 'Dual KKT failure'
    assert scaled <= 1e-12, 'Scaled free equation failure'
    assert np.max(abs(g*lam)/scale, initial=0) <= 1e-12, 'Complementarity failure'
    r['info']['scaled_free_residual'] = scaled
    r['info']['min_element_gap'] = element_gap_minimum(g, p)
    return r


def solve(case, nodes, degree, maxiter, initial_solution, initial_multiplier,
          *, inner=None, measure=True):
    kw = dict(maxiter=maxiter, initial_solution=initial_solution,
              initial_multiplier=initial_multiplier, pdas_rel_tol=inner,
              measure_pdas=measure)
    if case == 'sine_dirac':
        r = sine_solve(nodes, degree, **kw)
    elif case == 'no_exact':
        r = load_solve(nodes, degree, **kw)
    else:
        amplitude = int(case.removeprefix('split'))
        load = lambda x: -12 + amplitude*np.cos(4*np.pi*np.asarray(x))
        r = load_solve(nodes, degree, load=load, **kw)
    return check_result(r)


def run_study(inner=None, max_dofs=30000):
    folder = OUT if inner is None else VALID
    for case in CASES:
        for p, method in PAIRS:
            rows = run_series(p, method, partial(solve, case, inner=inner),
                error_function=compute_errors if case == 'sine_dirac' else partial(reference_errors, case),
                benchmark=case, max_levels=45, max_dofs=max_dofs, maxiter=2000,
                output_dir=folder/case, mesh_rel_tol=1e-3)
            assert rows[-1]['stop_reason'] in ('relative_estimator', 'dof_cap', 'max_levels')
    meta = dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, mesh_rel_tol=1e-3, pdas_rel_tol=inner,
                max_dofs=max_dofs, max_levels=45, initial_elements=20,
                marking='maximum', theta=0.5, cases=list(CASES),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(list((ROOT/'src').glob('*.py')) +
                                    list((ROOT/'experiments/key').rglob('*.py')))})
    (folder/'configuration.json').write_text(json.dumps(meta, indent=2)+'\n')


def warm_starts():
    """Same coarse solution and fine mesh; times include transfer/assembly/indicator.

    The common coarse solve is excluded. Three repetitions, no per-iterate monitor.
    """
    rows = []
    for case in ('sine_dirac', 'split40'):
        for p in (1, 2):
            prolong = prolongate_p1 if p == 1 else prolongate_p2
            transfer = transfer_l2_multiplier_p1 if p == 1 else transfer_l2_multiplier_p2
            for n in (80, 320, 1280):
                coarse = solve(case, np.linspace(0, 1, n//2+1), p, 5000, None, None, measure=False)
                x = np.linspace(0, 1, n+1)
                reference = None
                for method in ('cold', 'primal', 'riesz'):
                    times = []
                    for repeat in range(3):
                        start = perf_counter()
                        u0 = None if method == 'cold' else prolong(coarse['nodes'], coarse['W'], x)
                        l0 = transfer(coarse['nodes'], coarse['Lambda'], x) if method == 'riesz' else None
                        r = solve(case, x, p, 5000, u0, l0, measure=False)
                        times.append(perf_counter()-start)
                    if reference is None:
                        reference = r['U']
                    difference = float(np.max(abs(reference-r['U'])))
                    assert difference <= 1e-8, 'Warm starts disagree on final solution'
                    rows.append(dict(case=case, degree=p, elements=n, dofs=p*n-1,
                        method=method, iterations=r['info']['iterations'],
                        median_seconds=float(np.median(times)), max_solution_difference=difference))
                    print('warm:', rows[-1], flush=True)
                    write_rows(OUT/'warm_starts.csv', rows)


def inner_audit():
    """Compare enabled/disabled inner stopping on the same complete mesh histories."""
    rows = []
    for case in CASES:
        for p, method in PAIRS:
            name = f'{case}_p{p}_{method}.csv'
            base, trial = read_rows(OUT/case/name), read_rows(VALID/case/name)
            assert len(base) == len(trial)
            saved = stops = rejected = 0
            for a, b in zip(base, trial):
                assert a['dofs'] == b['dofs']
                assert np.isclose(float(a['eta']), float(b['eta']), rtol=1e-9, atol=1e-12)
                saved += int(a['iterations']) - int(b['iterations'])
                stops += b['pdas_stop_reason'] == 'relative_indicator'
                rejected += len(json.loads(b['pdas_rejected_indicator_steps']))
            rows.append(dict(case=case, degree=p, method=method, meshes=len(base),
                             inner_relative_stops=stops, iterations_saved=saved,
                             rejected_candidates=rejected))
    write_rows(VALID/'inner_audit.csv', rows)
    print('INNER AUDIT:', {key:sum(int(r[key]) for r in rows) for key in
          ('meshes','inner_relative_stops','iterations_saved','rejected_candidates')})


def reference(case):
    """Independent solution derived AFTER prescribing load and obstacle.

    Return value/derivative and free boundaries. Never used for refinement.
    """
    if case == 'no_exact':
        a = brentq(lambda x: -6*x*x + (np.sin(np.pi*x)-np.pi*x*np.cos(np.pi*x))/np.pi**2+1, .2, .49)
        c = -12*a - np.cos(np.pi*a)/np.pi
        def exact(x):
            t = np.minimum(x, 1-x)
            u = np.where(t<a, 6*t*t+np.sin(np.pi*t)/np.pi**2+c*t, -1.)
            du = np.where(t<a, 12*t+np.cos(np.pi*t)/np.pi+c, 0.)
            return u, np.where(x<=.5, du, -du)
        return exact, [a, 1-a]
    amplitude = int(case.removeprefix('split'))
    w = 4*np.pi
    a = brentq(lambda x: -6*x*x+amplitude*(np.cos(w*x)-1)/w**2+amplitude*x*np.sin(w*x)/w+1, .1, .34)
    b = brentq(lambda x: 12*(x-.5)-amplitude*np.sin(w*x)/w, .26, .499)
    assert a < b < .5
    c = -12*a + amplitude*np.sin(w*a)/w
    def exact(x):
        t = np.minimum(x, 1-x)
        left = 6*t*t+amplitude*(np.cos(w*t)-1)/w**2+c*t
        center = -1+6*((t-.5)**2-(b-.5)**2)+amplitude*(np.cos(w*t)-np.cos(w*b))/w**2
        u = np.where(t<a, left, np.where(t>b, center, -1.))
        du = np.where(t<a, 12*t-amplitude*np.sin(w*t)/w+c,
                      np.where(t>b, 12*(t-.5)-amplitude*np.sin(w*t)/w, 0.))
        return u, np.where(x<=.5, du, -du)
    # Validate the assumed contact topology independently of FEM output.
    xx = np.linspace(0, 1, 20001)
    u, _ = exact(xx)
    contact = ((xx>=a)&(xx<=b)) | ((xx>=1-b)&(xx<=1-a))
    assert u.min() >= -1-1e-12
    assert np.min(12-amplitude*np.cos(w*xx[contact])) >= 0
    return exact, [a,b,1-b,1-a]


def reference_errors(case, result):
    from numpy.polynomial.legendre import leggauss
    from src import fem, fem_p2
    q, weights = leggauss(12)
    exact, breaks = reference(case)
    nodes, u, p = result['nodes'], result['U'], result['degree']
    module = fem if p == 1 else fem_p2
    cuts = np.sort(np.unique(np.r_[nodes, breaks]))
    xx = (cuts[:-1,None]+cuts[1:,None])/2 + np.diff(cuts)[:,None]*q/2
    ww = np.diff(cuts)[:,None]*weights/2
    true, deriv = exact(xx.ravel())
    value = module.evaluate_fem_solution(nodes, u, xx.ravel())
    slope = module.evaluate_fem_derivative(nodes, u, xx.ravel())
    return (float(np.sqrt(np.sum(ww.ravel()*(true-value)**2))),
            float(np.sqrt(np.sum(ww.ravel()*(deriv-slope)**2))))


def reference_checks():
    rows = []
    for case in CASES[1:]:
        exact, breaks = reference(case)
        for p in (1, 2):
            data = np.load(OUT/case/f'{case}_p{p}_adaptive_final.npz')
            l2,h1 = reference_errors(case,dict(nodes=data['nodes'],U=data['U'],degree=p))
            rows.append(dict(case=case, degree=p, dofs=len(data['U']),
                l2_error=l2, h1_error=h1, free_boundaries=json.dumps(breaks)))
    write_rows(OUT/'independent_reference.csv', rows)


def scaling():
    rows = []
    for p in (1, 2):
        prev = None
        for n in (20*2**j for j in range(13)):
            x = np.linspace(0, 1, n+1)
            start = perf_counter()
            u0 = l0 = None
            if prev is not None:
                u0 = (prolongate_p1 if p == 1 else prolongate_p2)(prev['nodes'],prev['W'],x)
                l0 = (transfer_l2_multiplier_p1 if p == 1 else transfer_l2_multiplier_p2)(prev['nodes'],prev['Lambda'],x)
            r = solve('sine_dirac', x, p, 2000, u0, l0, measure=False)
            seconds = perf_counter()-start
            rows.append(dict(degree=p, dofs=p*n-1, iterations=r['info']['iterations'],
                seconds=seconds, eta=float(np.linalg.norm(r['indicator']['eta'])),
                scaled_free_residual=r['info']['scaled_free_residual']))
            prev = r
            write_rows(OUT/'scaling.csv',rows)
            print('scaling:', rows[-1], flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':11, 'axes.grid':True, 'grid.alpha':.2})
    def curves(ax, case, key):
        for (p,m), color in zip(PAIRS,COLORS):
            r = read_rows(OUT/case/f'{case}_p{p}_{m}.csv')
            ax.loglog([int(t['dofs']) for t in r], [float(t[key]) for t in r],
                      '-o' if m=='adaptive' else '--s', ms=3, color=color,label=f'P{p} {m}')
        ax.set_xlabel('Interior degrees of freedom')
        ax.legend(fontsize=9)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,key,title in zip(axes,('h1_error','l2_error'),('True H1 seminorm error','True L2 error')):
        curves(ax,'sine_dirac',key); ax.set_title(title)
    fig.savefig(FIG/'sine_errors.png',dpi=190); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,case in zip(axes,('sine_dirac','no_exact')):
        curves(ax,case,'eta_ratio'); ax.axhline(1e-3,color='black',ls=':')
        ax.set_title(case.replace('_',' ')); ax.set_ylabel(r'$\eta_\ell/\eta_0$')
    fig.savefig(FIG/'relative_mesh.png',dpi=190); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    xx=np.linspace(0,1,2001)
    from src.fem_p2 import evaluate_fem_solution
    for case in ('split20','split40','split80'):
        data=np.load(OUT/case/f'{case}_p2_adaptive_final.npz')
        yy=evaluate_fem_solution(data['nodes'],data['U'],xx)
        axes[0].plot(xx,yy,label='A='+case[5:])
    axes[0].axhline(-1,color='black',ls='--',label='obstacle')
    axes[0].set(xlabel='x',ylabel='u_h(x)',title='Two separated contact intervals');axes[0].legend()
    curves(axes[1],'split40','eta_ratio');axes[1].axhline(1e-3,color='black',ls=':')
    axes[1].set(title='A=40: indicator reduction',ylabel=r'$\eta_\ell/\eta_0$')
    fig.savefig(FIG/'split_contact.png',dpi=190);plt.close(fig)
    if (OUT/'warm_starts.csv').exists():
        rows=read_rows(OUT/'warm_starts.csv')
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for i,case in enumerate(('sine_dirac','split40')):
            for j,p in enumerate((1,2)):
                ax=axes[i,j]
                for method in ('cold','primal','riesz'):
                    data=[r for r in rows if r['case']==case and int(r['degree'])==p and r['method']==method]
                    ax.loglog([int(r['dofs']) for r in data],[int(r['iterations']) for r in data],'-o',label=method)
                ax.set(title=f'{case}: P{p}',xlabel='Interior degrees of freedom',ylabel='PDAS iterations');ax.legend()
        fig.savefig(FIG/'warm_starts.png',dpi=190);plt.close(fig)


def tables():
    """Write report/slide tables from the actual local CSVs, not copied numbers."""
    def history(case,p,m):
        return read_rows(OUT/case/f'{case}_p{p}_{m}.csv')
    def scientific(value):
        a,b=f'{float(value):.2e}'.split('e')
        return f'${a}\\times10^{{{int(b)}}}$'
    def command(name,headers,rows):
        cols='l'*len(headers)
        body=['\\newcommand{\\'+name+'}{%', '\\begin{tabular}{'+cols+'}\\toprule',
              ' & '.join(headers)+r' \\ \midrule']
        body += [' & '.join(map(str,row))+r' \\' for row in rows]
        body += [r'\bottomrule\end{tabular}}']
        return '\n'.join(body)
    labels={'sine_dirac':'Sine--Dirac','no_exact':'Single contact','split40':'Two contacts'}
    mesh=[]; accuracy=[]
    for case,label in labels.items():
        for p,m in PAIRS:
            r=history(case,p,m);last=r[-1]
            mesh.append([label,f'$P_{p}$',m,last['dofs'],scientific(last['eta_ratio']),
                         'target' if last['stop_reason']=='relative_estimator' else 'DOF cap'])
        accuracy.append([label]+[next((r['dofs'] for r in history(case,p,m)
                          if float(r['h1_error'])<=1e-3),'--') for p,m in PAIRS])
    warm=[]
    rows=read_rows(OUT/'warm_starts.csv')
    for case in ('sine_dirac','split40'):
        for p in (1,2):
            data=[r for r in rows if r['case']==case and int(r['degree'])==p and r['elements']=='1280']
            warm.append([labels[case],f'$P_{p}$',str(p*1280-1)]+[r['iterations'] for r in data])
    sweep=[]
    for case in ('split20','split40','split80'):
        r1=history(case,1,'adaptive')[-1]; r2=history(case,2,'adaptive')[-1]
        sweep.append([case[5:],r1['dofs'],r2['dofs'],scientific(r1['h1_error']),scientific(r2['h1_error'])])
    chunks=[command('MeshTable',['Problem','FEM','Mesh','$N$','$\\eta_\\ell/\\eta_0$','Stop'],mesh),
            command('AccuracyTable',['Problem','$P_1$ uniform','$P_1$ adaptive','$P_2$ uniform','$P_2$ adaptive'],accuracy),
            command('WarmTable',['Problem','FEM','$N$','Cold','Primal','Riesz'],warm),
            command('SweepTable',['Amplitude','$N$ ($P_1$)','$N$ ($P_2$)','$H^1$ ($P_1$)','$H^1$ ($P_2$)'],sweep)]
    audit=read_rows(VALID/'inner_audit.csv')
    chunks += [f'\\newcommand{{\\InnerMeshes}}{{{sum(int(r["meshes"]) for r in audit)}}}',
               f'\\newcommand{{\\InnerStops}}{{{sum(int(r["inner_relative_stops"]) for r in audit)}}}',
               f'\\newcommand{{\\InnerSaved}}{{{sum(int(r["iterations_saved"]) for r in audit)}}}']
    (OUT/'tables.tex').write_text('\n\n'.join(chunks)+'\n')
    summary=[]
    for case in CASES:
        for p,m in PAIRS:
            r=history(case,p,m)[-1]
            summary.append({k:r[k] for k in ('benchmark','degree','method','dofs','eta_ratio','h1_error',
                                             'stop_reason','min_element_gap','scaled_free_residual')})
    write_rows(OUT/'summary.csv',summary)
    print('Saved tables.tex and summary.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','inner','warm','reference','scaling','plot','tables'))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='run': run_study()
    elif args.action=='inner': run_study(inner=1e-3); inner_audit()
    elif args.action=='warm': warm_starts()
    elif args.action=='reference': reference_checks()
    elif args.action=='scaling': scaling()
    elif args.action=='tables': tables()
    else: plot()


if __name__=='__main__':
    main()
