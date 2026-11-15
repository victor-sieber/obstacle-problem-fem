"""Report plots for Grassle meeting: sine-Dirac and no-exact obstacle benchmarks.

Input: sine-Dirac and no-exact CSVs and solution snapshots in results/key/.
This script NEVER solves the PDE or modifies existing CSV data.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
RESULTS = None
FIGURES = None
CASES = [(1, 'uniform'), (1, 'adaptive'), (2, 'uniform'), (2, 'adaptive')]
COLORS = {(1,'uniform'):'#457b9d', (1,'adaptive'):'#1d3557',
          (2,'uniform'):'#e09f3e', (2,'adaptive'):'#9e2a2b'}



def benchmark_file(override, kind, topic, name):
    folder = override if override is not None else BASE / kind / "key" / topic
    return folder / name


def load(benchmark, degree, method):
    path = benchmark_file(RESULTS, "results", benchmark, f"{benchmark}_p{degree}_{method}.csv")
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError(f'Need >=3 refinement rows: {path}')
    return rows


def arr(rows, key):
    return np.array([float(r[key]) for r in rows])


def fit_slope(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    x, y = x[mask][-4:], y[mask][-4:]
    if len(x) < 3:
        return np.nan
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def decorate(ax, ylabel):
    ax.set(xscale='log', yscale='log', xlabel='Interior degrees of freedom', ylabel=ylabel)
    ax.grid(True, which='both', alpha=0.17)
    ax.legend(fontsize=8, loc='best')


def curves(ax, benchmark, key, annotate_slopes=False):
    for degree, method in CASES:
        rows=load(benchmark, degree, method)
        x, y=arr(rows,'dofs'),arr(rows,key)
        slope=fit_slope(x,y)
        label=f'P{degree} {method}'
        if annotate_slopes and np.isfinite(slope):
            label+=f' (slope {slope:+.2f})'
        ax.loglog(x,y, '-o' if method=='adaptive' else '--s',
                  color=COLORS[(degree,method)], label=label,
                  markersize=3.8, linewidth=1.5)


def sine_error_plots():
    fig, (a,b)=plt.subplots(1,2,figsize=(13.3,5),constrained_layout=True)
    curves(a,'sine_dirac','h1_error',True)
    curves(b,'sine_dirac','l2_error',True)
    a.set_title('True $H^1$-seminorm error')
    b.set_title('True $L^2$ error')
    decorate(a, r'$|u-u_h|_{H^1}$')
    decorate(b, r'$\|u-u_h\|_{L^2}$')
    fig.suptitle('Sine-perturbed Dirac benchmark: P1/P2, uniform/adaptive; max marking $\\theta=0.5$')
    dest = benchmark_file(FIGURES, "figures", "sine_dirac", "errors.png")
    fig.savefig(dest,dpi=210)
    plt.close(fig)
    print('Saved:',dest)


def sine_indicator_plots():
    fig,ax=plt.subplots(figsize=(7.5,5.1),constrained_layout=True)
    curves(ax,'sine_dirac','eta',False)
    decorate(ax,r'$\eta_\ell$')
    ax.set_title('Sine-Dirac: residual indicator (not true error)')
    dest = benchmark_file(FIGURES, "figures", "sine_dirac", "estimator.png")
    fig.savefig(dest,dpi=210)
    plt.close(fig)
    print('Saved:',dest)


def eval_on_grid(nodes,U,degree,x):
    if degree==1:
        return np.interp(x,nodes,np.r_[0.,U,0.])
    from src.fem_p2 import evaluate_fem_solution
    return evaluate_fem_solution(nodes,U,x)


def no_exact_plots():
    fig,(a,b)=plt.subplots(1,2,figsize=(13.5,5),constrained_layout=True)
    curves(a,'no_exact','eta',True)
    a.set_title('Residual indicator (no exact solution)')
    decorate(a,r'$\eta_\ell$')
    xx=np.linspace(0.,1.,1000)
    for degree,color in ((1,'#1d3557'),(2,'#9e2a2b')):
        p = benchmark_file(RESULTS, "results", "no_exact", f"no_exact_p{degree}_adaptive_final.npz")
        with np.load(p) as dat:
            nodes=np.array(dat['nodes'])
            U=np.array(dat['U'])
        yy=eval_on_grid(nodes,U,degree,xx)
        b.plot(xx,yy,color=color,label=f'P{degree} adaptive',linewidth=1.7)
        if degree==1:
            x_dof=nodes[1:-1]
        else:
            x_dof=np.empty(len(U));x_dof[0::2]=(nodes[:-1]+nodes[1:])/2; x_dof[1::2]=nodes[1:-1]
        contact=np.abs(U+1)<=1e-8
        inds=np.flatnonzero(contact)
        if len(inds)>100:
            inds=inds[np.linspace(0,len(inds)-1,100).astype(int)]
        b.scatter(x_dof[inds],U[inds],s=8,color=color,alpha=.65)
        print(f'P{degree} contact nodes: {int(contact.sum())}/{len(U)}')
    b.axhline(-1.,color='black',ls='--',lw=1.1,label=r'obstacle $\psi=-1$')
    b.set(xlabel='x',ylabel=r'$u_h(x)$',title='Solutions and contact DOFs (pointwise feasibility not certified)')
    b.grid(alpha=.2)
    b.legend(fontsize=8)
    fig.suptitle(r'Non-manufactured problem: $f=-12+\sin(\pi x)$, no Dirac loads')
    dest = benchmark_file(FIGURES, "figures", "no_exact", "contact.png")
    fig.savefig(dest,dpi=210)
    plt.close(fig)
    print('Saved:',dest)


def no_exact_nested_plot():
    fig,ax=plt.subplots(figsize=(7.5,5.1),constrained_layout=True)
    for degree,method in CASES:
        rows=load('no_exact',degree,method)
        xx=np.array([float(r['dofs']) for r in rows[1:]])
        yy=np.array([float(r['delta_h1_previous']) for r in rows[1:]])
        good=(yy>0)&np.isfinite(yy)
        ax.loglog(xx[good],yy[good], '-o' if method=='adaptive' else '--s',
                  color=COLORS[(degree,method)],markersize=3.8,
                  label=f'P{degree} {method}')
    ax.set_title('Successive nested-mesh solution differences')
    decorate(ax,r'$|u_{h_k}-u_{h_{k-1}}|_{H^1}$ (not true error)')
    dest = benchmark_file(FIGURES, "figures", "no_exact", "mesh_differences.png")
    fig.savefig(dest,dpi=210)
    plt.close(fig)
    print('Saved:',dest)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    RESULTS, FIGURES = args.results_dir, args.figures_dir
    if FIGURES is not None:
        FIGURES.mkdir(parents=True, exist_ok=True)
    sine_error_plots()
    sine_indicator_plots()
    no_exact_plots()
    no_exact_nested_plot()
