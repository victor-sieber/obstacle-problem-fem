"""Check no-exact experiment CSVs and saved solution snapshots before plotting."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path,
                    default=Path(__file__).resolve().parents[3] / "results" / "key" / "no_exact")
root = parser.parse_args().results_dir
for degree in (1,2):
    for method in ('uniform','adaptive'):
        path=root/f'no_exact_p{degree}_{method}.csv'
        with path.open(newline='') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)>=3, path
        dofs=np.array([int(r['dofs']) for r in rows])
        eta=np.array([float(r['eta_ratio']) for r in rows])
        assert np.all(np.diff(dofs)>0),f'Nonincreasing DOFs {path}'
        assert np.all(np.isfinite(eta)) and np.all(eta>=0),f'Bad eta {path}'
        assert abs(eta[0]-1.)<1e-12
        assert all(r['h1_error']=='' and r['l2_error']=='' for r in rows),f'Accidental true error {path}'
        for r in rows:
            assert math.isfinite(float(r['min_gap']))
            assert float(r['min_gap'])>=-1e-8
            assert float(r['min_active_multiplier'])>=-1e-8
            assert float(r['max_complementarity'])<=1e-8
            assert float(r['free_equation_residual'])<=1e-8
        if "pdas_eta_history" in rows[0]:
            for row in rows:
                history = np.asarray(json.loads(row["pdas_eta_history"]))
                assert len(history) == int(row["iterations"]) + 1
                assert np.all(np.isfinite(history)) and np.all(history >= 0)
                assert np.isclose(history[-1], float(row["eta"]), rtol=1e-12, atol=0.0)
                if history[0] > 0:
                    assert np.isclose(float(row["pdas_eta_ratio"]), history[-1] / history[0])
                else:
                    assert row["pdas_eta_ratio"] == ""
                assert row["pdas_stop_reason"] in {"relative_indicator", "active_set_stable"}
                if row["pdas_stop_reason"] == "relative_indicator":
                    tolerance = float(row["pdas_rel_tol"])
                    assert history[0] > 0 and history[-1] <= tolerance * history[0]
                if row["pdas_stop_reason"] == "active_set_stable":
                    assert row["active_set_stable"] == "True"
        last=rows[-1]
        if method=='adaptive':
            if 'pdas_eta_history' not in last:
                assert (eta[-1]<=1e-3)==(last['stop_reason']=='relative_estimator'),f'Stopping mismatch {path}'
            with np.load(root/f'no_exact_p{degree}_{method}_final.npz') as data:
                nodes=np.array(data['nodes']);U=np.array(data['U'])
            assert len(U)==int(dofs[-1]);assert np.all(np.diff(nodes)>0)
            contact=int(np.count_nonzero(np.abs(U+1.)<=1e-8))
            assert contact>0 and contact<len(U),f'Contact set missing/nontrivial {degree}'
        else:
            contact=None
        print(f'P{degree} {method:8s} DOFs={dofs[-1]:6d}  eta/eta0={eta[-1]:.6e}  '
              f'stop={last["stop_reason"]:20s}  max free res={max(float(r["free_equation_residual"]) for r in rows):.2e}'
              + (f'  contact DOFs={contact}' if contact is not None else ''))
print('ALL FOUR NO-EXACT HISTORIES VALIDATED (nodal KKT and recorded indicator history).')
