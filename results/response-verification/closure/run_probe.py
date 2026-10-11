"""Reproduce the bounded standalone CPU development probe, outside sealed DAGs."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy
import scipy
import torch
from tdn.analysis.response.closure import run

OUT = Path(__file__).resolve().parent
WORK = ROOT / '.runtime/response-module-checks/closure-verification'
WORK.mkdir(parents=True, exist_ok=True)
PROTOCOL = {'profile': 'pilot', 'root_seed': 112010001,
    'response': {'closure': {'train_parents': 32, 'validation_parents': 16,
        'test_parents': 16, 'noise_levels': [0., 1e-4, 1e-3], 'rollout_steps': 32,
        'coarse_cutoff': 6, 'shadow_cutoff': 12, 'fine_cutoff': 18, 'step': .01,
        'kappa': .0005, 'reaction': .7, 'step_multipliers': [1., 2.],
        'reaction_rates': [.7, 2.1], 'ridges': [1e-5, .001, .1]}}}


class Budget:
    def __init__(self): self.start = time.monotonic()
    def check(self):
        if time.monotonic()-self.start > 120:
            raise TimeoutError('Standalone protected CPU probe exceeded 120 seconds')


class Context:
    path, protocol, budget = WORK, PROTOCOL, Budget()
    def record(self, *args, **kwargs): pass


if __name__ == '__main__':
    torch.set_num_threads(1)
    details = run(Context())
    source = WORK / 'closure_results.json'
    payload = json.loads(source.read_text())
    hashes = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            file = Path(filename).resolve()
            if file.is_relative_to(ROOT / 'tdn') and file.suffix == '.py':
                hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    summary = dict(schema='tdn.response-standalone-closure-verification/v1',
        verification_scope='Standalone unsealed CPU development module probe; not native Fedora, not fresh confirmation, not an end-to-end DAG',
        actual_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python results/response-verification/closure/run_probe.py',
        protocol=PROTOCOL, source_hashes=hashes,
        software=dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__, torch=torch.__version__, platform=platform.platform()),
        result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), module_results=details,
        source_artifact=str(source.relative_to(ROOT)), cost=payload['cost'], onset=payload['onset'],
        comparisons=payload['comparisons'], selections=payload['selections'], hidden_ambiguity=payload['hidden_ambiguity'],
        independent_test_parents=16, teacher_references=len(payload['teacher_references']),
        unresolved_teacher_references=sum(not r['accepted'] for r in payload['teacher_references']),
        maximum_reference_uncertainty=max(r['uncertainty_maximum'] for r in payload['teacher_references']),
        failed_predictions=sum(not r['prediction_finite'] for r in payload['rows']),
        retained_raw_scope='Concise comparisons and provenance only; original detailed arrays remain in project-local ignored runtime workspace')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    chosen = [r for r in payload['comparisons'] if r['method'] == 'noise_aware_observer']
    table = '\n'.join(f"| {r['noise']:g} | {r['reaction_rate']:g} | {r['h']:g} | {r['rollout_rms_geometric_ratio']:.4f} | {r['eligible_parents']}/{r['attempted_parents']} | {r['decision']} |" for r in chosen)
    (OUT / 'README.md').write_text('# Protected Response closure verification\n\n'
        'This is a **standalone, unsealed CPU development probe**, not a native Fedora campaign or confirmation result. '
        'The observed noise-aware method is a linear ridge startup observer, not a neural network. It assimilates two coarse observations once and then evolves autonomously.\n\n'
        'Run the exact command in `summary.json`; `run_probe.py` retains full intermediate outputs under `.runtime/response-module-checks/closure-verification`. '
        'No historical science artifacts are changed. Source hashes, software, inputs, actual measured costs, reference refinements and all control comparisons are in `summary.json`.\n\n'
        '| Noise RMS | Reaction | Step | Observer / coarse rollout RMS | Eligible parents | Development outcome |\n'
        '|---:|---:|---:|---:|---:|---|\n'+table+'\n\n'
        'Ratios below 1 favor the observer; 0.9 is the declared 10% effect threshold. Each row has 16 independent development parents, '
        'with the same paired noise supplied to all methods. These are descriptive geometric effects, not adjusted confirmatory tests. '
        'The clean observer is the identity by construction. This experiment does not establish a useful noise-aware observer gain. '
        'Reaction-only initially resolved closure has second-order onset; nonzero hidden initialization allows first-order onset. '
        'Reference refinement covers the declared finite Galerkin equation, not a continuum certificate.\n')
    print(json.dumps(details, indent=2))
