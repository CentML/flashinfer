"""SM107 MegaMoE phase-interleave FC1/FC2 scheduler: cluster-uniform state (CPU guard).

The FC1 prologue wait of the vendored ``PhaseInterleavedFc12Scheduler`` must not change scheduler
state from its CTA-local peek of the global FC1 work-ID counter. Every CTA of a cluster runs the
claim state machine on the leader's broadcast work IDs; a CTA-local observation that set
``fc1_exhausted`` let CTAs of one cluster diverge and deadlock the EP group (VENDOR.md, FlashInfer
kernel fix). This guards the fix against a re-export without it; the GPU reproduction is
``test_moe_ep_sm107_phase_interleave_stress_multirank.py``.
"""

import ast
from pathlib import Path

import flashinfer


def test_phase_interleave_prologue_wait_keeps_cluster_uniform_state():
    src = (
        Path(flashinfer.__file__).parent
        / "moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/kernel_src/schedulers/fc12_scheduler.py"
    )
    tree = ast.parse(src.read_text())
    waits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_wait_for_fc1_prologue_claims"
    ]
    assert len(waits) == 1
    assigned = set()
    for node in ast.walk(waits[0]):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            assigned |= {
                name.id for name in ast.walk(target) if isinstance(name, ast.Name)
            }
    assert "fc1_exhausted" not in assigned
