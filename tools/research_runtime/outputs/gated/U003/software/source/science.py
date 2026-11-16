"""Run original offline P26 modules with reviewed U001 accounting functions."""
import runpy
import sys
from common import parents
from go2nav import research

if __name__ == '__main__':
    name = sys.argv.pop(1)
    assert name in ['go2nav.data.slot_tracks','go2nav.training.object_jepa','go2nav.eval.slot_dynamics']
    _, _, unlimited, _ = parents()
    research.gpu_run = unlimited.accounting_run
    research.budget_remaining = unlimited.accounting_remaining
    runpy.run_module(name, run_name='__main__', alter_sys=True)
