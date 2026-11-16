"""Execute unchanged object evaluators with reviewed U001 accounting functions."""
import runpy
import sys
from common import parents, qualified, research

if __name__ == '__main__':
    qualified()
    name = sys.argv.pop(1)
    assert name in ['go2nav.eval.balls_binding', 'go2nav.eval.balls_probes', 'go2nav.eval.balls_planning']
    unlimited = parents()[2]
    research.gpu_run = unlimited.accounting_run
    research.budget_remaining = unlimited.accounting_remaining
    runpy.run_module(name, run_name='__main__', alter_sys=True)
