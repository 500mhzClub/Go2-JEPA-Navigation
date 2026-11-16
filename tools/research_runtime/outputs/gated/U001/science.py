"""Execute the original evaluator with explicitly unlimited time accounting."""
import runpy
import sys
from common import accounting_remaining, accounting_run, research

if __name__ == '__main__':
    name = sys.argv.pop(1)
    assert name == 'go2nav.eval.balls_planning'
    research.gpu_run = accounting_run
    research.budget_remaining = accounting_remaining
    runpy.run_module(name, run_name='__main__', alter_sys=True)
