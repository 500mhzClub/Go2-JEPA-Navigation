"""Execute original A019 numerics with U001 accounting and null allowance metadata."""
import runpy
import sys
from common import parents,report_remaining
from go2nav import research

if __name__=='__main__':
    name=sys.argv.pop(1)
    assert name=='go2nav.eval.balls_inference'
    _,_,unlimited,_=parents()
    research.gpu_run=unlimited.accounting_run
    research.budget_remaining=report_remaining
    runpy.run_module(name,run_name='__main__',alter_sys=True)
