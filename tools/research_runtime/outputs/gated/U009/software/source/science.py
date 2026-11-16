"""Execute original slot trainer with reviewed accounting-only policy."""
import runpy,sys
from common import parents
from go2nav import research
if __name__=='__main__':
 name=sys.argv.pop(1)
 assert name=='go2nav.training.balls_slots'
 _,_,_,_,unlimited=parents()
 research.gpu_run=unlimited.accounting_run
 research.budget_remaining=unlimited.accounting_remaining
 runpy.run_module(name,run_name='__main__',alter_sys=True)
