"""A012 fixed primitive diagnostic; no model/checkpoint access."""
import faulthandler
import json
import time

def event(stage, **fields):
    print(json.dumps(dict(stage=stage, at=time.time(), **fields)), flush=True)

faulthandler.dump_traceback_later(10, repeat=False, exit=False)
event('import_before')
import torch
event('import_after', torch=torch.__version__, hip=torch.version.hip, source=torch.__file__)
event('device_before')
props = torch.cuda.get_device_properties(0)
assert props.total_memory >= 16 * 2**30, props
free, total = torch.cuda.mem_get_info(0)
assert 1-free/total < .85
event('device_after', name=props.name, total=total, free=free)
x = torch.arange(4096, dtype=torch.float32)
event('allocate_before')
empty = torch.empty(4096, device='cuda:0')
event('allocate_after')
event('h2d_before')
gx = x.to('cuda:0')
event('h2d_after')
torch.cuda.synchronize()
event('copy_sync_after')
event('add_before')
gy = gx + 1
torch.cuda.synchronize()
event('add_sync_after')
event('d2h_before')
y = gy.cpu()
event('d2h_after')
assert torch.equal(y, x + 1)
event('passed', exact_roundtrip=True)
faulthandler.cancel_dump_traceback_later()
