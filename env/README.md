# Environment

One locked Python 3.12 environment on ROCm 7.2.

- `requirements-rocm7.2.lock.txt`: the exact distribution versions the programme ran with (111 packages).
- `setup_rocm_env.sh`: creates the virtual environment, installing torch, torchvision and triton from the ROCm wheel index
  and everything else from the lock file.

The original programme bridged torch into the simulator environment through a `.pth` file pointing at a second venv. This
repository uses one environment instead.

The original harness also hard-checked the kernel, GPU driver and dpkg package versions before each episode. Here those
checks are recorded in each run's environment report rather than enforced, so the code runs on other ROCm machines.
Equivalence with the logged results is established by the replay gates, not by the environment check.
