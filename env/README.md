# Environment

One locked Python 3.12 environment on ROCm 7.2, plus a second, render-only environment for the May corpus.

- `requirements-rocm7.2.lock.txt`: the exact distribution versions the programme ran with (111 packages).
- `setup_rocm_env.sh`: creates the virtual environment, installing torch, torchvision and triton from the ROCm wheel index
  and everything else from the lock file.

The original programme bridged torch into the simulator environment through a `.pth` file pointing at a second venv. This
repository uses one environment instead.

The original harness also hard-checked the kernel, GPU driver and dpkg package versions before each episode. Here those
checks are recorded in each run's environment report rather than enforced, so the code runs on other ROCm machines.
Equivalence with the logged results is established by the replay gates, not by the environment check.

## Corpus render environment

The May corpus frames were rendered with genesis-world 0.3.14 on its Vulkan backend, in a separate environment
(`requirements-render.lock.txt`, created by `setup_render_env.sh`); the rollouts themselves run in the main
environment on Genesis 0.4.6. Only `go2nav.corpus.render` runs there, launched by
`python -m go2nav.corpus.datagen render ... --render-python <render venv>/bin/python`. The OpenGL device is chosen
through EGL: `EGL_DEVICE_ID` (the datagen `--egl-device`, 1 on the development machine, the discrete R9700).
