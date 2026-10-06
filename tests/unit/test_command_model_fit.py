"""T6: refitting C1 from the exported contexts reproduces the shipped command model."""
import json
from pathlib import Path

import numpy as np
import pytest

from go2nav.training import command_model

SHIPPED = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_short_pulse_command_control_v1_attempt_001/command_only.npz')


@pytest.mark.skipif(not (SHIPPED.exists() and command_model.CONTEXTS.exists()), reason='needs the shipped C1 and exported contexts')
def test_refit_reproduces_shipped_c1():
    contexts = [json.loads(line) for line in command_model.CONTEXTS.read_text().splitlines()]
    model = command_model.fit(contexts)
    with np.load(SHIPPED) as shipped:
        for key in ('mean', 'scale', 'bias', 'coefficient'):
            np.testing.assert_allclose(getattr(model, key), shipped[key], rtol=1e-9, atol=1e-12)
