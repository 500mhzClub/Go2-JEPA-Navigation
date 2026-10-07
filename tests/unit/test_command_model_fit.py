"""T6: refitting C1 from the exported contexts reproduces the shipped command model."""
import json

import numpy as np
import pytest

from go2nav import artifacts
from go2nav.training import command_model


@pytest.mark.skipif(not (artifacts.available('t6_c1_command_model') and command_model.CONTEXTS.exists()),
                    reason='needs the shipped C1 and exported contexts')
def test_refit_reproduces_shipped_c1():
    contexts = [json.loads(line) for line in command_model.CONTEXTS.read_text().splitlines()]
    model = command_model.fit(contexts)
    with np.load(artifacts.path('t6_c1_command_model')) as shipped:
        for key in ('mean', 'scale', 'bias', 'coefficient'):
            np.testing.assert_allclose(getattr(model, key), shipped[key], rtol=1e-9, atol=1e-12)
