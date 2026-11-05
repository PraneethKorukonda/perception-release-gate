from pathlib import Path

import pandas as pd
import pytest

from releasegate import sim
from releasegate.gate import ModelRun
from releasegate.match import match

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def small_sim():
    """40 simulated clips with all three models matched. Shared, so don't mutate it."""
    d = sim.simulate(sim.SimConfig(n_videos=40, frames_per_video=60), seed=3)
    runs = {}
    for name, preds in d.preds.items():
        gm, pm = match(d.gt, preds)
        runs[name] = ModelRun(name, gm, pm)
    return d, runs


def boxes_df(rows, columns):
    return pd.DataFrame(rows, columns=columns)
