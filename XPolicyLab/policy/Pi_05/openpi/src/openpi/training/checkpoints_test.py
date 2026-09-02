from flax import nnx
import jax.numpy as jnp
import optax

from openpi.shared import array_typing as at
from openpi.training import checkpoints
from openpi.training import utils


class _RecordingCheckpointManager:
    def __init__(self):
        self.events = []

    def save(self, step, items):
        del items
        self.events.append(("save", step))

    def wait_until_finished(self):
        self.events.append(("wait", None))


class _UnusedDataLoader:
    def data_config(self):
        raise AssertionError("The assets callback should stay lazy in this test.")


def test_save_waits_before_donated_state_can_be_reused():
    params = nnx.State({"weight": nnx.VariableState(nnx.Param, jnp.ones((2, 2)))})
    with at.disable_typechecking():
        state = utils.TrainState(
            step=jnp.array(1),
            params=params,
            model_def=None,
            tx=optax.sgd(0.1),
            opt_state=(),
            ema_decay=None,
            ema_params=None,
        )
    manager = _RecordingCheckpointManager()

    checkpoints.save_state(manager, state, _UnusedDataLoader(), step=100)

    assert manager.events == [("save", 100), ("wait", None)]
