"""Exercise the real PI05 checkpoint loader with a small model."""

from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class TinyPI05(PI05Policy):
    def __init__(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Linear(2, 2)

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        return state_dict


def test_pi05_loads_weights_and_forwards_download_options(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    weights = {"model.weight": torch.full((2, 2), 3.0), "model.bias": torch.ones(2)}
    save_file(weights, path)
    cached_file = Mock(return_value=str(path))
    monkeypatch.setattr("transformers.utils.cached_file", cached_file)
    config = PI05Config(device="cpu")
    model = TinyPI05.from_pretrained("test/repo", config=config, revision="pinned", local_files_only=True)
    torch.testing.assert_close(model.model.weight, weights["model.weight"])
    assert cached_file.call_args.kwargs["revision"] == "pinned"
    assert cached_file.call_args.kwargs["local_files_only"] is True


@pytest.mark.parametrize("failure", ["download", "missing_keys", "shape"])
def test_pi05_never_silently_returns_unloaded_weights(tmp_path, monkeypatch, failure):
    path = tmp_path / "model.safetensors"
    weights = {"model.weight": torch.ones((3, 2) if failure == "shape" else (2, 2))}
    save_file(weights, path)
    cached_file = Mock(return_value=str(path))
    if failure == "download":
        cached_file.side_effect = OSError("unavailable")
    monkeypatch.setattr("transformers.utils.cached_file", cached_file)
    with pytest.raises(RuntimeError, match="Could not load pretrained PI05 weights"):
        TinyPI05.from_pretrained("test/repo", config=PI05Config(device="cpu"))
