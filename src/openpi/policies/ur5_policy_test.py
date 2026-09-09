import numpy as np

from openpi.models import model as _model
from openpi.policies import ur5_policy


def test_ur5_inputs_outputs_smoke():
    inp = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    data = {
        "joints": np.zeros(6, dtype=np.float32),
        "gripper": np.zeros(1, dtype=np.float32),
        "base_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "wrist_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "prompt": "test",
        "actions": np.zeros((10, 7), dtype=np.float32),
    }
    mid = inp(data)
    assert mid["state"].shape == (7,)
    assert mid["actions"].shape == (10, 7)

    out_t = ur5_policy.UR5Outputs()
    final = out_t({"actions": np.zeros((10, 32), dtype=np.float32)})
    assert final["actions"].shape == (10, 7)


def test_ur5_single_camera_inputs_smoke():
    inp = ur5_policy.UR5SingleCameraInputs(model_type=_model.ModelType.PI05)
    data = {
        "state": np.zeros(7, dtype=np.float32),
        "base_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "prompt": "task",
        "actions": np.zeros((5, 7), dtype=np.float32),
    }
    mid = inp(data)
    assert mid["state"].shape == (7,)
    assert mid["actions"].shape == (5, 7)
    assert bool(mid["image_mask"]["base_0_rgb"])
    assert not bool(mid["image_mask"]["left_wrist_0_rgb"])
