"""Hand trigger processing uses the configured robot joint names."""

import pytest

from lerobot.configs import PipelineFeatureType
from tests.robots.test_pico_g1_teleop import pipeline_state as pipeline_state


def test_hand_mapping_preserves_custom_motor_names(pipeline_state):
    s = pipeline_state
    s.step.hand_features = ("left_hand.thumb_rotation.pos", "left_hand.index.pos", "right_hand.pinky.pos")
    result = s.processor((s.action, s.observation))
    assert result["left_hand.thumb_rotation.pos"] == 0.3
    assert result["left_hand.index.pos"] == 0.3
    assert result["right_hand.pinky.pos"] == 0.8
    features = s.step.transform_features({PipelineFeatureType.ACTION: {}})
    assert set(features[PipelineFeatureType.ACTION]) == set(result)
    assert not any("trigger" in key or "tcp" in key for key in result)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_hand_input_rejects_whole_action(pipeline_state, value):
    s = pipeline_state
    s.step.hand_features = ("left_hand.index.pos",)
    s.action["left.trigger"] = value
    with pytest.raises(ValueError, match="triggers"):
        s.processor((s.action, s.observation))
