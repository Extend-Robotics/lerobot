"""Controller trigger to normalized Revo2 closure."""

import math

from lerobot.robots.brainco_revo2.config_brainco_revo2 import BraincoRevo2HandConfig


class PicoRevo2TriggerMapper:
    """Released trigger opens the hand; a full pull closes all six channels.

    Output feature names match UnitreeG1 actions.
    Rate limiting and start/pause controls belong to the shared robot session.
    """

    def __init__(self, sides: tuple[str, ...] = ("left", "right")):
        if not sides or len(set(sides)) != len(sides) or any(s not in ("left", "right") for s in sides):
            raise ValueError("Select left, right, or both hands")
        self.sides = sides

    def map(self, triggers: dict[str, float]) -> dict[str, float]:
        values = {side: float(triggers[side]) for side in self.sides}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("Controller triggers must be finite values in [0, 1]")
        return {
            f"{side}_hand.{name}.pos": values[side]
            for side in self.sides
            for name in BraincoRevo2HandConfig.motor_names
        }
