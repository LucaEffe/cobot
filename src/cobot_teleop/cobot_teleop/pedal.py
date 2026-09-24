"""
Foot pedal listener (PCsensor FootSwitch via evdev).

Turns raw key events into high-level teach events:
    F13 -> "waypoint"
    F14 -> "toggle_gripper"
    F15 -> "finish"

Kept separate from the node so the mapping is in one obvious place and the
node only deals with the resulting high-level events.
"""

from __future__ import annotations

from typing import Iterator

import evdev


# high-level event names
WAYPOINT = "waypoint"
TOGGLE_GRIPPER = "toggle_gripper"
FINISH = "finish"


class PedalListener:
    KEYMAP = {
        evdev.ecodes.KEY_F13: WAYPOINT,
        evdev.ecodes.KEY_F14: TOGGLE_GRIPPER,
        evdev.ecodes.KEY_F15: FINISH,
    }

    def __init__(self, device_path: str):
        self.device = evdev.InputDevice(device_path)

    @property
    def name(self) -> str:
        return self.device.name

    def events(self) -> Iterator[str]:
        """Yield high-level event names on key-down (blocking read loop)."""

        for event in self.device.read_loop():
            if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                name = self.KEYMAP.get(event.code)
                if name is not None:
                    yield name
