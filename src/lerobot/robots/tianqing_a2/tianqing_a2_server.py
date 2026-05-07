#!/usr/bin/env python

# Copyright 2025 Project contributors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tianqing A2 mock server for local development and testing.

This script simulates the robot controller side of the ZMQ bridge:
  - Binds a PUB socket on ``state_port`` and publishes fake joint state at ~50 Hz.
  - Binds a SUB socket on ``action_port`` and logs received action commands.

Usage::

    python -m lerobot.robots.tianqing_a2.tianqing_a2_server

Press Ctrl+C to stop.
"""

import json
import logging
import math
import signal
import time

import zmq

# ---- defaults (must match TianqingA2Config) ----
STATE_PORT = 5555
ACTION_PORT = 6666
STATE_TOPIC = "tianqing_state"
ACTION_TOPIC = "tianqing_action"
PUBLISH_HZ = 50.0

MOTOR_NAMES = [
    "joint51", "joint52", "joint53", "joint54", "joint55", "joint56", "joint57",
    "joint61", "joint62", "joint63", "joint64", "joint65", "joint66", "joint67",
    "left_joint1", "right_joint1",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fake_state(t: float) -> dict[str, float]:
    """Generate slowly oscillating joint positions for visual testing."""
    return {name: round(math.sin(t + i * 0.3) * 0.05, 6) for i, name in enumerate(MOTOR_NAMES)}


def main() -> None:
    ctx = zmq.Context()

    # Server PUB: bind so clients can connect and subscribe
    state_pub = ctx.socket(zmq.PUB)
    state_pub.bind(f"tcp://*:{STATE_PORT}")

    # Server SUB: bind so clients can connect and publish actions to us
    action_sub = ctx.socket(zmq.SUB)
    action_sub.bind(f"tcp://*:{ACTION_PORT}")
    action_sub.setsockopt_string(zmq.SUBSCRIBE, ACTION_TOPIC)
    action_sub.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking

    logger.info(f"Tianqing A2 mock server started.")
    logger.info(f"  Publishing state  -> tcp://127.0.0.1:{STATE_PORT}  (topic: {STATE_TOPIC})")
    logger.info(f"  Receiving actions <- tcp://127.0.0.1:{ACTION_PORT} (topic: {ACTION_TOPIC})")
    logger.info("Press Ctrl+C to stop.\n")

    stop = False

    def _handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    interval = 1.0 / PUBLISH_HZ
    last_pub = time.perf_counter()

    while not stop:
        now = time.perf_counter()

        # Publish state at PUBLISH_HZ
        if now - last_pub >= interval:
            state = _fake_state(now)
            payload = json.dumps(state)
            state_pub.send_string(f"{STATE_TOPIC} {payload}")
            last_pub = now

        # Drain incoming action messages
        try:
            while True:
                raw = action_sub.recv_string(zmq.NOBLOCK)
                _, _, payload = raw.partition(" ")
                action = json.loads(payload)
                logger.info(f"Received action: { {k: round(v, 4) for k, v in action.items()} }")
        except zmq.Again:
            pass

        time.sleep(0.001)

    logger.info("Shutting down server...")
    state_pub.close()
    action_sub.close()
    ctx.term()
    logger.info("Server stopped.")


if __name__ == "__main__":
    main()
