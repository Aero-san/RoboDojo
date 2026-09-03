from __future__ import annotations

import unittest

from src.eval_client.episode_termination import resolve_episode_status


class EpisodeTerminationTest(unittest.TestCase):
    def test_normal_evaluation_stops_on_success_or_failure(self):
        self.assertEqual(
            resolve_episode_status(
                reward=1.0,
                success=True,
                steps=42,
                max_steps=200,
                fixed_horizon=False,
            ),
            (True, True),
        )
        self.assertEqual(
            resolve_episode_status(
                reward=0.0,
                success=False,
                steps=51,
                max_steps=200,
                fixed_horizon=False,
            ),
            (False, True),
        )

    def test_fixed_horizon_keeps_success_and_failure_running(self):
        self.assertEqual(
            resolve_episode_status(
                reward=1.0,
                success=True,
                steps=42,
                max_steps=200,
                fixed_horizon=True,
            ),
            (True, False),
        )
        self.assertEqual(
            resolve_episode_status(
                reward=0.0,
                success=False,
                steps=51,
                max_steps=200,
                fixed_horizon=True,
            ),
            (False, False),
        )

    def test_fixed_horizon_stops_exactly_at_max_steps(self):
        self.assertEqual(
            resolve_episode_status(
                reward=0.0,
                success=False,
                steps=200,
                max_steps=200,
                fixed_horizon=True,
            ),
            (False, True),
        )
        self.assertEqual(
            resolve_episode_status(
                reward=1.0,
                success=True,
                steps=200,
                max_steps=200,
                fixed_horizon=True,
            ),
            (True, True),
        )

    def test_environment_failure_remains_sticky_during_fixed_horizon(self):
        self.assertEqual(
            resolve_episode_status(
                reward=1.0,
                success=False,
                steps=100,
                max_steps=200,
                fixed_horizon=True,
            ),
            (False, False),
        )


if __name__ == "__main__":
    unittest.main()
