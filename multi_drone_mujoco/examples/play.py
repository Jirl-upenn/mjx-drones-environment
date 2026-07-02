"""Example: Play/visualize a trained policy.

Usage:
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip --save_rgb output.mp4
"""

import argparse
import numpy as np


def play(model_path: str, episodes: int = 3, save_rgb: str = None):
    """Load and visualize a trained policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    from multi_drone_mujoco.envs.task_aviary import TaskAviary
    from multi_drone_mujoco.envs.example_plugins import SimpleHoverPlugin

    def HoverAviary(**kwargs):
        kwargs.setdefault("episode_len_sec", 10.0)
        return TaskAviary(plugin=SimpleHoverPlugin(), **kwargs)

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    render_mode = "rgb_array" if save_rgb else None
    env = HoverAviary(ctrl_freq=48, sim_freq=240, render_mode=render_mode)

    writer = None
    if save_rgb:
        import cv2
        fps = env.CTRL_FREQ

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if save_rgb:
                frame = env.render()
                if frame is not None:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(save_rgb, fourcc, fps, (w, h))
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if terminated or truncated:
                break

        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    if writer is not None:
        writer.release()
        print(f"Video saved to: {save_rgb}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--save_rgb", type=str, default=None, metavar="OUTPUT.mp4",
                        help="Save rollout as a video file using cv2 (e.g. output.mp4)")
    args = parser.parse_args()
    play(args.model_path, args.episodes, args.save_rgb)
