"""Export a Stable-Baselines3 TD3 actor to the ONNX contract used by run_RL_balance.py.

The output named 'joint_actions' is scaled to the Gym environment's actual action
units (radians), not the actor's internal [-1, 1] normalized action range.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3 import TD3


class ExportableTD3Actor(nn.Module):
    def __init__(self, actor: nn.Module, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.actor = actor
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, imu_state: torch.Tensor) -> torch.Tensor:
        # TD3's squashed actor produces normalized actions; convert to Box units.
        normalized_actions = self.actor(imu_state)
        return normalized_actions * self.action_scale + self.action_bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kondo_td3_v2.zip", help="Stable-Baselines3 TD3 .zip model")
    parser.add_argument("--output", default="kondo_balance_v2.onnx", help="Output ONNX filename")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    model = TD3.load(str(model_path), device="cpu")
    actor = model.policy.actor.eval()
    action_low = np.asarray(model.action_space.low, dtype=np.float32)
    action_high = np.asarray(model.action_space.high, dtype=np.float32)
    if action_low.shape != (14,) or action_high.shape != (14,):
        raise RuntimeError(f"Expected 14-action policy, found action bounds {action_low.shape} / {action_high.shape}.")

    wrapper = ExportableTD3Actor(actor, action_low, action_high).eval()
    example = torch.zeros((1, 6), dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        example,
        str(output_path),
        input_names=["imu_state"],
        output_names=["joint_actions"],
        dynamic_axes={"imu_state": {0: "batch"}, "joint_actions": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported {model_path} -> {output_path}")
    print("ONNX input: imu_state [batch, 6]")
    print("ONNX output: joint_actions [batch, 14], scaled to environment action units (radians)")

    # Check the exported graph against SB3's action scaling when ONNX Runtime is available.
    try:
        import onnxruntime as ort
        runtime = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        test_obs = np.random.default_rng(17).normal(size=(4, 6)).astype(np.float32)
        expected = np.asarray(model.predict(test_obs, deterministic=True)[0], dtype=np.float32)
        actual = runtime.run(["joint_actions"], {"imu_state": test_obs})[0]
        max_error = float(np.max(np.abs(expected - actual)))
        print(f"ONNX parity check: max absolute difference={max_error:.8g}")
        if max_error > 1e-4:
            raise RuntimeError("ONNX actions do not match model.predict() within 1e-4; do not deploy this export.")
    except ImportError:
        print("ONNX Runtime not installed here; parity check skipped. Verify ONNX outputs before hardware use.")


if __name__ == "__main__":
    main()
