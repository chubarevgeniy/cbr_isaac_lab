import torch
import torch.nn as nn
import os
import argparse
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Define the model (copied from run_pendv2_sim.py to avoid importing and triggering Isaac Sim launch)
class Policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_container = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64,32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net_container(inputs["states"]), {}

class ExportPolicy(nn.Module):
    def __init__(self, policy, scaler):
        super().__init__()
        self.net = policy.net_container
        # Register scaler parameters as buffers so they are saved with the model
        self.with_scaler = scaler is not None

        print("With scaler:", self.with_scaler)

        if scaler is not None:
            self.register_buffer("mean", scaler.running_mean.clone())
            self.register_buffer("var", scaler.running_variance.clone())
            self.clip_threshold = float(scaler.clip_threshold)

    def forward(self, x):
        # Apply scaler
        if self.with_scaler:
            print("Scaled Input:", x.dtype,self.mean.dtype)
            x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
            x = torch.clamp(x, -self.clip_threshold, self.clip_threshold)
            print("Scaled Input:", x.dtype)
        # Apply policy
        return self.net(x)

def main():
    print("Exporting SKRL policy to TorchScript...")
    parser = argparse.ArgumentParser(description="Export SKRL policy to TorchScript.")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoints'), help="Directory containing checkpoints.")
    parser.add_argument("--output_file", type=str, default="policy_script.pt", help="Output TorchScript file name.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda).")
    parser.add_argument("--preprocessor", type=bool, default=False, help="Define if there is a preprocessor.")
    args = parser.parse_args()
    do_preprocessor = args.preprocessor

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Configuration from run_pendv2_sim.py
    num_observations = 19
    num_actions = 4
    
    # Instantiate the model
    print("Instantiating Policy...")
    policy = Policy(num_observations, num_actions, device, clip_actions=False)
    
    # Instantiate state-preprocessor
    if do_preprocessor:
        print("Instantiating Scaler...")
        state_preprocessor = RunningStandardScaler(num_observations, device=device)

    # Load checkpoints
    policy_path = os.path.join(args.checkpoint_dir, "best_policy.pt")
    if do_preprocessor:
        scaler_path = os.path.join(args.checkpoint_dir, "best_state_preprocessor.pt")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy checkpoint not found at {policy_path}")
    if do_preprocessor and not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler checkpoint not found at {scaler_path}")

    print(f"Loading policy from {policy_path}")
    policy.load(policy_path)
    
    if do_preprocessor:
        print(f"Loading scaler from {scaler_path}")
        state_preprocessor.load_state_dict(torch.load(scaler_path, map_location=device))
    else:
        state_preprocessor = None

    # Create export wrapper
    print("Creating ExportPolicy wrapper...")
    export_model = ExportPolicy(policy, state_preprocessor).to(device).to(torch.float32)
    export_model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, num_observations, device=device)

    # Trace the model
    print("Tracing model with dummy input...")
    traced_script = torch.jit.trace(export_model, dummy_input)

    # Save the traced model
    output_path = os.path.join(args.checkpoint_dir, args.output_file)
    traced_script.save(output_path)
    print(f"Exported TorchScript model saved to {output_path}")

if __name__ == "__main__":
    main()