"""
Export PyTorch Poker AI to ONNX for Browser Use
================================================
This script converts the trained PyTorch model to ONNX format
for use with ONNX.js in the browser.
"""

import torch
import torch.nn as nn

# Define the same network architecture as training
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super(DuelingDQN, self).__init__()
        
        # Common Feature Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Stream 1: Value
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Stream 2: Advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine them: Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

def export_to_onnx():
    print("Loading trained PyTorch Dueling DQN model...")
    
    # Create model and load weights
    model = DuelingDQN(input_dim=4, output_dim=4)
    checkpoint = torch.load("poker_dueling_model.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Model loaded. Epsilon was: {checkpoint['epsilon']:.4f}")
    
    # Create dummy input matching expected shape
    # Input: [MyStrength, PotOdds, Stage, StackRatio]
    # StackRatio = PlayerStack / StartingStack (e.g. 1000/1000 = 1.0)
    dummy_input = torch.randn(1, 4)
    
    # Export to ONNX
    output_path = "poker_ui_lite/poker_ai.onnx"
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['game_state'],
        output_names=['action_values'],
        dynamic_axes={
            'game_state': {0: 'batch_size'},
            'action_values': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"✅ Model exported to {output_path}")
    
    # Verify the export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified successfully!")
    
    # Test with ONNX Runtime
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession(output_path)
    # Test Input: [Strength=0.8, Odds=0.1, Stage=0, StackRatio=1.0]
    test_input = np.array([[0.8, 0.1, 0.0, 1.0]], dtype=np.float32)
    result = session.run(None, {'game_state': test_input})
    
    print(f"\nTest inference:")
    print(f"  Input: {test_input[0]}")
    print(f"  Output Q-values: {result[0][0]}")
    print(f"  Best action: {np.argmax(result[0][0])} ({['FOLD', 'CALL', 'RAISE_MIN', 'RAISE_BIG'][np.argmax(result[0][0])]})")

if __name__ == "__main__":
    export_to_onnx()
