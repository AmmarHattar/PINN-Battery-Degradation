import os
import json
import torch
import torch.nn as nn

# 1. We must redefine the architecture so Azure knows what shape to load
class BatteryPINN(nn.Module):
    def __init__(self):
        super(BatteryPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. init() runs ONCE when the server boots up. It loads your .pth file.
def init():
    global model
    # AZUREML_MODEL_DIR is the folder where Azure automatically put your registered model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'thermal_pinn_weights.pth')
    
    model = BatteryPINN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("PINN Model loaded into Azure successfully!")

# 3. run() executes EVERY TIME a user sends a JSON request to the API
def run(raw_data):
    try:
        # Parse the incoming JSON (e.g., {"x": 0.5, "t": 0.2})
        data = json.loads(raw_data)
        x_val = float(data['x'])
        t_val = float(data['t'])
        
        # Convert to a PyTorch tensor
        input_tensor = torch.tensor([[x_val, t_val]], dtype=torch.float32)
        
        # Ask the PINN for a prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # Return the prediction back to the user as JSON
        return json.dumps({"predicted_temperature": prediction.item()})
        
    except Exception as e:
        return json.dumps({"error": str(e)})
