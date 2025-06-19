import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_FEATURES = 1
OUTPUT_FEATURES = 1
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
NUM_SAMPLES = 100
RANDOM_SEED = 42 # For reproducible data

# For reproducible results
torch.manual_seed(RANDOM_SEED)

# --- 1. Data Preparation ---
# Create synthetic data: y = true_w * X + true_b + noise
X_train = torch.randn(NUM_SAMPLES, INPUT_FEATURES) * 10  # Scale X for better visualization
true_w = torch.tensor([[2.0]]) # True weight
true_b = torch.tensor([[1.0]]) # True bias
y_train = torch.matmul(X_train, true_w) + true_b + torch.randn(NUM_SAMPLES, OUTPUT_FEATURES) * 2 # Add some noise

print("--- Data Shapes ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# --- 2. Model Definition ---
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(INPUT_FEATURES, OUTPUT_FEATURES)
print("\n--- Model Architecture ---")
print(model)

# --- 3. Loss Function ---
criterion = nn.MSELoss() # Mean Squared Error Loss

# --- 4. Optimizer ---
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop ---
print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    # Forward pass: Compute predicted y by passing X to the model
    outputs = model(X_train)

    # Compute loss
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad() # Clear gradients from previous epoch
    loss.backward()       # Compute gradients of the loss w.r.t. model parameters
    optimizer.step()      # Update model parameters

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print("--- Training Finished ---")

# --- 6. Visualization (Optional) ---
print("\n--- Plotting Results ---")
predicted = model(X_train).detach().cpu().numpy() # Get predictions and detach for plotting
X_plot = X_train.detach().cpu().numpy()
y_plot = y_train.detach().cpu().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_plot, y_plot, color='blue', label='Original data', alpha=0.7)
plt.plot(X_plot, predicted, color='red', linewidth=2, label='Fitted line')
plt.title('Linear Regression with PyTorch')
plt.xlabel('X (Input Feature)')
plt.ylabel('y (Target Variable)')
plt.legend()
plt.grid(True)
plt.show()

print("Learned parameters:")
[print(name, param.data) for name, param in model.named_parameters() if param.requires_grad]