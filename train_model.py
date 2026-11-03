import torch
import torch.nn as nn
import torch.optim as optim

# 1. Check for Apple Silicon GPU (MPS) and set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Generate synthetic training data: y = 3x + 2
# Let's create 100 sample points for x and calculate y
torch.manual_seed(42) # For reproducible random numbers
X_train = torch.randn(100, 1, device=device) # 100 inputs
true_w, true_b = 3.0, 2.0
y_train = X_train * true_w + true_b + torch.randn(100, 1, device=device) * 0.1 # Add a little noise

# 3. Define a simple Linear Model
# in_features=1, out_features=1
model = nn.Linear(in_features=1, out_features=1).to(device)

# 4. Define Loss Function (Mean Squared Error) and Optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) # lr = learning rate

# 5. The Training Loop (Training for 1000 epochs)
epochs = 1000
for epoch in range(epochs):
    # Step A: Forward pass (Make a prediction with current weights)
    predictions = model(X_train)
    
    # Step B: Compute the loss (How bad are the predictions?)
    loss = criterion(predictions, y_train)
    
    # Step C: Backward pass (Reset gradients, compute new gradients via autograd)
    optimizer.zero_grad()
    loss.backward()
    
    # Step D: Update weights using the optimizer
    optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. Check what the model learned!
learned_w = model.weight.item()
learned_b = model.bias.item()
print("\n--- Training Finished! ---")
print(f"Target rule: y = 3.0x + 2.0")
print(f"Model learned: y = {learned_w:.4f}x + {learned_b:.4f}")

# Test the model with a new input
test_input = torch.tensor([[5.0]], device=device)
predicted_output = model(test_input)
print(f"Prediction for input x = 5.0: {predicted_output.item():.4f} (Expected ~17.0)")