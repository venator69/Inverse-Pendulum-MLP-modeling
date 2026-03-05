import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("cartpole_dataset.csv")

X = df[["x", "x_dot", "theta", "theta_dot", "action"]].values
Y = df[["x_next", "x_dot_next", "theta_next", "theta_dot_next"]].values

# -------------------------
# Normalize data
# -------------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
Y = y_scaler.fit_transform(Y)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -------------------------
# MLP Model
# -------------------------
class CartPoleMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


model = CartPoleMLP()

# -------------------------
# Training setup
# -------------------------
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size = 256

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):

    perm = torch.randperm(X_train.size(0))

    for i in range(0, X_train.size(0), batch_size):

        idx = perm[i:i+batch_size]

        batch_x = X_train[idx]
        batch_y = y_train[idx]

        pred = model(batch_x)

        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")

# -------------------------
# Test error
# -------------------------


with torch.no_grad():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)

print("Test loss:", test_loss.item())

torch.save(model.state_dict(), "cartpole_model.pth")

joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(y_scaler, "y_scaler.save")

print("Model and scalers saved.")