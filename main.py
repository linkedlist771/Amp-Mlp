# Reimporting necessary libraries and modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # For additional plot types
epochs = 5000000

# Reload the data
data = pd.read_excel('cascode.xlsx')
X = data.iloc[:, [4,5]].values  # Input features
y = data.iloc[:, 0:4].values    # Output features

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 50)  # Input layer to hidden layer
        self.fc2 = nn.Linear(50, 4)  # Hidden layer to output layer
        self.relu = nn.ReLU()        # ReLU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP().to(device)

# Training settings
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.8, verbose=True)

loss_history = []
test_loss_history = []  # Store test losses
progress_bar = tqdm(range(epochs), desc='Training Progress')
for epoch in progress_bar:
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # Evaluate on test set after each epoch
    model.eval()
    with torch.no_grad():
        predicted = model(X_test)
        test_loss = criterion(predicted, y_test)
        test_loss_history.append(test_loss.item())
        # Update tqdm progress bar with loss and test loss
        progress_bar.set_postfix(loss=loss.item(), test_loss=test_loss.item())
        
    # Update learning rate scheduler
    scheduler.step(test_loss)


# Evaluate on test set
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    test_loss = criterion(predicted, y_test)
    print(f'Test Loss: {test_loss.item()}')

# 1. Training and Test Loss (as before)
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='blue')
plt.plot(test_loss_history, label='Test Loss', color='orange')
plt.title('Training and Test Loss Per Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('loss_curves.png')  # Save the plot
plt.show()

# 2. Learning Rate Schedule
lrs = [group['lr'] for group in optimizer.param_groups]  # Extract learning rates
plt.figure(figsize=(10, 5))
plt.plot(lrs, color='green')
plt.title('Learning Rate Schedule', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('learning_rate_schedule.png')
plt.show()

# 3. Feature Importance (Correlation Heatmap)
correlations = data.corr()  # Calculate correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
plt.show()

# 4. Output Predictions vs. True Values (Scatterplot Matrix)
y_pred = model(X_test).cpu().detach().numpy()
y_true = y_test.cpu().numpy()

# Combine predictions and true values for plotting
results = pd.DataFrame(np.hstack([y_pred, y_true]), 
                       columns=[f'Pred_{i}' for i in range(4)] + [f'True_{i}' for i in range(4)])

sns.pairplot(results, x_vars=[f'Pred_{i}' for i in range(4)],
             y_vars=[f'True_{i}' for i in range(4)], 
             diag_kind='kde')

plt.suptitle('Output Predictions vs. True Values (Scatterplot Matrix)', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('prediction_scatterplot_matrix.png')
plt.show()