import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Load the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')

# Extract features and target variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Define a simple linear regression model using PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    Y_pred = model(X_train_tensor)

    # Compute the loss
    loss = criterion(Y_pred, Y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert the predictions back to numpy arrays
Y_train_pred = model(X_train_tensor).detach().numpy()
Y_test_pred = model(X_test_tensor).detach().numpy()

# Visualize the training set results using Seaborn
sns.scatterplot(x=X_train.flatten(), y=Y_train, color='red')
sns.lineplot(x=X_train.flatten(), y=Y_train_pred.flatten(), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set results using Seaborn
sns.scatterplot(x=X_test.flatten(), y=Y_test, color='red')
sns.lineplot(x=X_train.flatten(), y=Y_train_pred.flatten(), color='blue')
plt.title('Salary Vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# OLS Estimations
X2 = sm.add_constant(X_train)
model = sm.OLS(Y_train, X2)
result = model.fit()
print(result.summary())

# Finding intercept and coefficient
print("Coefficients:", model.params[1])
print("Intercept:", model.params[0])

# Finding prediction for 12 years of experience
new_data_tensor = torch.tensor([[12]], dtype=torch.float32)
predicted_salary = model.predict(sm.add_constant(new_data_tensor)).values[0]
print("Predicted Salary for 12 years of experience:", predicted_salary)
