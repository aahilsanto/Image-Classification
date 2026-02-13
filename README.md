# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

To develop a Convolutional Neural Network (CNN) for classifying fashion apparel images from the Fashion-MNIST dataset, including data preprocessing, model training with loss visualization, and evaluation of model performance on test images as well as handwritten scanned images.

<img width="832" height="589" alt="image" src="https://github.com/user-attachments/assets/73e92ce4-5949-407f-94ce-3d4fee6d967a" />

<img width="667" height="408" alt="image" src="https://github.com/user-attachments/assets/5a0a1c45-4f07-413e-bd51-03cd82e58287" />

## Neural Network Model

<img width="1225" height="667" alt="image" src="https://github.com/user-attachments/assets/155850fb-6648-4056-8796-373a5c4703b6" />


## DESIGN STEPS

### STEP 1: 

Load Fashion-MNIST dataset from torchvision, apply transformations, and create DataLoaders for batch processing

### STEP 2: 

Build CNN architecture with 3 convolutional layers (32,64,128 filters) and 3 fully connected layers (128,64,10 nodes)

### STEP 3: 

Train model using CrossEntropyLoss and Adam optimizer while tracking training and validation loss metrics

### STEP 4: 

Evaluate model performance using confusion matrix, classification report, and test on new handwritten images

### STEP 5: 

Visualize results with loss plots and display predictions with actual vs predicted labels

## PROGRAM

### Name: Ahil Santo A
### Register Number: 212224040018
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self, x):
      x = self.pool(torch.relu(self.conv1(x)))
      x = self.pool(torch.relu(self.conv2(x)))
      x = self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print('Name: Ahil Santo A')
    print('Register Number: 212224040018')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="584" height="245" alt="image" src="https://github.com/user-attachments/assets/1d44795e-9ddf-4b68-97b8-4eebe30645a1" />

### Confusion Matrix

<img width="971" height="772" alt="image" src="https://github.com/user-attachments/assets/5e835441-1b25-434a-ba71-dd5344ce80b1" />

### Classification Report

<img width="942" height="610" alt="image" src="https://github.com/user-attachments/assets/462287be-f0db-4bb9-8e0b-bd63d3715c20" />

### New Sample Data Prediction

<img width="814" height="678" alt="image" src="https://github.com/user-attachments/assets/57f57733-3bad-42c2-8a84-3d9fd71dee62" />

## RESULT

Successfully developed and trained a CNN model on Fashion-MNIST dataset achieving good classification accuracy across 10 fashion categories with proper loss visualization and validation on test images.
