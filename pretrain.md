---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: seismic-detection
    language: python
    name: python3
---

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import UnsupervisedLunarSeismicDataset, collate_fn, train_test_split_dataset
from models import CNNAutoencoder
```

```python
# Define model and training hyperparameters
num_epochs = 30
batch_size=16
learning_rate=1e-3
data_dir = 'data/lunar/unsupervised'
```

```python
# Load the dataset
unsupervised_dataset = UnsupervisedLunarSeismicDataset(data_dir=data_dir)
train_dataset, test_dataset = train_test_split_dataset(unsupervised_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
```

```python
# Init the model
autoencoder = CNNAutoencoder()

# Training set up
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
```

```python
# Training Loop
for epoch in range(num_epochs):
    autoencoder.train()
    for inputs in train_loader:
        if inputs is None:
            continue
        inputs = inputs.unsqueeze(1)  # Add channel dimension
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')

    # Evaluate on the test set
    autoencoder.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs in test_loader:
            if inputs is None:
                continue
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}')

    # Save model checkpoint
    checkpoint_path = f'./checkpoints/CNNAutoencoder/model_epoch_{epoch+1}.pth'
    torch.save(autoencoder.state_dict(), checkpoint_path)
    print(f'Model saved to {checkpoint_path}')
```
