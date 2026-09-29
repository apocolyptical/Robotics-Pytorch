import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image
from joblib import Parallel, delayed
import joblib


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes

if __name__ == "__main__":
    data_dir = 'playingCardData/train'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = PlayingCardDataset(data_dir, transform)

    # Creates a dictionary specifing what number is what card
    target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
    print(target_to_class)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Pytorch Model
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

if __name__ == "__main__":
    model = SimpleCardClassifier(num_classes=53)

    for images, labels in dataloader:
        break

    example_out = model(images)
    print(example_out.shape)

    # Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(criterion(example_out, labels))

    train_folder = 'playingCardData/train'
    valid_folder = 'playingCardData/valid'
    test_folder = 'playingCardData/test'

    train_dataset = PlayingCardDataset(train_folder, transform=transform)
    val_dataset = PlayingCardDataset(valid_folder, transform=transform)
    test_dataset = PlayingCardDataset(test_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_epoch = 5
    train_losses, val_losses = [], []

    model.to(device)

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(train_loader.dataset)
            val_losses.append(val_loss)

            # Print epoch stats
            print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Validation loss: {val_loss}")

    joblib.dump(model, "model.pkl")

    # Load and preprocess the image
    def preprocess_image(image_path, transform):
        image = Image.open(image_path).convert("RGB")
        return image, transform(image).unsqueeze(0)

    # Predict using the model
    def predict(model, image_tensor, device):
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy().flatten()

    # Visualization
    def visualize_predictions(original_image, probabilities, class_names):
        fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
        
        # Display image
        axarr[0].imshow(original_image)
        axarr[0].axis("off")
        
        # Display predictions
        axarr[1].barh(class_names, probabilities)
        axarr[1].set_xlabel("Probability")
        axarr[1].set_title("Class Predictions")
        axarr[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()

    # Example usage
    test_image = "jokerCard.jpg"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)

    # Assuming dataset.classes gives the class names
    class_names = dataset.classes 
    visualize_predictions(original_image, probabilities, class_names)