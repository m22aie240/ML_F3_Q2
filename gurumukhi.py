
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

class SimpleNN(nn.Module):
    def __init__(self, num_classes, image_width, image_height):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(image_width * image_height, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GurmukhiDigitDataset(Dataset):
    def __init__(self, images, label_indices):
        self.images = images
        self.label_indices = label_indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.label_indices[idx]

def load_data(data_path, num_classes):
    images = []
    labels = []
    for label in range(num_classes):
        label_folder = os.path.join(data_path, str(label))
        for file in os.listdir(label_folder):
            if file.endswith('.bmp') or file.endswith('.tiff'):
                image = Image.open(os.path.join(label_folder, file))
                image_array = np.array(image).reshape(-1)
                images.append(image_array)
                labels.append(label)
    images = np.stack(images)
    labels = np.array(labels)

    first_image = Image.open(os.path.join(data_path, '0', next(iter(filter(lambda x: x.endswith('.bmp') or x.endswith('.tiff'), os.listdir(os.path.join(data_path, '0')))))))
    image_width, image_height = first_image.size

    return images, labels, image_width, image_height


def preprocess_data(images):
    images = images.astype(np.float32) / 255.0
    return images

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(data_loader), 100 * correct / total

def main():
    train_path = '/Users/ms/Downloads/train'
    val_path = '/Users/ms/Downloads/val'
    num_classes = 10
    images_train, labels_train, image_width, image_height = load_data(train_path, num_classes)
    images_val, labels_val, _, _ = load_data(val_path, num_classes)
    images_train = preprocess_data(images_train)
    images_val = preprocess_data(images_val)
    X_train, X_test, y_train, y_test = train_test_split(images_train, labels_train, test_size=0.2, random_state=42)

    X_train, X_test, X_val = torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(images_val).float()
    y_train, y_test, y_val = torch.tensor(y_train).long(), torch.tensor(y_test).long(), torch.tensor(labels_val).long()

    train_dataset = GurmukhiDigitDataset(X_train, y_train)
    test_dataset = GurmukhiDigitDataset(X_test, y_test)
    val_dataset = GurmukhiDigitDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(num_classes, image_width, image_height).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print("Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%".format(
            epoch + 1, num_epochs, train_loss, test_loss, test_accuracy, val_loss, val_accuracy))

if __name__ == "__main__":
    main()





