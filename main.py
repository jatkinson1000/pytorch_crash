import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image


# Generate a custom class object to store dataset
# Methods to:
# 	- init, read in dataset
# 	- getitem, return a single item
class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# subclass nn.Module
# Methods to:
# 	- init, initialise the layers
# 	- getitem, return a single item
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print("".join([
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, ",
        f"Avg loss: {test_loss:>8f} \n"
    ])
    )


if __name__ == "__main__":

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Importing data
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Load the FashionMNIST dataset (training and testing) from a folder /data
    # If not available download from the internet
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    # Dataloader is an API that produces 'minibatches' from the dataset
    # It can also set up shuffling of data every epoch to reduce overfitting
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Build Neural Network
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Check if GPU available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create instance of the Neural Net class defined above and load to device
    model = NeuralNetwork().to(device)
    print(model)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Train the model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    learning_rate = 1e-3  # How much to update parameters at each epoch
    epochs = 10  # Number of times to iterate

    # Set loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialise optimiser with parameters to train and learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Saving model and re-loading
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # save full model including structure to file
    torch.save(model, 'model.pth')
    # read from file
    model = torch.load('model.pth')
    # # save weights but not structure to file
    # torch.save(model.state_dict(), 'model_weights.pth')
    # # read from file
    # model.load_state_dict(torch.load('model_weights.pth'))

    # set in evaluation mode i.e. module.train(False)
    model.eval()

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Example of using the model on the test data
    for i in range(0, 10):
        x, y = test_data[i][0], test_data[i][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
