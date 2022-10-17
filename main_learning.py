import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
import matplotlib.pyplot as plt
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
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":

    # Following tutorial at https://pytorch.org/tutorials/beginner/basics/intro.html

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Importing data
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Load the FashionMNIST dataset (training and testing) from a folder /data
    # If not available download from the internet
    # transform specifies label and feature transformations
    #   ToTensor transforms arrays and images to FloatTensor object
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    # Create a dictionary to may numeric labels in dataset to readable items
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # # Plot 9 random items from the training data with their labels
    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # Dataloader is an API that produces 'minibatches' from the dataset
    # It can also set up shuffling of the data every epoch to reduce overfitting
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # # Produce a minibatch and display first item data and image
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

    # Import the training data, this time using a transform
    #   - Images are transformed to FloatTensor object
    #   - Integer labels are transformed to a 1D Tensor with a 1 in the appropriate location
    ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        ),
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Build Neural Network
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Check if GPU available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create an instance of the Neural Net class defined above and load to device
    model = NeuralNetwork().to(device)
    print(model)

    # Pass the input data to the model which executes forward()
    # Returns a 2D tensor dim0=values of each output, dim1=probabilities of each output
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)

    # Examine the probabilities and output the result with maximum p
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    # Examine the model structure
    # prints out the weights and biases at each layer
    print(f"Model structure: {model}\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Train the model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    learning_rate = 1e-3  # How much to update parameters at each epoch
    batch_size = 64  # Number of samples to pass through before updating parameters
    epochs = 10  # Number of times to iterate

    # Set loss function
    # nn.MSELoss (Mean Square Error) for regression tasks
    # nn.NLLLoss (Negative Log Likelihood) for classification
    # nn.CrossEntropyLoss (combines nn.LogSoftmax and nn.NLLLoss)
    loss_fn = nn.CrossEntropyLoss()

    # Initialise optimiser with parameters to train and learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # # These are the functions contained in the train_loop and test_loop below
    # # zero the model gradients
    # optimizer.zero_grad()
    # # Back propagate the prediction loss to generate gradients
    # loss.backward()
    # # perform backward pass to update model parameters
    # optimizer.step()

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
    # # save weights but not structure to file
    # torch.save(model.state_dict(), 'model_weights.pth')
    # read from file
    model.load_state_dict(torch.load('model_weights.pth'))
    # set in evaluation mode i.e. module.train(False)
    model.eval()


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Examine Neural Net step by step to understand
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Generate 3 random images
    # input_image = torch.rand(3,28,28)
    # print(input_image.size())

    # # Convert each image to 1x784 array
    # flatten = nn.Flatten()
    # flat_image = flatten(input_image)
    # print(flat_image.size())

    # # apply linear transformation to data using stored weights and biases
    # # generates array of output size to be passed to results or next step
    # layer1 = nn.Linear(in_features=28*28, out_features=20)
    # hidden1 = layer1(flat_image)
    # print(hidden1.size())

    # # Introduce nonlinearity between linear layers.
    # # Use ReLU here, other options available
    # print(f"Before ReLU: {hidden1}\n\n")
    # hidden1 = nn.ReLU()(hidden1)
    # print(f"After ReLU: {hidden1}")

    # # Sequential creates an ordered container of modules - used to build a NN.
    # # Data is passed through modules in the order defined
    # seq_modules = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(in_features=28*28, out_features=20),
    #     nn.ReLU(),
    #     nn.Linear(20, 10)
    # )
    # input_image = torch.rand(3,28,28)
    # logits = seq_modules(input_image)

    # # Normalise output [-inf, inf] to [0, 1] summing to 1
    # softmax = nn.Softmax(dim=1)
    # pred_probab = softmax(logits)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Examine Optimisation via Back propagation to understand
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # weights are adjusted based on the of the gradient of the loss function to
    # # to the parameter d y_pred / d w

    # # define some simple input and output matrices
    # # required_grad tells us that we want to optimise this value and need a gradient function for it
    # x = torch.ones(5)  # input tensor
    # y = torch.zeros(3)  # expected output
    # w = torch.randn(5, 3, requires_grad=True)
    # b = torch.randn(3, requires_grad=True)
    # z = torch.matmul(x, w)+b
    # z = torch.matmul(x, w)+b

    # # If we just want to run the model, not train, disable gradient calculation to speed up
    # print(z.requires_grad)
    # with torch.no_grad():
    #     z = torch.matmul(x, w)+b
    # print(z.requires_grad)
    # # Alternatively use detach:
    # z = torch.matmul(x, w)+b
    # z_det = z.detach()
    # print(z_det.requires_grad)

    # # Define the loss function and compare prediction to actual
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    # # Compute derivatives and extract the values for w and b
    # loss.backward()
    # print(w.grad)
    # print(b.grad)
