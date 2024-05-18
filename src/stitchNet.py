import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime

class StitchNet(nn.Module):
    """Neural network architecture for predicting the number of stitches in an image."""
    def __init__(self):
        super(StitchNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        """Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class StitchNet2(nn.Module):
    """Alternative neural network architecture for predicting the number of stitches."""
    def __init__(self):
        super(StitchNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 35, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        """Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 35)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StitchDataset(Dataset):
    """Dataset class for loading images and their corresponding stitch counts."""

    def __init__(self, images, im_names, polyline):
        """Initialize the dataset.

        Args:
            images (dict): Dictionary containing image data.
            im_names (list): List of image names.
            polyline (dict): Dictionary containing polyline data.
        """
        self.images = images
        self.im_names = im_names
        self.polyline = polyline
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
        #self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((49, 152)), transforms.ToTensor()])

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the image and the number of stitches.
        """
        img_name = self.im_names[idx]
        img = self.transform(self.images[img_name])
        stitches = len(self.polyline[img_name]['stitches'])

        return img, stitches
    

def train_nn(train_loader, net, criterion, optimizer, epochs, plot_loss=False, print_progress=False):
    """
    Trains the neural network.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        net (nn.Module): Neural network model.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        epochs (int): Number of training epochs.
        plot_loss (bool, optional): If True, plot the loss history. Defaults to False.
        print_progress (bool, optional): If True, print the loss for each epoch. Defaults to False.
    """

    loss_history = []

    # Loop over the dataset multiple times
    for epoch in range(epochs):
        running_loss = 0.0

        # Loop over the dataset
        for i, data in enumerate(train_loader):
            img, stitches = data

            optimizer.zero_grad()

            outputs = net(img)
            loss = criterion(outputs, stitches)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_history.append(running_loss)

        if print_progress:
            print(f'Epoch {epoch + 1}, loss: {running_loss}')

    # Plot the loss history
    if plot_loss:
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss history')
        plt.show()

def test(net, test_loader, train_loader, print_pred=False):
    """Tests the neural network's accuracy on test and train datasets.

    Args:
        net (nn.Module): Neural network model.
        test_loader (DataLoader): DataLoader for test data.
        train_loader (DataLoader): DataLoader for train data.
        print_pred (bool, optional): If True, print predicted and actual values. Defaults to False.

    Returns:
        float: Accuracy of the network on the test dataset.
    """
    correct = 0
    total = 0   

    with torch.no_grad():
        # Loop over the test set
        for data in test_loader:
            images, stitches = data
            outputs = net(images)

            # Get the predicted number of stitches
            _, predicted = torch.max(outputs.data, 1)
            total += stitches.size(0)

            # Count the number of correct predictions
            correct += (predicted == stitches).sum().item()

            if print_pred:
                print(f'Predicted: {predicted}')
                print(f'Actual: {stitches}')

    # Calculate the accuracy
    percentage = 100 * correct / total
    percentage = round(percentage, 2)
    print(f'Accuracy of the network on the test images: {percentage}%')

    correct = 0
    total = 0

    #same for train data
    with torch.no_grad():
        for data in train_loader:
            images, stitches = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += stitches.size(0)
            correct += (predicted == stitches).sum().item()

            if print_pred:
                print(f'Predicted: {predicted}')
                print(f'Actual: {stitches}')

    print(f'Accuracy of the network on the train images: {100 * correct / total}%')

    return percentage

#save nn to nn folder. Use date and time as name
def save(percentage, net):
    """
    Function to save the neural network model to the 'nn' folder with the filename including 
    the percentage and the current date and time.

    Parameters:
    percentage (float): The accuracy percentage or any other relevant metric to include in the filename.
    net (torch.nn.Module): The neural network model to be saved.
    """

    now = datetime.datetime.now()
    torch.save(net, f'nn_saves/{percentage}_{now}.pth')
    print(f'NN saved as {now}.pth')