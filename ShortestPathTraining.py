import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import pdb


def readData():
    with open("datasetShortestPath", "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        return(b)
        pdb.set_trace()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4




#trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                          shuffle=True, num_workers=2)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      # download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#classes = ("beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", 	"bottles", "bowls", "cans", "cups", "plates", "apples", "mushrooms", "oranges", "pears", "sweet peppers"
#           , "clock", "computer" "keyboard", "lamp", "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road", "skyscraper"
#           , "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee", "elephant", "kangaroo","fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman", 	"crocodile", "dinosaur", "lizard", "snake", "turtle"
#           , "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", 	"lawn-mower", "rocket", "streetcar", "tank", "tractor")

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, start, goal, currPos):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        xnew = torch.concat(x, start, goal, currPos)
        x = F.relu(self.fc2(xnew))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    
    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dataset = readData()

    for epoch in range(2):
        for i in range(len(dataset)):
            d = dataset[i]
            inputs = d[0]
            labels = d[1]
            for j in range(len(inputs)):
                image = inputs[j][0]
                start = inputs[j][1]
                end = inputs[j][2]
                intermediate_labels = labels[j]
                image_t = torch.tensor(np.array(image), dtype=torch.float32)
                start_t = torch.tensor(np.array(start), dtype=torch.float32)
                end_t = torch.tensor(np.array(end), dtype=torch.float32)
                #do same for start and end
                pdb.set_trace()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, start_t, end_t)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')