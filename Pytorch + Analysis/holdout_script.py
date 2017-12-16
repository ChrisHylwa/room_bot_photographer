import torch
import torchvision as torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

classes = ('lab', 'office', 'hallway')
NUM_CLASSES = len(classes)
BATCH_SIZE = 8


data_transform  = transforms.Compose(
    [transforms.Scale(32),
     transforms.CenterCrop(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

room_dataset = datasets.ImageFolder(root='train_set/',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(room_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=2)

test_set = datasets.ImageFolder(root='test_set/', transform = data_transform)
testsetLoader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print 

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def eval(testsetLoader, net ):

    correct = 0
    total = 0
    for data in testsetLoader:

        images, labels = data
        #print("num images = "+ str(len(images)))
        #print("labeles = "+str(labels))
        #print len(labels)
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        #print("outputs = "+str(outputs))
        #print("predicted = "+str(predicted))	
        #total += labels.size(0)
       #correct += (predicted == labels).sum()

        for test_point_idx in range(0,len(predicted)):
            total += 1
            if predicted[test_point_idx] == labels[test_point_idx]:
                correct += 1

        #print("total = " + str(total))
        #print("correct = "+str(correct))

    #print('Accuracy of the network on the test images: %d %%' % (
    #100 * correct / total))

    return [correct, total] 

# get some random training images
dataiter = iter(dataset_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print(labels)
print(' '.join('%5s' % classes[labels[j]-1] for j in range(BATCH_SIZE)))
plt.pause(2)
# print labels


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataset_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    [correct,total] = eval(testsetLoader,net)
    print("Epoch "+str(epoch)+"\t"+str(correct)+"\t"+str(total))
print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.
net.train(False)

dataiter = iter(testsetLoader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(Variable(images))

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(BATCH_SIZE)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.



correct = 0
total = 0
for data in testsetLoader:

    images, labels = data
    print("num images = "+ str(len(images)))
    print("labels = "+str(labels))
    print len(labels)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print("outputs = "+str(outputs))
    print("predicted = "+str(predicted))	
    #total += labels.size(0)
    #correct += (predicted == labels).sum()

    for test_point_idx in range(0,len(predicted)):
        total += 1
        if predicted[test_point_idx] == (labels[test_point_idx]+2):
            correct += 1
#adding 1 to if conditon to line up the two arrays
    print("total = " + str(total))
    print("correct = "+str(correct))

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

#class_correct = list(0. for i in range(NUM_CLASSES))
#class_total = list(0. for i in range(NUM_CLASSES))
#for data in testsetLoader:
#    images, labels = data
#    outputs = net(Variable(images))
#    _, predicted = torch.max(outputs.data, 1)
#    c = (predicted == labels).squeeze()
#    for i in range(NUM_CLASSES):
#    label = labels[0]
#    class_correct[label] += c[0]
#    class_total[label] += 1

#commenting out the following lines since they're irrelavant for a holdout method
#for i in range(NUM_CLASSES):
#    if class_total[i] == 0: 
#	 continue
#    print('Accuracy of %5s : %2d %%' % (
#        classes[0], 100 * class_correct[0] / class_total[0]))

