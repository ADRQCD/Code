import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
print("train data length ：{}".format(train_data_size))


batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False, drop_last = True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten input to fit into the linear layer
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)          # Output layer (raw logits)
        return x


SimpleNN = SimpleNN()

loss_fn = nn.CrossEntropyLoss(reduction='none')

learning_rate = 1e-2
optimizer = torch.optim.SGD(SimpleNN.parameters(), lr=learning_rate)

total_train_step = 0


epoch = 50
# store current loss for all samples
loss_matrix = torch.zeros(train_data_size)

for i in range(epoch):
    print("------- {} epoch begins-------".format(i+1))
    batch_n = 0
    SimpleNN.train()
    for data in train_dataloader:
        imgs, targets = data
        if i == 0:
            outputs = SimpleNN(imgs)
            loss = loss_fn(outputs, targets)
            loss_matrix[batch_n * batch_size : (batch_n+1) * batch_size] = loss  # drop_last = True
        else :
            valid_loss, idxs = torch.topk(loss_matrix[batch_n * batch_size : (batch_n + 1)* batch_size], int(0.5 * batch_size))
            outputs = SimpleNN(imgs[idxs])
            loss = loss_fn(outputs, targets[idxs])
            loss_matrix[idxs] = loss
        loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_n  = batch_n + 1
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("train_step：{}, Loss: {}".format(total_train_step, loss.item()))



