import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import numpy as np
import matplotlib.pyplot as plt
import time


# make tensor.view() Module to use it in Sequential
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4*4*32, 128), nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 4*4*32), nn.Tanh(),
            Reshape(-1, 32, 4, 4),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z

# generate and save img
def generate_and_save_images(pic, epoch):
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pic.cpu().data[i, :, :, :].permute(1, 2, 0))
        plt.axis('off')
    if epoch == 0:
        plt.savefig('./result_torch/test_sample.png')
    else:
        plt.savefig('./result_torch/image_at_epoch_{:04d}.png'.format(epoch))

# load cifar10 data in torchvision.datasets
def prepare_data(batch_size):
    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    n_sample = len(trainset)
    train_size = int(n_sample * 0.8)
    print("train data: {}, validation data: {}".format(train_size, n_sample-train_size))
    subset1_indices = list(range(0,train_size))
    subset2_indices = list(range(train_size,n_sample))
    train_dataset = Subset(trainset, subset1_indices)
    val_dataset   = Subset(trainset, subset2_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    data_loaders = {"train": train_loader, "val": val_loader}
    return data_loaders


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Training Parameters
    batch_size = 32
    epochs = 30
    lr = 0.0001

    # prepare data
    data_loaders = prepare_data(batch_size)
    for inputs, _ in data_loaders['val']:
        inputs = get_torch_vars(inputs)
        generate_and_save_images(inputs,0) # save sample img
        break

    # create model
    model = CAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    def calc_loss(x,loss_func):
        x = get_torch_vars(x)
        x_pred, _ = model(x)
        loss = loss_func(x, x_pred)
        return loss, x_pred

    # train
    for epoch in range(1, epochs+1):
        start_time = time.time()
        losses = {'train': [], 'val': []}
        for phase in ['train', 'val']:
            for i, (inputs, _) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    model.train()
                    loss, _ = calc_loss(inputs, criterion)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                elif phase == 'val':
                    model.eval()
                    loss, x_pred = calc_loss(inputs, criterion)
                    if i == 0 and epoch % 5 == 0:
                        generate_and_save_images(x_pred, epoch)

                losses[phase].append(loss.data.item())

        end_time = time.time()
        print('Epoch: {}, Train set loss: {}, validation set loss: {}, time elapse for current epoch: {}'
                .format(epoch, np.array(losses['train']).mean(), np.array(losses['val']).mean(), end_time - start_time))
