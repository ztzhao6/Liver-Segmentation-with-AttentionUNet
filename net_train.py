import os
from net_framework import AttentionUNet2D
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from data_load import NrrdReader3D
import matplotlib.pyplot as plt
from loss import SoftDiceLoss
plt.switch_backend('Agg')


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, criterion, epoch, n_epochs, print_freq):
    losses = AverageMeter()

    # Model on train mode
    model.train()

    for batch_idx, sample_batched in enumerate(loader):
        if torch.cuda.is_available():
            data_var = sample_batched['data'].cuda()
            label_var = sample_batched['label'].cuda()
        else:
            data_var = sample_batched['data']
            label_var = sample_batched['label']

        # compute output
        output = model(data_var)
        loss = criterion(output, label_var)

        # measure accuracy and record loss
        batch_size = sample_batched['label'].size(0)
        losses.update(loss.data, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Loss %.4f (%.4f)' % (losses.value, losses.avg),  # losses.val is the value of this batch
            ])
            print(res)

    # Return summary statistics
    return losses.avg


def val_epoch(model, loader, criterion, print_freq=1):
    losses = AverageMeter()

    # Model on eval mode
    model.eval()

    for batch_idx, sample_batched in enumerate(loader):
        if torch.cuda.is_available():
            data_var = sample_batched['data'].cuda()
            label_var = sample_batched['label'].cuda()
        else:
            data_var = sample_batched['data']
            label_var = sample_batched['label']

        # compute output
        with torch.no_grad():
            output = model(data_var)
            loss = criterion(output, label_var)

            # measure accuracy and record loss
            batch_size = sample_batched['label'].size(0)
            losses.update(loss.data, batch_size)

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Loss %.4f (%.4f)' % (losses.value, losses.avg),
            ])
            print(res)
    print('')

    # Return summary statistics
    return losses.avg


def draw_loss(n_epochs, losses, val_losses):
    epochs_range = range(n_epochs)
    plt.plot(epochs_range, losses, '-g', label='train loss')
    plt.plot(epochs_range, val_losses, '-.k', label='validation loss')
    plt.title('loss')
    plt.legend()
    plt.savefig('figure/loss_figure.png')


def train_net(model,
              train_data_path,
              train_label_path,
              val_data_path,
              val_label_path,
              n_epochs,
              batch_size,
              # weight,
              checkpoint_dir='weights',
              lr=1e-4):

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # data_transform = transforms.RandomHorizontalFlip()

    train_dataset = NrrdReader3D(train_data_path, train_label_path)
    val_dataset = NrrdReader3D(val_data_path, val_label_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
    '''.format(n_epochs, batch_size, lr, train_dataset.__len__(),
               val_dataset.__len__()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    # criterion = nn.NLLLoss(weight=weight)
    criterion = SoftDiceLoss(n_classes=3)

    losses = []
    val_losses = []
    for epoch in range(n_epochs):
        losses_avg = train_epoch(model,
                                 train_dataloader,
                                 optimizer,
                                 criterion,
                                 epoch,
                                 n_epochs,
                                 print_freq=100)

        val_losses_avg = val_epoch(model,
                                   val_dataloader,
                                   criterion,
                                   print_freq=10)

        losses.append(round(losses_avg.cpu().numpy().tolist(), 4))
        val_losses.append(round(val_losses_avg.cpu().numpy().tolist(), 4))

        # save model parameters
        parameters_name = str(epoch) + '.pkl'
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, parameters_name))

    # save loss figure
    draw_loss(n_epochs, losses, val_losses)

    # save loss data
    with open('loss/loss.txt', 'w') as loss_file:
        loss_file.write('train loss:\n')
        for i, loss in enumerate(losses):
            output = '{' + str(i) + '}: {' + str(loss) + '}\n'
            loss_file.write(output)
        loss_file.write('-' * 50)
        loss_file.write('\n')
        loss_file.write('validation loss:\n')
        for i, val_loss in enumerate(val_losses):
            output = '{' + str(i) + '}: {' + str(val_loss) + '}\n'
            loss_file.write(output)


if __name__ == '__main__':
    # torch.cuda.set_device(1)
    net = AttentionUNet2D(n_channels=1, n_classes=2)
    train_net(model=net,
              train_data_path='',
              train_label_path='',
              val_data_path='',
              val_label_path='',
              n_epochs=50,
              batch_size=48)
              # weight=torch.FloatTensor([0.2, 15, 15]).cuda())



