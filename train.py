from torchgeo.datasets import LoveDA
from torch.utils.data import DataLoader, SequentialSampler
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from model import U2NET
from pre import augment, class_to_onehot, flip_axis

# ------- 1. define loss function --------

cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')


def multi_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = cross_entropy_loss(d0, labels_v)
    loss1 = cross_entropy_loss(d1, labels_v)
    loss2 = cross_entropy_loss(d2, labels_v)
    loss3 = cross_entropy_loss(d3, labels_v)
    loss4 = cross_entropy_loss(d4, labels_v)
    loss5 = cross_entropy_loss(d5, labels_v)
    loss6 = cross_entropy_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.data.item(), loss1.data.item(
    ), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss

def multi_acc(pred, label):
    probs = torch.log_softmax(pred, dim = 1)
    _, tags = torch.max(probs, dim = 1)
    corrects = (tags == label).float()
    acc = corrects.sum()/torch.numel(corrects)
    acc = np.round(acc.data.item(), decimals=4)*100
    return acc

# ------- 2. set the directory of training dataset --------

model_name = 'u2net'
epoch_num = 500
batch_size_train = 16
batch_size_valid = 8

transforms = augment(256, 256)

train_dataset = LoveDA(root="./data", transforms=transforms)
valid_dataset = LoveDA(root="./data", split="val", transforms=transforms)

train_sampler = SequentialSampler(train_dataset)
valid_sampler = SequentialSampler(valid_dataset)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size_train ,sampler=train_sampler, shuffle=False, pin_memory=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size_valid ,sampler=valid_sampler, shuffle=False, pin_memory=True, drop_last=True)

train_num = len(train_dataset)
valid_num = len(valid_dataset)

# ------- 3. define model --------
# define the net
net = U2NET(3, 8)

# if torch.cuda.is_available():
net.cuda()
torch.manual_seed(42)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
save_frq = 2000  # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, sample in enumerate(train_dataloader):
        ite_num = ite_num + 1

        inputs, labels = sample['image'], torch.from_numpy(class_to_onehot(sample['mask'].numpy()))

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=True), Variable(labels.cuda(), requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss0, loss = multi_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        train_loss = loss.data.item()
        train_tar_loss = loss0.data.item()

        acc = multi_acc(d0, sample['mask'].cuda())

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss0, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, acc %2f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, train_loss, train_tar_loss, acc))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(),  "./data/saved_models/"+ model_name+"_ce_itr_%d_train_%3f_tar_%3f_acc_%2f.pth" %
                       (ite_num, train_loss, train_tar_loss, acc))
            net.train()  # resume train

    net.eval()

    for i, sample in enumerate(valid_dataloader):
        inputs, labels = sample['image'], torch.from_numpy(class_to_onehot(sample['mask'].numpy()))
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=True), Variable(labels.cuda(), requires_grad=True)
        
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss0, loss = multi_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        valid_loss = loss.data.item()
        valid_tar_loss = loss0.data.item()

        acc = multi_acc(d0, sample['mask'].cuda())

        del d0, d1, d2, d3, d4, d5, d6, loss0, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d] val loss: %3f, tar: %3f, acc %2f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_valid, valid_num, valid_loss, valid_tar_loss, acc))
