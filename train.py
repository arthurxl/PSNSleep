import time
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, classification_report

from data_augmentation import MixUp


def train_feature(args, model, optimizer, train_loader, lossFunc_feature):
    print("******************train_feature******************")
    mix_up = MixUp(args,0.8,100)
    model.train()
    start_time = time.time()
    total_loss = 0
    num= 0
    for step, (data, label) in enumerate(train_loader):
        num+= 1



        data = data.to(args.device)
        data_aug = mix_up(data)

        x1_online, x1_target, x2_online, x2_target=model(data,data_aug)

        loss1=lossFunc_feature(x1_online,x2_target)
        loss2=lossFunc_feature(x1_target,x2_online)
        loss=(loss1+loss2)/2

        # update
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("train_feature_time:", time.time() - start_time)
    print("loss", total_loss / num)
    return total_loss / num


def train_class_epoch(args, train_loader, context_model, model, optimizer, lossFunc_class):
    model.train()
    start = time.time()
    print("******************train_class******************")
    num = 0
    total_loss = 0
    total_acc = 0
    for i,(data, label) in enumerate(train_loader):
        num += len(label)
        data = data.to(args.device)
        label = label.to(args.device)

        with torch.no_grad():
            x_online = context_model.test(data)
        x_online = x_online.detach()
        # forward pass
        output = model(x_online)
        loss = lossFunc_class(output, label)
        total_loss += loss.item()
        total_acc += (output.argmax(1) == label).sum()
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("train_class_epoch:", time.time() - start)
    return total_acc / (num), total_loss / num


def test_class_epoch(args, test_loader, context_model, model, lossFunc_class):
    model.eval()
    start = time.time()
    print("******************test_class******************")
    num = 0
    total_loss = 0
    total_acc = 0
    y_true = []
    y_pred = []
    for i, (data, label) in enumerate(test_loader):
        num+=len(label)
        data = data.to(args.device)

        for j in range(0, len(label.reshape(-1))):
            y_true.append(label.reshape(-1)[j].cpu())
        label = label.to(args.device)


        with torch.no_grad():
            x_online= context_model.test(data)
            x_online = x_online.detach()
            # forward pass
            output = model(x_online)
            loss = lossFunc_class(output, label)
            total_loss += loss.item()
            total_acc += (output.argmax(1) == label).sum()
            output_temp = output.argmax(1)
            for j in range(0, len(output_temp.reshape(-1))):
                y_pred.append(output_temp.reshape(-1)[j].cpu())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    mf1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    k = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = classification_report(y_true, y_pred, digits=4)
    print("test_class_epoch:", time.time() - start)
    return total_acc / (num), total_loss / num, mf1, per_class_f1, k, cm, per_class_acc