import torch
import numpy as np
import random
import time
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from config import parse_args
from dataset import NonSeqDataSet
from model import SiamModel,Classifier,save_model,load_model,save_model_class
from train import train_feature,train_class_epoch,test_class_epoch
from loss_function import EucLoss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def main():

    args=parse_args()
    args.time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    args.out_dir = args.time

    all_data_path = glob.glob(r'./data*.npz')

    n_folds = 10
    kf = KFold(n_splits=n_folds)
    results_fold = []

    cm_sum=torch.zeros(5,5)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write("batch_size:"+str(args.batch_size) + "\n")
        f.write("learning_rate:" + str(args.learning_rate) + "\n")
        f.write("prediction_step:" + str(args.prediction_step) + "\n")
        f.write("seq_len:" + str(args.seq_len) + "\n")

    for i in range(0, n_folds):
        print('第{}次训练'.format(i + 1))
        args.current_epoch = args.start_epoch
        args.folds = str(i + 1)

        model_feature = SiamModel(args, is_train=True).to(args.device)
        model_feature.apply(weights_init)
        params = model_feature.parameters()
        optimizer_feature = torch.optim.Adam(params, lr=args.learning_rate)

        model_class = Classifier().to(args.device)
        model_class.apply(weights_init)
        params = model_class.parameters()
        optimizer_class = torch.optim.Adam(params, lr=args.learning_rate)
        lossFunc_feature = EucLoss()
        lossFunc_class = torch.nn.CrossEntropyLoss()

        for i_fold, (train_index, test_index) in enumerate(kf.split(all_data_path)):
            if i_fold == i:
                all_test_npz_path = []
                for k in range(0, len(test_index)):
                    all_test_npz_path.append(all_data_path[test_index[k]])
                all_train_npz_path = np.setdiff1d(all_data_path, all_test_npz_path)

        train_dataset=NonSeqDataSet(all_train_npz_path)
        test_dataset=NonSeqDataSet(all_test_npz_path)

        kwargs = {'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=False, **kwargs)

        best_acc = 0
        args.current_epoch = 0
        save_acc = []
        for idx in range(0, args.num_epochs):
            print("-----------------------epoch:{}-----------------------".format(idx + 1))
            args.current_epoch = idx + 1
            loss_feature = train_feature(args, model_feature, optimizer_feature, train_loader, lossFunc_feature)
            context_model = model_feature.eval()
            acc_train, loss_class_train = train_class_epoch(args, train_loader, context_model, model_class,
                                                            optimizer_class,
                                                            lossFunc_class)
            acc_test, loss_class_test, mf1, per_class_f1, k, cm, per_class_acc = test_class_epoch(args, test_loader,
                                                                                                  context_model,
                                                                                                  model_class,
                                                                                                  lossFunc_class)
            if (idx + 1 == args.num_epochs):
                save_model(args, model_feature)
                save_model_class(args,model_class)
                cm_sum+=cm
                results_fold.append(acc_test)
                with open("./log.txt", "a") as f:
                    f.write("***************************")
                    f.write(str(i) + "\n")
                    f.write(str(cm) + "\n")
                    f.write(str(acc_test)+ "\n")
            # scheduler_feature.step()
            # scheduler_class.step()
            args.current_epoch = idx + 1
            args.mf1 = mf1
            args.per_class_f1 = per_class_f1
            args.k = k
            args.cm = cm
            args.per_class_acc = per_class_acc
            outout = os.path.join(args.out_dir, args.folds)
            if not os.path.exists(outout):
                os.makedirs(outout)
            with open(os.path.join(outout, "log.txt"), "a") as f:
                f.write("epoch:\n")
                f.write(str(args.current_epoch) + "\n")
                f.write("mf1:\n")
                f.write(str(args.mf1) + "\n")
                f.write("per_class_f1:\n")
                f.write(str(args.per_class_f1) + "\n")
                f.write("k:\n")
                f.write(str(args.k) + "\n")
                f.write("cm:\n")
                f.write(str(args.cm) + "\n")
                f.write("per_class_acc:\n")
                f.write(str(args.per_class_acc) + "\n")

            save_acc.append(acc_test)
            print("acc_train:{} ,loss_train:{}".format(acc_train, loss_class_train))
            print("acc_test:{} ,loss_test:{}".format(acc_test, loss_class_test))
            if (acc_test > best_acc):
                best_acc = acc_test
                args.best_acc = best_acc
                print("best_acc!!!:{}".format(best_acc))
        print("Final_best_acc!!!:{}".format(best_acc))
        save_acc_1 = []
        for j in range(0, len(save_acc)):
            save_acc_1.append(save_acc[j].cpu())
        plt.plot(save_acc_1)
        out = os.path.join(args.out_dir, args.folds, 'acc.png')
        plt.savefig(out)
        plt.pause(1)
        plt.close()
        del model_feature, model_class
    print(results_fold)
    print(cm_sum)
    with open("./log.txt", "a") as f:
        f.write(str(cm_sum) + "\n")

    a = np.array(cm_sum)
    acc = calculate_all_prediction(a)
    k = kappa(cm_sum)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write("\n"+"acc:"+str(acc) + "\n")
        f.write("kappa:" + str(k) + "\n")

    MF_sum = 0
    for i in range(0, 5):
        with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
            PR = calculate_label_prediction(a, i)
            f.write(str(i) + ",PR:" + str(PR)+"\n")
            RC = calculate_label_recall(a, i)
            f.write(str(i) + ",RC:" + str(RC)+"\n")
            MF = calculate_f1(PR, RC)
            f.write(str(i) + ",MF:" + str(MF)+"\n")
            MF_sum += MF

    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write("MF_sum:"+str(MF_sum / 5))
    return cm_sum




def calculate_all_prediction(confMatrix):
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction

def calculate_label_prediction(confMatrix, labelidx):

    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction

def calculate_label_recall(confMatrix, labelidx):

    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall

def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)

def kappa(confusion_matrix):
    confusion=confusion_matrix.cpu().numpy()
    pe_rows = np.sum(confusion, axis=0)
    pe_cols = np.sum(confusion, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion) / float(sum_total)
    return (po - pe) / (1 - pe)



if __name__ == '__main__':
    start_time=time.time()
    setup_seed(2022)
    cm_sum = main()
    a = np.array(cm_sum)
    acc = calculate_all_prediction(a)
    k=kappa(cm_sum)
    print("acc:", acc)
    print("kappa:",k)
    MF_sum = 0
    for i in range(0, 5):
        PR = calculate_label_prediction(a, i)
        print(str(i) + ",PR:" + str(PR))
        RC = calculate_label_recall(a, i)
        print(str(i) + ",RC:" + str(RC))
        MF = calculate_f1(PR, RC)
        print(str(i) + ",MF:" + str(MF))
        MF_sum += MF
        print("**********")
    print(MF_sum / 5)

    end_time=time.time()
    print("time:",end_time-start_time)