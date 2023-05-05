import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, ):
        super(Encoder, self).__init__()
        self.fs = 100
        self.input_size = 3000
        self.num_classes = 5

        # small-size CNNs
        self.conv_time = nn.Conv1d(in_channels=1,
                                   out_channels=64,
                                   kernel_size=int(self.fs / 2),
                                   stride=int(self.fs / 16),
                                   padding=0)

        self.conv_time_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, padding=0)
        self.conv_time_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=0)
        self.conv_time_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=0)

        self.bn_time = nn.BatchNorm1d(self.conv_time.out_channels)
        self.bn_time_1 = nn.BatchNorm1d(self.conv_time_1.out_channels)
        self.bn_time_2 = nn.BatchNorm1d(self.conv_time_2.out_channels)
        self.bn_time_3 = nn.BatchNorm1d(self.conv_time_3.out_channels)

        self.dropout_time = nn.Dropout(p=0.5)

        self.pool_time_1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.pool_time_2 = nn.MaxPool1d(kernel_size=4, stride=4)


        # big-size CNNs
        self.conv_fre = nn.Conv1d(in_channels=1,
                                  out_channels=64,
                                  kernel_size=int(self.fs * 4),
                                  stride=int(self.fs / 2),
                                  padding=2)

        self.conv_fre_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, padding=2)
        self.conv_fre_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, padding=2)
        self.conv_fre_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, padding=2)

        self.bn_fre = nn.BatchNorm1d(self.conv_fre.out_channels)
        self.bn_fre_1 = nn.BatchNorm1d(self.conv_fre_1.out_channels)
        self.bn_fre_2 = nn.BatchNorm1d(self.conv_fre_2.out_channels)
        self.bn_fre_3 = nn.BatchNorm1d(self.conv_fre_3.out_channels)

        self.dropout_fre = nn.Dropout(p=0.5)

        self.pool_fre_1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool_fre_2 = nn.MaxPool1d(kernel_size=2, stride=2)


        self.dp = nn.Dropout(0.5)



    def forward(self, inputs):
        x1 = self.conv_time(inputs)
        x1 = F.relu(self.bn_time(x1))
        x1 = self.pool_time_1(x1)
        x1 = self.dropout_time(x1)

        x1 = self.conv_time_1(x1)
        x1 = F.relu(self.bn_time_1(x1))
        x1 = self.conv_time_2(x1)
        x1 = F.relu(self.bn_time_2(x1))
        x1 = self.conv_time_3(x1)
        x1 = F.relu(self.bn_time_3(x1))
        x1 = self.pool_time_2(x1)


        x2 = self.conv_fre(inputs)
        x2 = F.relu(self.bn_fre(x2))
        x2 = self.pool_fre_1(x2)
        x2 = self.dropout_fre(x2)

        x2 = self.conv_fre_1(x2)
        x2 = F.relu(self.bn_fre_1(x2))
        x2 = self.conv_fre_2(x2)
        x2 = F.relu(self.bn_fre_2(x2))
        x2 = self.conv_fre_3(x2)
        x2 = F.relu(self.bn_fre_3(x2))
        x2 = self.pool_fre_2(x2)

        x = torch.cat((x1, x2), dim=-1)
        x = self.dp(x)
        return x

class Autoregressor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(Autoregressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out



class MLP_Proj(nn.Module):
    def __init__(self):
        super(MLP_Proj, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(64 * 15, 64 * 15),
            nn.BatchNorm1d(64*15),
            nn.ReLU(),
            nn.Linear(64 * 15, 64 * 15),

        )
    def forward(self, x):
        x=self.proj(x)
        return x


class SubModel(nn.Module):
    def __init__(self,):
        super(SubModel,self).__init__()
        self.encoder = Encoder()
        self.autoregressor = Autoregressor()
        self.projection=MLP_Proj()
    def forward(self,x):
        x=self.encoder(x).permute(0, 2, 1)
        x=self.autoregressor(x)
        x = x.contiguous().view(x.size()[0], -1)
        x=self.projection(x)
        return x


class SiamModel(nn.Module):
    def __init__(self,args,is_train):
        super(SiamModel, self).__init__()
        self.online=SubModel()
        self.target=SubModel()
        self.args=args
        self.is_train=is_train
        self.prediction=nn.Sequential(
            nn.Linear(64 * 15,64 * 15),
            nn.BatchNorm1d(64 * 15),
            nn.ReLU(),
            nn.Linear(64 * 15,64 * 15),
        )


        self.m = 0.1
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data = param_target.data * self.m + param_online.data * (1. - self.m)

    def trian(self,x1,x2):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        x1_online = self.online(x1)
        x1_online = self.prediction(x1_online)

        x1_target = self.target(x1)

        x2_online = self.online(x2)
        x2_online = self.prediction(x2_online)

        x2_target = self.target(x2)
        return x1_online, x1_target, x2_online, x2_target


    def test(self,x1):
        x1_online = self.online(x1)

        return x1_online


    def forward(self,x1,x2):
        if(self.is_train):

            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()
            x1_online = self.online(x1)
            x1_online = self.prediction(x1_online)

            x1_target = self.target(x1)

            x2_online = self.online(x2)
            x2_online = self.prediction(x2_online)

            x2_target = self.target(x2)
            return x1_online,x1_target,x2_online,x2_target

        else:
            x1_online = self.online(x1)

            return x1_online


class Classifier(nn.Module):
    def __init__(self, length=15):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 15, 64),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.classifier(x)


def save_model(args, model):
    outout = os.path.join(args.out_dir, args.folds)
    if not os.path.exists(outout):
        os.makedirs(outout)
    out = os.path.join(outout, "checkpoint_{}.tar".format(args.current_epoch))
    torch.save(model.state_dict(), out)

def save_model_class(args, model):
    outout = os.path.join(args.out_dir, args.folds)
    if not os.path.exists(outout):
        os.makedirs(outout)
    out = os.path.join(outout, "checkpoint_class_{}.tar".format(args.current_epoch))
    torch.save(model.state_dict(), out)

def load_model(args,path, epoch,model):

    print("### LOADING MODEL FROM checkpoint ###")
    model_fp = os.path.join(path, args.folds,"checkpoint_{}.tar".format(epoch))
    model.load_state_dict(torch.load(model_fp))
    model = model.to(args.device)
    return model
