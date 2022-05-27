import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from tqdm import tqdm
import math, sys, os
import torch
from torch.utils.data import DataLoader
import torch

from dataset import CustomDataset, Operatedataset
from model import DNNModel

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0, path='./', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), "%s" %(self.path))
        self.val_loss_min = val_loss


def train(args):
    var           = args.var
    values        = args.values
    count         = args.count
    sdate         = args.sdate
    edate         = args.edate
    save_data_dir = args.save_data_dir
    station       = pd.read_csv(args.station_dir)
    data_dir      = args.data_dir

    model_dir     = args.model_dir
    epochs        = args.epochs
    patience      = args.patience
    data_mode     = args.data_mode


    early_stopping    = EarlyStopping(patience = patience, path = '%s/%s_scale_data_mode:%s_checkpoint.pt'%(model_dir, var, data_mode))

    train_dataset, valid_dataset, _, x_len = CustomDataset(save_data_dir, station, count, var, values, sdate, edate, data_dir, data_mode)
    train_loader  = DataLoader(dataset = train_dataset, batch_size = 512)
    valid_loader  = DataLoader(dataset = valid_dataset, batch_size = 128)
    print(x_len)
    if args.model == 'DNN':
        model = DNNModel(x_len)

    criterion     = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters())
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    running_results = {'train_loss':[], 'valid_loss':[]}
    for epoch in range(1,epochs+1):
        train_loss = 0
        bsizes     = 0
        train_bar = tqdm(train_loader)
        model.train()
        for data, target in train_bar:
            bsize = data.size(0)
            bsizes += bsize
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            out = model(data)
            loss = criterion(out, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.set_description(desc='[%d/%d] TrainLoss : %.5f' %(epoch, epochs, math.sqrt(train_loss / bsizes))) 
        model.eval()
        with torch.no_grad():
             valid_loss = 0
             val_sizes  = 0
             val_bar = tqdm(valid_loader)
             for vdata, vtarget in val_bar:
                 vsize = vdata.size(0)
                 val_sizes += vsize
                 if torch.cuda.is_available():
                     vdata = vdata.cuda()
                     vtarget = vtarget.cuda()
                 vout = model(vdata)
                 vloss = criterion(vout, vtarget)
                 valid_loss += vloss.item()
                 val_bar.set_description(desc='ValLoss : %.5f' %(math.sqrt(valid_loss / val_sizes)))               
        running_results['train_loss'].append(train_loss / bsizes)
        running_results['valid_loss'].append(valid_loss / val_sizes)
        
        closs = math.sqrt(valid_loss / val_sizes)
        early_stopping(closs, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


        '''
        if epoch ==1:
            best_score = closs
            torch.save(model.state_dict(), '%s/%s_scale_checkpoint.pt'%(model_dir, var))
        elif closs< best_score:
            best_score = closs
            torch.save(model.state_dict(), '%s/%s_scale_checkpoint.pt'%(model_dir ,var))
            print('Validation loss decreased (%.5f --> %.5f).  Saving model ...'%(best_score, closs))
        '''

    data_frame = pd.DataFrame(data = {'train_loss': running_results['train_loss'], 'valid_loss' : running_results['valid_loss']},\
                                index = range(1, epoch+1))
    data_frame.to_csv('%s/%s_data_mode:%s_train_result.csv'%(args.logs_dir, var, args.data_mode), index_label = 'Epoch')
def test(args):
    var           = args.var
    values        = args.values
    count         = args.count
    sdate         = args.sdate
    edate         = args.edate
    save_data_dir = args.save_data_dir
    station       = pd.read_csv(args.station_dir)
    data_dir      = args.data_dir

    model_dir     = args.model_dir
    epochs        = args.epochs
    patience      = args.patience
    data_mode     = args.data_mode
    _, _, test_dataset, x_len = CustomDataset(save_data_dir, station, count, var, values, sdate, edate, data_dir, data_mode)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 512)
    if args.model == 'DNN':
        model = DNNModel(x_len)
    model.load_state_dict(torch.load('%s/%s_scale_data_mode:%s_checkpoint.pt'%(model_dir, var, data_mode)))

    criterion     = nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_sizes  = 0
        for tdata, ttarget in test_loader:
            test_sizes += tdata.size(0)
            if torch.cuda.is_available():
                tdata = tdata.cuda()
                ttarget = ttarget.cuda()
            tout = model(tdata)
            tloss = criterion(tout, ttarget)
            test_loss += tloss

    closs = math.sqrt(test_loss / test_sizes)
    print('%s RMSE = %s'%(var, str(closs)))

def operate(args):
    var           = args.var
    values        = args.values
    count         = args.count
    sdate         = args.sdate
    edate         = args.edate
    save_data_dir = args.save_data_dir
    station       = pd.read_csv(args.station_dir)
    data_dir      = args.data_dir

    model_dir     = args.model_dir
    epochs        = args.epochs
    patience      = args.patience
    data_mode     = args.data_mode
    coordinate    = np.load(args.coordinate)[133:601, 149:787, :]
    oper_out_dir  = args.oper_out_dir%(var)

    aa = coordinate
    make_size     = aa.shape[0] * aa.shape[1]

    if os.path.exists(oper_out_dir)==False:
        os.mkdir(oper_out_dir)

    x_tensor, x_len, date_list = Operatedataset(coordinate, save_data_dir, station, count,  var, values, sdate, edate, data_dir, data_mode)

    if args.model == 'DNN':
        model = DNNModel(x_len)
    model.load_state_dict(torch.load('%s/%s_scale_data_mode:%s_checkpoint.pt'%(model_dir, var, data_mode)))
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    print(torch.cuda.is_available())
    with torch.no_grad():
        for i, x_input in enumerate(range(0,x_tensor.shape[1], make_size)):
            if torch.cuda.is_available():
                x_input = torch.as_tensor(x_tensor[0][i*make_size: (i+1)*make_size], dtype=torch.float)
                print(x_input)
                x_input = x_input.cuda()
                y_pred = model(x_input)
                y_pred = y_pred.cpu()
                print(y_pred)
            else:
                x_input = Variable(x_input)
                y_pred = model(x_input)

            y_pred = y_pred.clone().detach().numpy()
            y_pred = np.reshape(y_pred, (coordinate.shape[0], coordinate.shape[1]))
            print(y_pred.shape)
            np.save('%s/%s_%s.npy'%(oper_out_dir, var, date_list[i]),y_pred)


def run(args):
    if args.mode == 'Train':
        train(args)
    elif args.mode == 'Test':
        test(args)
    elif args.mode == 'Operate':
        operate(args)
