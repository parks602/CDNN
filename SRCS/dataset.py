import numpy as np
import pandas as pd
from scipy.io import FortranFile
from datetime import datetime, timedelta
import sys, os
import torch
import torch.utils.data as data_utils 
from torch.utils.data.dataset import random_split

def Fortran2Npy(fname):
    with FortranFile(fname, 'r') as f:
      info    = f.read_ints(np.int32)
      stnlist = f.read_ints(np.int32)
      data    = f.read_reals(np.float32)
      data    = np.reshape(data, info[-5:])
    return data
    

def find_near(station, count, save_data_dir):
    for c in range(count):
        station['near_%s'%(str(c))] = np.nan

    ix = station.latitude
    iy = station.longitude
    iz = station.altitude
    for h in range(len(ix)):
        dist = cal_dist(ix, iy, iz, ix[h], iy[h], iz[h])
        pos =  np.unravel_index(dist.argmin(),dist.shape)
        dist[pos] = np.inf

        for k in range(count):
            pos =  np.unravel_index(dist.argmin(),dist.shape)
            station['near_%s'%(str(k))][h]=pos[0]
            dist[pos] = np.inf
    station.to_csv('%s/near_spot_document.csv'%(save_data_dir))
    return(station)

def oper_find_near(station, count, oper_data_dir, coorperate):
    col = ['longitude', 'latitude', 'altitude']
    stat= pd.DataFrame(np.reshape(coorperate,(coorperate.shape[0]*coorperate.shape[1],3)), columns=col)
    for c in range(count):
        stat['near_%s'%(str(c))] = np.nan

    ix = coorperate[:,:,1].flatten()
    iy = coorperate[:,:,0].flatten()
    iz = coorperate[:,:,2].flatten()

    stx = station.latitude
    sty = station.longitude
    stz = station.altitude

    for h in range(len(ix)):
        print(h)
        dist = cal_dist(stx, sty, stz, ix[h], iy[h], iz[h])
        pos =  np.unravel_index(dist.argmin(),dist.shape)
        dist[pos] = np.inf

        for k in range(count):
            pos =  np.unravel_index(dist.argmin(),dist.shape)
            stat['near_%s'%(str(k))][h]=pos[0]
            dist[pos] = np.inf
    stat.to_csv('%s/half_near_spot_document.csv'%(oper_data_dir))
    return(stat)
      
    
def MinMax(arr, arr2):
    minmax = (arr-arr2[0])/(arr2[1]-arr2[0])
    return(minmax)


def cal_dist(mx, my, mz, ox, oy, oz):
    return np.sqrt((mx - ox)**2 + (my - oy)**2 + (mz - oz)**2)


def near_data_document(station, var, count, save_data_dir):
    col = ['Lat', 'Lon', 'Hi']
    for k in range(count):
        col.append('Cos_Lat_%d'%k)
        col.append('Cos_Lon_%d'%k)
        col.append('Cos_Hi_%d'%k)
        col.append('Distance_%d'%k)
    dataset = pd.DataFrame(columns=col)

    dataset['Lat'] = station['latitude']
    dataset['Lon'] = station['longitude']
    dataset['Hi']  = station['altitude']

    for k in range(count):
        dataset['Distance_%d'%k] = np.sqrt(\
              ((dataset['Lat'].values - dataset['Lat'][station['near_%d'%k].reset_index(drop=True)].values)**2 +\
              (dataset['Lon'].values - dataset['Lon'][station['near_%d'%k].reset_index(drop=True)].values)**2 +\
              (dataset['Hi'].values - dataset['Hi'][station['near_%d'%k].reset_index(drop=True)].values)**2))

        dataset['Cos_Lat_%d'%k] = \
      np.sqrt((dataset['Lat'].values - dataset['Lat'][station['near_%d'%k].reset_index(drop=True)].values)**2) / \
      dataset['Distance_%d'%k].values
        dataset['Cos_Lon_%d'%k] = \
      np.sqrt((dataset['Lon'].values - dataset['Lon'][station['near_%d'%k].reset_index(drop=True)].values)**2) / \
      dataset['Distance_%d'%k].values
        dataset['Cos_Hi_%d'%k] = \
      np.sqrt((dataset['Hi'].values - dataset['Hi'][station['near_%d'%k].reset_index(drop=True)].values)**2) / \
      dataset['Distance_%d'%k].values
    dataset = dataset.fillna(0)
    dataset.to_csv('%s/half_near_station_with_count_%d.csv'%(save_data_dir, count))
    
    return dataset

def find_null(values):
    max_num = 0
    count = 0
    for i, v in enumerate(values):
        test = globals()['{}'.format(v)]
        test = np.reshape(test, (test.shape[0], test.shape[1]* test.shape[2]* test.shape[3]* test.shape[4]))
        test = test.transpose()
        test[test<-99] = np.nan
        test = pd.DataFrame(test)
        null_test = test.isnull().sum()
        null_num = len(null_test[null_test>int(test.shape[0]*0.1)])
        print(values[i], null_num)
        if i == 0:
            count = count
            max_num = null_num
            del_index = null_test[null_test>int(test.shape[0]*0.1)].index
        elif max_num < null_num:
            count = i
            max_num = null_num
            del_index = null_test[null_test>int(test.shape[0]*0.1)].index

    print(values[count], 'has has the most determination. count is %i'%(max_num))

    return del_index

def var_data_maker(var, values, document, data_dir, sdate, edate, count, station, save_data_dir):
    col = ['Year', 'Month', 'Day', 'Hour', 'Lat', 'Lon', 'Hi']
    for k in range(count):
        col.append('Cos_Lat_%d'%k)
        col.append('Cos_Lon_%d'%k)
        col.append('Cos_Hi_%d'%k)
        col.append('Distance_%d'%k)
        for i in values:
            col.append('%s_%d'%(i,k))
    col.append('Y')
    dataset = pd.DataFrame(columns=col)

    for v in values:
        globals()['{}'.format(v)] = Fortran2Npy('%s/obs_%s.2020010100-2020123123'%(data_dir, v.lower()))
    del_index = find_null(values)

    for v in values:
        test = globals()['{}'.format(v)]
        osize, y,m,d,h = test.shape[0], test.shape[1], test.shape[2], test.shape[3], test.shape[4]
        test = np.reshape(test, (osize, y*m*d*h))
        test = test.transpose()
        test[test<-90] = np.nan
        test = pd.DataFrame(test)
        test = test.drop(del_index, axis=1)
        test = test.interpolate(limit_direction = 'both').values
        test = test.transpose()
        test = np.reshape(test, (test.shape[0], y, m ,d, h))
        globals()['{}'.format(v)] = test

    fmt = '%Y%m%d%H'
    dt_sdate = datetime.strptime(sdate, fmt)
    dt_edate = datetime.strptime(edate, fmt)
    now = dt_sdate
    
    station_num = len(station)
    print(station_num)
    t = 0
    while now <= dt_edate:
        print(now)
        dataset = dataset.append(pd.DataFrame({"Year": range(station_num)}))
        dataset.Year[t*station_num : (t+1)*station_num] = now.year
        dataset.Month[t*station_num : (t+1)*station_num] = now.month
        dataset.Day[t*station_num : (t+1)*station_num] = now.day
        dataset.Hour[t*station_num : (t+1)*station_num] = now.hour
        dataset.Lat[t*station_num : (t+1)*station_num] = document.Lat.values
        dataset.Lon[t*station_num : (t+1)*station_num] = document.Lon.values
        dataset.Hi[t*station_num : (t+1)*station_num] = document.Hi.values
        dataset.Y[t*station_num : (t+1)*station_num] = globals()['{}'.format(var)][:,2020-now.year, now.month-1, now.day-1, now.hour]
        for k in range(count):
            dataset['Cos_Lat_%d'%k][t*station_num : (t+1)*station_num] = document['Cos_Lat_%d'%k].values
            dataset['Cos_Lon_%d'%k][t*station_num : (t+1)*station_num] = document['Cos_Lon_%d'%k].values
            dataset['Cos_Hi_%d'%k][t*station_num : (t+1)*station_num] = document['Cos_Hi_%d'%k].values
            dataset['Distance_%d'%k][t*station_num : (t+1)*station_num] = document['Distance_%d'%k].values
            for j in values:
                dataset['%s_%d'%(j, k)][t*station_num : (t+1)*station_num] =  globals()['{}'.format(j)][station['near_%d'%k].values.astype(np.int64), 2020-now.year, now.month-1, now.day-1, now.hour]
        t = t+1
        now = now + timedelta(hours=1)
    dataset = dataset[col]
    dataset.to_csv('%s/%s_%s-%s_count:%s.csv'%(save_data_dir, var, sdate, edate, str(count)))

    return dataset


def oper_data_maker(var, values, document, data_dir, sdate, edate, count, station, save_data_dir):
    col = ['Year', 'Month', 'Day', 'Hour', 'Lat', 'Lon', 'Hi']
    date_list  = []
    for k in range(count):
        col.append('Cos_Lat_%d'%k)
        col.append('Cos_Lon_%d'%k)
        col.append('Cos_Hi_%d'%k)
        col.append('Distance_%d'%k)
        for i in values:
            col.append('%s_%d'%(i,k))
    #col.append('Y')
    dataset = pd.DataFrame(columns=col)

    for v in values:
        globals()['{}'.format(v)] = Fortran2Npy('%s/obs_%s.2020010100-2020123123'%(data_dir, v.lower()))
    del_index = find_null(values)

    for v in values:
        test = globals()['{}'.format(v)]
        osize, y,m,d,h = test.shape[0], test.shape[1], test.shape[2], test.shape[3], test.shape[4]
        test = np.reshape(test, (osize, y*m*d*h))
        test = test.transpose()
        test[test<-90] = np.nan
        test = pd.DataFrame(test)
        test = test.drop(del_index, axis=1)
        test = test.interpolate(limit_direction='both').values
        test = test.transpose()
        test = np.reshape(test, (test.shape[0], y, m ,d, h))
        globals()['{}'.format(v)] = test

    fmt = '%Y%m%d%H'
    dt_sdate = datetime.strptime(sdate, fmt)
    dt_edate = datetime.strptime(edate, fmt)
    now = dt_sdate

    station_num = len(station)
    print(station_num)
    t = 0
    while now <= dt_edate:
        date_list.append(datetime.strftime(now, fmt))
        print(now)
        dataset = dataset.append(pd.DataFrame({"Year": range(station_num)}))
        dataset.Year[t*station_num : (t+1)*station_num] = now.year
        dataset.Month[t*station_num : (t+1)*station_num] = now.month
        dataset.Day[t*station_num : (t+1)*station_num] = now.day
        dataset.Hour[t*station_num : (t+1)*station_num] = now.hour
        dataset.Lat[t*station_num : (t+1)*station_num] = document.Lat.values
        dataset.Lon[t*station_num : (t+1)*station_num] = document.Lon.values
        dataset.Hi[t*station_num : (t+1)*station_num] = document.Hi.values
        #dataset.Y[t*station_num : (t+1)*station_num] = globals()['{}'.format(var)][:,2020-now.year, now.month-1, now.day-1, now.hour]
        for k in range(count):
            dataset['Cos_Lat_%d'%k][t*station_num : (t+1)*station_num] = document['Cos_Lat_%d'%k].values
            dataset['Cos_Lon_%d'%k][t*station_num : (t+1)*station_num] = document['Cos_Lon_%d'%k].values
            dataset['Cos_Hi_%d'%k][t*station_num : (t+1)*station_num] = document['Cos_Hi_%d'%k].values
            dataset['Distance_%d'%k][t*station_num : (t+1)*station_num] = document['Distance_%d'%k].values
            for j in values:
                dataset['%s_%d'%(j, k)][t*station_num : (t+1)*station_num] =  globals()['{}'.format(j)][station['near_%d'%k].values.astype(np.int64), 2020-now.year, now.month-1, now.day-1, now.hour]
        t = t+1
        now = now + timedelta(hours=1)
    dataset = dataset[col]
    dataset.to_csv('%s/half_oper_%s_%s-%s_count:%s.csv'%(save_data_dir, var, sdate, edate, str(count)))

    return dataset, date_list


def CustomDataset(save_data_dir, station, count,  var, values, sdate, edate, data_dir, data_mode):

    if os.path.exists('%s/near_spot_document.csv'%(save_data_dir))==False:
        station = find_near(station, count, save_data_dir)
    else:
        station = pd.read_csv('%s/near_spot_document.csv'%(save_data_dir), index_col=0)

    if os.path.exists('%s/near_station_with_count_%d.csv'%(save_data_dir, count))==False:
        document = near_data_document(station, var, count, save_data_dir)
    else:
        document = pd.read_csv('%s/near_station_with_count_%d.csv'%(save_data_dir, count), index_col=0)

    if os.path.exists('%s/%s_%s-%s_count:%s.csv'%(save_data_dir, var, sdate, edate, str(count))) == False:
        dataset = var_data_maker(var, values, document, data_dir, sdate, edate, count, station, save_data_dir)
    else:
        dataset = pd.read_csv('%s/%s_%s-%s_count:%s.csv'%(save_data_dir, var, sdate, edate, str(count)), index_col=0)

    if var == 'T3H':
        dataset[dataset['Y']<-50] = np.nan
    elif var == 'REH':
        dataset[dataset['Y']<0] = np.nan

    if data_mode == 'all':
        dataset = np.array(dataset.dropna(axis=0))
        for i in range(dataset.shape[1]-1):
            dataset[:,i] = MinMax(dataset[:,i])
    
        xdataset = dataset[:,:-1]
        np.save('%/%s.npy'%(save_data_dir, var), dataset)
        ydataset = np.reshape(dataset[:,-1],(xdataset.shape[0],1))
        x_tensor = torch.from_numpy(xdataset).float()
        y_tensor = torch.from_numpy(ydataset).float()
    
        print(x_tensor.shape)
        print(y_tensor.shape)
    
        dataset = data_utils.TensorDataset(x_tensor, y_tensor)
        train_dataset, valid_dataset = random_split(dataset, [int(len(x_tensor)*0.6), len(x_tensor)-int(len(x_tensor)*0.6)])
        valid_dataset, test_dataset = random_split(valid_dataset, [int(len(valid_dataset)/2), len(valid_dataset)-int(len(valid_dataset)/2)])
        return train_dataset, valid_dataset, test_dataset, x_tensor.shape[1]


    elif data_mode == 'spot':
        dataset = dataset.dropna(axis=0)

        hand = pd.read_csv('/home/ubuntu/pkw/CDNN/DATA/minmax.csv', index_col = 0)
        drop_col = ['Year', 'Month', 'Day', 'Hour', 'Lat', 'Lon']
        dataset = dataset.drop(drop_col, axis=1)
        hand = hand.drop(drop_col, axis=1)
 
        print(dataset.shape, hand.shape)


        for i in (dataset.columns[:-1]):
            dataset[i] = MinMax(dataset[i].values, hand[i])
        index         = dataset.index.unique()
        train_index   = index[:int(len(index)*0.6)]
        valid_index   = index[int(len(index)*0.6): int(len(index)*0.8)]
        test_index    = index[int(len(index)*0.8):]
        train_dataset = dataset.loc[train_index]
        valid_dataset = dataset.loc[valid_index] 
        test_dataset  = dataset.loc[test_index] 
        train_dataset = np.array(train_dataset.values, dtype=np.float64)
        valid_dataset = np.array(valid_dataset.values, dtype=np.float64)
        test_dataset  = np.array(test_dataset.values, dtype=np.float64)

        train_x, train_y = torch.from_numpy(train_dataset[:,:-1]).float(),\
                           torch.from_numpy(np.reshape(train_dataset[:,-1], (train_dataset.shape[0], 1))).float()
        valid_x, valid_y = torch.from_numpy(valid_dataset[:,:-1]).float(),\
                           torch.from_numpy(np.reshape(valid_dataset[:,-1],( valid_dataset.shape[0], 1))).float()
        test_x, test_y   = torch.from_numpy(test_dataset[:,:-1]).float(),\
                           torch.from_numpy(np.reshape(test_dataset[:,-1], (test_dataset.shape[0], 1))).float()
        print(train_x.shape[1])
        return(data_utils.TensorDataset(train_x, train_y),
               data_utils.TensorDataset(valid_x, valid_y),
               data_utils.TensorDataset(test_x, test_y),
               train_x.shape[1])



def Operatedataset(coordinate, save_data_dir, station, count,  var, values, sdate, edate, data_dir, data_mode):
    oper_data_dir = save_data_dir + '/Operate'
    if os.path.exists('%s/half_near_spot_document.csv'%(oper_data_dir))==False:
        station = oper_find_near(station, count, oper_data_dir, coordinate)
    else:
        station = pd.read_csv('%s/half_near_spot_document.csv'%(oper_data_dir), index_col=0)

    if os.path.exists('%s/half_near_station_with_count_%d.csv'%(oper_data_dir, count))==False:
        document = near_data_document(station, var, count, oper_data_dir)
    else:
        document = pd.read_csv('%s/half_near_station_with_count_%d.csv'%(oper_data_dir, count), index_col=0)
    if os.path.exists('%s/half_oper_%s_%s-%s_count:%s.csv'%(oper_data_dir, var, sdate, edate, str(count))) == False:
        print('make dataset')
        dataset, date_list = oper_data_maker(var, values, document, data_dir, sdate, edate, count, station, oper_data_dir)
    else:
        dataset = pd.read_csv('%s/half_oper_%s_%s-%s_count:%s.csv'%(oper_data_dir, var, sdate, edate, str(count)), index_col=0)
        date_list = []
        fmt = '%Y%m%d%H'
        dt_sdate = datetime.strptime(sdate, fmt)
        dt_edate = datetime.strptime(edate, fmt)
        now = dt_sdate
    
        while now <= dt_edate:
            date_list.append(datetime.strftime(now, fmt))
            now = now+ timedelta(hours=1)
    '''
    if var == 'T3H':
        dataset[dataset['Y']<-50] = np.nan
    elif var == 'REH':
        dataset[dataset['Y']<0] = np.nan
    '''

    hand = pd.read_csv('/home/ubuntu/pkw/CDNN/DATA/minmax.csv', index_col = 0)

    drop_col = ['Year', 'Month', 'Day', 'Hour', 'Lat', 'Lon']
    dataset = dataset.drop(drop_col, axis=1)
    hand = hand.drop(drop_col, axis=1)

    col     = dataset.columns
    print(dataset.shape, hand.shape)
    for i in col:
        dataset[i] = MinMax(dataset[i], hand[i])
    dataset = np.array(dataset.dropna(axis=0), dtype = np.float64)

    xdataset = np.expand_dims(dataset, axis=0)
    #xdataset = dataset
    np.save('%s/%s.npy'%(oper_data_dir, var), dataset)
    x_tensor = torch.from_numpy(xdataset).float()
    print(x_tensor.shape)

    return x_tensor, x_tensor.shape[2], date_list


if __name__ == '__main__':

    save_data_dir('/home/ubuntu/pkw/CDNN/DATA')
    station = pd.read_csv('/home/ubuntu/pkw/DATA/OBS/QC/aws_pos.csv') 
    count = 3
    var = 'T3H'
    values = ['T3H', 'REH', 'UUU', 'VVV']
    sdate = '2020010100'
    edate = '2020010500'
