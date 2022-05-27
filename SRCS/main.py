from argparse import ArgumentParser
from run import run
import warnings
import os


def set_argument_parser():
    parser = ArgumentParser(description='CDNN')
    parser.add_argument('--model', type = str, default = 'DNN', help = '[DNN, LSTM]')
    #parser.add_argument('--mode', type = str, default = 'Train', help = '[Train, Test, Operate]')
    parser.add_argument('--mode', type = str, default = 'Operate', help = '[Train, Test, Operate]')
    parser.add_argument('--var', type = str, default = 'T3H', help = '[T3H, REH, UUU, VVV]')
    parser.add_argument('--values', type = list, default = ['T3H', 'REH', 'UUU', 'VVV'])
    parser.add_argument('--count', type = int, default = 1)
    parser.add_argument('--sdate', type = str, default = '2020010100')
    #parser.add_argument('--edate', type = str, default = '2020123123')
    parser.add_argument('--edate', type = str, default = '2020010102')
    parser.add_argument('--save_data_dir', type = str, default = '/home/ubuntu/pkw/CDNN/DATA')
    parser.add_argument('--data_dir', type = str, default = '/home/ubuntu/pkw/DATA/OBS/QC')
    parser.add_argument('--station_dir', type = str, default = '/home/ubuntu/pkw/DATA/OBS/QC/new_aws.csv')
    parser.add_argument('--model_dir', type = str, default = '/home/ubuntu/pkw/CDNN/DAIN')
    parser.add_argument('--logs_dir', type = str, default = '/home/ubuntu/pkw/CDNN/LOGS')
    parser.add_argument('--data_mode', type = str, default = 'spot', help = ['all', 'spot'])
    parser.add_argument('--oper_out_dir', type = str, default = '/home/ubuntu/pkw/CDNN/OPER/%s')
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--patience', type = int, default = 40)
    parser.add_argument('--coordinate', type = str, default = '/home/ubuntu/pkw/DATA/ETC/1km_grid_elev.npy')

    return parser


def main():
    import time
    start = time.time()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"


    parser = set_argument_parser()
    args   = parser.parse_args()
    run(args)
    print("time:", time.time() - start)
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    main()


