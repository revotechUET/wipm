import os
import argparse
import pandas as pd
import pickle
from service.ml_toolkit.anfis import Anfis
from service.ml_toolkit import himpe
from service.config import config_object


headers = ['GR', 'NPHI', 'RHOB', 'DT', 'VCL', 'PHIE', 'PERM_CORE']

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='anfis', type=str,
                    help='I can put himpe or anfis or ...')
parser.add_argument('--data', default='train.csv', type=str,
                    help='Train data file with csv type')
parser.add_argument('--cluster', default=6, type=int,
                        help='Number of cluster for CMean algorithm')
args = parser.parse_args()

def train_anfis(datafile):
    df = pd.read_csv(datafile, header=0, index_col=None)[headers]
    model = Anfis()
    model.fit(df.values[:,:-1], df.values[:,-1])
    with open(os.path.join(config_object['path']['anfis'], 'anfis'), 'wb') as f:
        pickle.dump(model, f)
    print('Complete')

def train_himpe(ftrain, n_clusters):
    fmodels = os.path.join(config_object['path']['himpe'], 'himpe')
    himpe.create_models(ftrain=ftrain, fmodels=fmodels, fval=None, n_clusters=n_clusters)
    print('Complete')

if __name__ == '__main__':
    data = args.data
    n_clusters = args.cluster
    if args.type == 'anfis':
        train_anfis(data)
    elif args.type == 'himpe':
        train_himpe(data, n_clusters)
    else:
        print('No model support')
