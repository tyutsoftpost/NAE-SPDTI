import pandas as pd
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, \
    precision_score
from models import binary_cross_entropy, cross_entropy_logits
from models import BINDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 1
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="BINDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='human')
parser.add_argument('--split', default='random1', type=str, metavar='S', help="split task",
                    choices=['random', 'random1', 'random2', 'random3', 'random4'])
args = parser.parse_args()

torch.cuda.empty_cache()
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
cfg = get_cfg_defaults()
set_seed(cfg.SOLVER.SEED)
print("start...")
print(f"dataset:{args.data}")
print(f"Hyperparameters: {dict(cfg)}")
print(f"Running on: {device}", end="\n\n")
dataFolder = f'../datasets/{args.data}'
dataFolder = os.path.join(dataFolder, str(args.split))
test_path = os.path.join(dataFolder, "test.csv")
df_test = pd.read_csv(test_path)
test_dataset = DTIDataset(df_test.index.values, df_test)
print(f'train_dataset:{len(test_dataset)}')
params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
          'drop_last': True, 'collate_fn': graph_collate_func}
params['shuffle'] = False
params['drop_last'] = False
test_generator = DataLoader(test_dataset, **params)

test_loss = 0
y_label, y_pred = [], []
data_loader = test_generator
num_batches = len(data_loader)
#
model = BINDTI(device=device, **cfg).to(device=device)
model.load_state_dict(torch.load('../output/result/human/random1/best_model_epoch_99.pth'))
df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}
#
with torch.no_grad():
    model.eval()
    for i, (v_d, v_p, labels, t) in enumerate(data_loader):
        v_d, v_p, labels = v_d.to(device), v_p.to(device), labels.float().to(device)
        v_d, v_p, f, score = model(v_d, v_p)

        n, loss = binary_cross_entropy(score, labels)
        test_loss += loss.item()
        y_label = y_label + labels.to("cpu").tolist()
        y_pred = y_pred + n.to("cpu").tolist()

        df['drug'] = df['drug'] + v_d.to('cpu').tolist()
        df['protein'] = df['protein'] + v_p.to('cpu').tolist()

auroc = roc_auc_score(y_label, y_pred)
auprc = average_precision_score(y_label, y_pred)
test_loss = test_loss / num_batches

fpr, tpr, thresholds = roc_curve(y_label, y_pred)
prec, recall, _ = precision_recall_curve(y_label, y_pred)
try:
    precision = tpr / (tpr + fpr)
except RuntimeError:
    raise ('RuntimeError: the divide==0')
f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
thred_optim = thresholds[5:][np.argmax(f1[5:])]
y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
cm1 = confusion_matrix(y_label, y_pred_s)
accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

precision1 = precision_score(y_label, y_pred_s)
df['y_label'] = y_label
df['y_pred'] = y_pred
data = pd.DataFrame(df)
data.to_csv('../output/result/human/human_visualization.csv', index=False)
