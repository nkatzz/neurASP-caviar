import sys
sys.path.append('../')
import torch
from network import Lstm
import pickle5 as pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import setup1_utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('../data/3-fold_dict.pkl', 'rb') as handle:
	data_dict = pickle.load(handle)


with open('stats.pkl', 'rb') as handle:
	stats = pickle.load(handle)



complex_label_encoder=data_dict["complex_label_encoder"]

folds={
	"fold1":{
		"train":[{"tensor":s["concat_tensor"],"labels":s["complex_labels"]} for s in data_dict["fold1"]["train"]],
		"test":[{"tensor":s["concat_tensor"],"labels":s["complex_labels"]} for s in data_dict["fold1"]["test"]]
	},
	"fold2":{
		"train":[{"tensor":s["concat_tensor"],"labels":s["complex_labels"]} for s in data_dict["fold2"]["train"]],
		"test":[{"tensor":s["concat_tensor"],"labels":s["complex_labels"]} for s in data_dict["fold2"]["test"]]
	},

	"fold3":{
		"train":[{"tensor":s["concat_tensor"],"labels":s["complex_labels"]} for s in data_dict["fold3"]["train"]],
		"test":[{"tensor":s["concat_tensor"],"labels":s["complex_labels"]} for s in data_dict["fold3"]["test"]]
	}
}


for f in folds:

	print("-------FOLD: {} TRAIN TIME: {} minutes-----------------".format(f,stats[f]["total_time"]))
	loaded_model = torch.load("neural_models/LSTM_setup1-{}.pt".format(f))
	[all_preds,all_targets]=setup1_utils.infer(loaded_model,folds[f]["test"])
	idx2class = {complex_label_encoder[k]:k for k in complex_label_encoder.keys()}
	confusion_matrix_df = pd.DataFrame(confusion_matrix(all_targets, all_preds)).rename(columns=idx2class, index=idx2class)
	print(classification_report(all_targets, all_preds,zero_division=0))
	print(confusion_matrix_df)

	print("--------------------------------------------------------".format(f)) 