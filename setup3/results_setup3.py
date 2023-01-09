import sys
sys.path.append('../')
import torch
from network import Lstm
import pickle5 as pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import setup3_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('../data/3-fold_dict.pkl', 'rb') as handle:
	data_dict = pickle.load(handle)

complex_label_encoder=data_dict["complex_label_encoder"]
simple_label_encoder=data_dict["simple_label_encoder"]


fold_dict={
	"fold1":data_dict["fold1"],
	"fold2":data_dict["fold2"],
	"fold3":data_dict["fold3"]
}


folds={
	"fold1":{
		"simple_train":[],
		"simple_test":[],
		"complex_train":[],
		"complex_test":[]
	},
	"fold2":{
		"simple_train":[],
		"simple_test":[],
		"complex_train":[],
		"complex_test":[]
	},
	"fold3":{
		"simple_train":[],
		"simple_test":[],
		"complex_train":[],
		"complex_test":[]
	}
}


for f in fold_dict:
	for s in fold_dict[f]["train"]:
		if(len(s["models"])>0):

			folds[f]["simple_train"].append({"tensor":s["p1_tensor"],"labels":s["p1_labels"]})
			folds[f]["simple_train"].append({"tensor":s["p2_tensor"],"labels":s["p2_labels"]})


		folds[f]["complex_train"].append({"tensors":{"p1_tensor":s["p1_tensor"],"p2_tensor":s["p2_tensor"]},"labels":s["complex_labels_as_list"],"atoms":s["atoms"]})

	for s in fold_dict[f]["test"]:

		folds[f]["simple_test"].append({"tensor":s["p1_tensor"],"labels":s["p1_labels"]})
		folds[f]["simple_test"].append({"tensor":s["p2_tensor"],"labels":s["p2_labels"]})

		folds[f]["complex_test"].append({"tensors":{"p1_tensor":s["p1_tensor"],"p2_tensor":s["p2_tensor"]},"labels":s["complex_labels_as_list"],"atoms":s["atoms"]})



model_count=5

with open('stats.pkl'.format(model_count), 'rb') as handle:
	stats = pickle.load(handle)



all_f1=0

for f in folds:

	print("-------FOLD: {} TRAIN TIME: {} minutes".format(f,stats[f]["total_time"]))
	loaded_model = torch.load("neural_models/LSTM_setup3-{}-asp_models={}.pt".format(f,model_count))
	[_,all_preds,all_targets]=setup3_utils.infer_on_complex(loaded_model,folds[f]["complex_test"],complex_label_encoder)
	idx2class = {complex_label_encoder[k]:k for k in complex_label_encoder.keys()}
	confusion_matrix_df = pd.DataFrame(confusion_matrix(all_targets, all_preds)).rename(columns=idx2class, index=idx2class)

	complex_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
	cur_f1 = complex_ev_scores['macro avg']['f1-score']
	all_f1+=cur_f1

	print(classification_report(all_targets, all_preds,zero_division=0))
	print(confusion_matrix_df)

	print("--------------------------------------------------------".format(f)) 

print("Overall f1 score {:.2f}".format(all_f1/3))