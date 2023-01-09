import sys
sys.path.append('../')
import torch
from network import Lstm
import pickle5 as pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import setup2_utils
import random

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



stats={}

batchSize=8
epochs=1000
lr=0.0000015

models_paths=[]

for f in folds:

	model_file_name="./neural_models/LSTM_setup2-{}.pt".format(f)
	models_paths.append(model_file_name)

	acc_stats_file_complex="./plots/LSTM_setup2-{}_acc_simple.csv".format(f)
	loss_stats_file_complex="./plots/LSTM_setup2-{}_loss_simple.csv".format(f)



	fold_stats={
		"best_test_acc_simple":0,
		"best_train_acc_simple":0,
		"best_test_acc_complex":0,
		"best_train_acc_complex":0,
		"best_test_f1_complex":0,
		"best_train_f1_complex":0,
		"best_test_f1_simple":0,
		"best_train_f1_simple":0,
		"total_time":0,
		"infer_time":0,
		"best_epoch":0
	}

	stats[f]=fold_stats


	trainData=folds[f]["simple_train"]
	testData=folds[f]["simple_test"]

	model = Lstm()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	model.to(device)

	total_infer_time=0
	startTime = time.time()
	best_epoch=0
	for epoch in range(epochs):
		random.shuffle(trainData)
		[training_acc,training_loss,validation_acc,validation_loss]=setup2_utils.train_setup2(
			model,
			criterion,
			optimizer,
			trainData,
			testData,
			batchSize
		)


		print("-------------------------- Epoch {} / Fold {} ---------------------------------------".format(epoch+1,f))
		[all_preds,all_targets] = setup2_utils.infer_on_simple(model,folds[f]["simple_train"])
		train_simple_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		train_acc_simple = train_simple_ev_scores['accuracy']
		train_f1_simple = train_simple_ev_scores['macro avg']['f1-score']


		[all_preds,all_targets] = setup2_utils.infer_on_simple(model,folds[f]["simple_test"])
		simple_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		test_acc_simple = simple_ev_scores['accuracy']
		test_f1_simple = simple_ev_scores['macro avg']['f1-score']

		print("Simple Events Train acc {:.2f} / Test acc {:.2f}".format(train_acc_simple,test_acc_simple))
		print("Simple Events Train f1 {:.2f} / Test f1 {:.2f}".format(train_f1_simple,test_f1_simple))

		if(stats[f]["best_test_f1_simple"] < test_f1_simple):
		
			stats[f]["best_test_f1_simple"] = test_f1_simple
	
			[_,all_preds,all_targets]=setup2_utils.infer_on_complex(model,folds[f]["complex_test"],complex_label_encoder)
			complex_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
			test_f1_complex = complex_ev_scores['macro avg']['f1-score']
			if(stats[f]["best_test_f1_complex"] < test_f1_complex):
				best_epoch=epoch+1
				stats[f]["best_epoch"] = best_epoch
				stats[f]["best_test_f1_complex"] = test_f1_complex
				torch.save(model,model_file_name)
				best_record="Best Complex Events f1 {:.2f}".format(test_f1_complex)

		print(best_record)


	fold_time=int((time.time() - startTime)/60)

	print("--------------------------------------------------------------------------")

	stats[f]["total_time"]=fold_time
	stats[f]["infer_time"]=total_infer_time


with open('./stats.pkl', 'wb') as handle:
	pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
