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
from neurasp import NeurASP
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dprogram = '''
window(p1). window(p2).

nn(simpleEvent(24,S), [active, inactive, running, walking]) :- window(S).
'''



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

			folds[f]["complex_train"].append({"tensors":{"p1":s["p1_tensor"],"p2":s["p2_tensor"]},"labels":s["complex_labels_as_list"],"atoms":s["atoms"],"tag":s,"models":s["models"]})

	for s in fold_dict[f]["test"]:

			folds[f]["simple_test"].append({"tensor":s["p1_tensor"],"labels":s["p1_labels"]})
			folds[f]["simple_test"].append({"tensor":s["p2_tensor"],"labels":s["p2_labels"]})

			folds[f]["complex_test"].append({"tensors":{"p1_tensor":s["p1_tensor"],"p2_tensor":s["p2_tensor"]},"labels":s["complex_labels_as_list"],"atoms":s["atoms"]})



stats={}

batchSize=4
epochs=1000

lr=0.0000015
model_count=5
alpha=0


models_paths=[]


for f in folds:

	m = Lstm()



	nnMapping = {'simpleEvent': m}
	optimizers = {'simpleEvent': torch.optim.Adam(m.parameters(), lr=lr)}

	NeurASPobj = NeurASP(dprogram, nnMapping, optimizers,gpu=True)
	model_file_name="./neural_models/LSTM_setup3-{}-asp_models={}.pt".format(f,model_count)
	models_paths.append(model_file_name)

	
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


	trainData=folds[f]["complex_train"]
	testData=folds[f]["simple_test"]


	startTime = time.time()
	best_record=""
	best_epoch=0
	for epoch in range(epochs):
		random.shuffle(trainData)
		model=NeurASPobj.train_setup3(trainData,alpha,batchSize,model_count)

		print("-------------------------- Epoch {} / Fold {} ---------------------------------------".format(epoch+1,f))
		[all_preds,all_targets] = setup3_utils.infer_on_simple(model,folds[f]["simple_train"])
		train_simple_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		train_acc_simple = train_simple_ev_scores['accuracy']
		train_f1_simple = train_simple_ev_scores['macro avg']['f1-score']


		[all_preds,all_targets] = setup3_utils.infer_on_simple(model,folds[f]["simple_test"])
		simple_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		test_acc_simple = simple_ev_scores['accuracy']
		test_f1_simple = simple_ev_scores['macro avg']['f1-score']
		
		print("Simple Events Train acc {:.2f} / Test acc {:.2f}".format(train_acc_simple,test_acc_simple))
		print("Simple Events Train f1 {:.2f} / Test f1 {:.2f}".format(train_f1_simple,test_f1_simple))

		if(stats[f]["best_test_f1_simple"] < test_f1_simple):
		
			stats[f]["best_test_f1_simple"] = test_f1_simple
			
			[_,all_preds,all_targets]=setup3_utils.infer_on_complex(model,folds[f]["complex_test"],complex_label_encoder)
			complex_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
			test_f1_complex = complex_ev_scores['macro avg']['f1-score']
			if(stats[f]["best_test_f1_complex"] < test_f1_complex):
				best_epoch=epoch+1
				stats[f]["best_epoch"] = best_epoch
				stats[f]["best_test_f1_complex"] = test_f1_complex
				torch.save(model,model_file_name)
				best_record="Best complex events f1 {:.2f}".format(test_f1_complex)

		print(best_record)


	fold_time=int((time.time() - startTime)/60)

	print("--------------------------------------------------------------------------")

	stats[f]["total_time"]=fold_time

with open('./stats.pkl'.format(model_count), 'wb') as handle:
	pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)