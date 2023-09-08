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

'''
Use gpu if present
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
NeurAsp program where two time windows of two persons are provided as input.
nn(simpleEvent(24,S), [active, inactive, running, walking]) :- window(S) means
that the network expects input of 24 time-steps and it will output classifications
on simpleEvents between the classes active,inactive,running,walking 
'''
dprogram = '''
window(p1). window(p2).

nn(simpleEvent(24,S), [active, inactive, running, walking]) :- window(S).
'''


'''
Load the dictionary that contains the data from all three folds
'''
with open('../data/3-fold_dict.pkl', 'rb') as handle:
	data_dict = pickle.load(handle)

'''
The loaded dictionary includes also the label encoders
'''

complex_label_encoder=data_dict["complex_label_encoder"]
simple_label_encoder=data_dict["simple_label_encoder"]

'''
extract data from each fold from the generated dictionary
'''
fold_dict={
	"fold1":data_dict["fold1"],
	"fold2":data_dict["fold2"],
	"fold3":data_dict["fold3"]
}

'''
Create an empty dictionary in order to insert the data
in an organized matter.
For each fold :
-----train data----------
simple_train: this list will contain the tensors that contain the features and their labels on simple events -----> [tensor,labels]
complex_train: this list will contain the pairs of tensors of the involved persons and their labels on complex events.
Also the stable asp models are included as well as the auxialiary atoms needed for the asp inference. -----> [[tensor_p1,tensor_p2],labels,aux_atoms,asp_models] 

-----test data----------
organized in similar way as the train data.
'''
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

'''
Populate the fold dictionary as described above
'''
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

'''
Neural Network hyperparams
'''
batchSize=4
epochs=10

lr=0.0000015
model_count=5
alpha=0
models_paths=[]
'''
For each fold
'''
for f in folds:

	m = Lstm()#create an lstm model instance

	nnMapping = {'simpleEvent': m} #create the name mapping of the neural network as NeurAsp demands
	optimizers = {'simpleEvent': torch.optim.Adam(m.parameters(), lr=lr)} #use Adam optimizer

	'''
	This is where we use the NeurAsp Framework. In order to create an neurAsp object we pass the following parameters.
	@param dprogram: the NeurAsp program mentioned in line 23
	@param nnMapping: the mapping of the network with its name (simpleEvent)
	@param optimizers: the optimizer (Adam)

	'''
	NeurASPobj = NeurASP(dprogram, nnMapping, optimizers,gpu=True)
	'''
	Each fold to be tested needs an instance of the trained neural network
	so we create seperate names for each model to be saved in the folder neural_models
	'''
	model_file_name="./neural_models/LSTM_setup3-{}-asp_models={}.pt".format(f,model_count)
	models_paths.append(model_file_name)

	'''
	a simple dictionary to store statistics of the training procedure
	'''
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

	'''
	create a stats dict for every fold
	'''
	stats[f]=fold_stats

	'''
	get the training data in the form of [[tensor_p1,tensor_p2],labels,aux_atoms,asp_models]
	'''
	trainData=folds[f]["complex_train"]

	startTime = time.time()
	best_record=""
	best_epoch=0
	'''
	Training loop
	'''
	for epoch in range(epochs):
		random.shuffle(trainData)
		''''
		This is where the neurAsp training is taking place. For more information refer to neurasp python script.
		@param trainData: the training data on simple events alongside with their pre generated asp models
		@param alpha: This defines the impact of the semantic loss (left with its default value as digit addition)
		@model_count: This defines how many stable models to use in the training (default is 5, seems to get the best results)
		'''
		model=NeurASPobj.train_setup3(trainData,alpha,batchSize,model_count)

		print("-------------------------- Epoch {} / Fold {} ---------------------------------------".format(epoch+1,f))
		'''
		Results of training samples on simple events such as accuracy and f1 score
		'''
		 
		[all_preds,all_targets] = setup3_utils.infer_on_simple(model,folds[f]["simple_train"])
		train_simple_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		train_acc_simple = train_simple_ev_scores['accuracy']
		train_f1_simple = train_simple_ev_scores['macro avg']['f1-score']

		'''
		Results of testing samples on simple events such as accuracy and f1 score
		'''
		[all_preds,all_targets] = setup3_utils.infer_on_simple(model,folds[f]["simple_test"])
		simple_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		test_acc_simple = simple_ev_scores['accuracy']
		test_f1_simple = simple_ev_scores['macro avg']['f1-score']
		
		print("Simple Events Train acc {:.2f} / Test acc {:.2f}".format(train_acc_simple,test_acc_simple))
		print("Simple Events Train f1 {:.2f} / Test f1 {:.2f}".format(train_f1_simple,test_f1_simple))

		'''
		Get the best performance
		'''

		if(stats[f]["best_test_f1_simple"] < test_f1_simple):
		
			stats[f]["best_test_f1_simple"] = test_f1_simple
			
			'''
			Inference on complex events by using the infer_on_complex method. For more information about this method refer to setup3_utils.py script
			@param model: the instance of the so far trained model
			@param folds[f]["complex_test"]: the data [[tensor_p1,tensor_p2],labels,aux_atoms,asp_models] , actually this method uses the aux atoms
			@param complex_label_encoder: the label encoder on complex events
			'''
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
	'''
	Store the training time of each fold
	'''
	stats[f]["total_time"]=fold_time

'''
Store the stats in a dictionary
'''
with open('./stats.pkl'.format(model_count), 'wb') as handle:
	pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
