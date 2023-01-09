import sys
sys.path.append('../')
import torch
from network import Lstm
import pickle5 as pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import setup1_utils
import torch
from torchmetrics import F1



NUM_OF_CLASSES=3
ws=24

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('../data/3-fold_dict.pkl', 'rb') as handle:
	data_dict = pickle.load(handle)


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



stats={}

batchSize=4
epochs=1000
lr=0.0000015
models_paths=[]

for f in folds:


	model_file_name="./neural_models/LSTM_setup1-{}.pt".format(f)


	models_paths.append(model_file_name)

	acc_stats_file_complex="./plots/LSTM_setup1-{}_acc_complex.csv".format(f)
	loss_stats_file_complex="./plots/LSTM_setup1-{}_loss_complex.csv".format(f)



	fold_stats={
		"best_f1":0,
		"total_time":0,
		"infer_time":0,
		"epoch_best":0,
		"macro_f1_record":[],
		"acc_stats":{
			"train":[],
			"test":[]
		},
		"loss_stats":{
			"train":[],
			"test":[]
		}
	}


	stats[f]=fold_stats


	trainData=folds[f]["train"]
	testData=folds[f]["test"]

	model = Lstm()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	model.to(device)

	total_infer_time=0
	startTime = time.time()

	for epoch in range(epochs):

		setup1_utils.train(
			model,
			trainData,
			# testData,
			optimizer,
			criterion,
			batchSize
		)

		print("-------------------------- Epoch {} / Fold {} ---------------------------------------".format(epoch+1,f))
		start_infer_time = time.time()
		all_preds,all_targets=setup1_utils.infer(model,testData)




		complex_ev_scores = classification_report(all_targets, all_preds,zero_division=0,output_dict=True)
		complex_macro_f1 = round(complex_ev_scores['macro avg']['f1-score'],2)
		stats[f]["macro_f1_record"].append(complex_macro_f1)
		complex_acc = round(complex_ev_scores['accuracy'],2)
		print("Complex Events at epoch {}:  Val Acc = {} / macro F1 score {}".format(epoch+1,complex_acc,complex_macro_f1))
		infer_time = (time.time() - start_infer_time)
		total_infer_time+=infer_time

		if(stats[f]["best_f1"] < complex_macro_f1):
			stats[f]["best_f1"] = complex_macro_f1
			stats[f]["best_epoch"] = epoch +1
			torch.save(model,model_file_name)


		print("Current Best f1: {} at epoch {}".format(stats[f]["best_f1"],stats[f]["best_epoch"]))




		# print("Complex Events: Train Acc = {} Train Loss = {} | Val Acc = {} Val Loss = {}".format(
		# 	training_acc,
		# 	training_loss,
		# 	validation_acc,
		# 	validation_loss
		# 	)
		# )

		# stats[f]["acc_stats"]["train"].append(training_acc)
		# stats[f]["acc_stats"]["test"].append(validation_acc)


		# stats[f]["loss_stats"]["train"].append(training_loss)
		# stats[f]["loss_stats"]["test"].append(validation_loss)



		# if(stats[f]["Best_val_acc"] < validation_acc):
		# 	stats[f]["Best_val_acc"] = validation_acc
		# 	torch.save(model,model_file_name)


	fold_time=int((time.time() - startTime)/60)


	print("Fold {} total time from beginning: {} minutes , Best f1 : {} at epoch {}".format(f, fold_time,stats[f]["best_f1"],stats[f]["best_epoch"]))
	print("--------------------------------------------------------------------------")

	stats[f]["total_time"]=fold_time
	stats[f]["infer_time"]=total_infer_time



	# print("Fold {} total time from beginning: {} minutes".format(f, epoch_time))
	# print("--------------------------------------------------------------------------")


	# train_val_acc_df_complex = pd.DataFrame.from_dict(stats[f]["acc_stats"]).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
	# train_val_loss_df_complex = pd.DataFrame.from_dict(stats[f]["loss_stats"]).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})# Plot the dataframes

	# train_val_acc_df_complex.to_csv(acc_stats_file_complex)
	# train_val_loss_df_complex.to_csv(loss_stats_file_complex)
	# stats[f]["total_time"]=epoch_time




# for m in models_paths:
# 	f=m.split("-")[1].replace(".pt","")
# 	print("{} total time {}".format(f,stats[f]["total_time"]))
# 	loaded_model = torch.load(model_file_name)
# 	[all_preds,all_targets]=setup1_utils.infer(loaded_model,folds[f]["test"])
# 	idx2class = {complex_label_encoder[k]:k for k in complex_label_encoder.keys()}
# 	confusion_matrix_df = pd.DataFrame(confusion_matrix(all_targets, all_preds)).rename(columns=idx2class, index=idx2class)
# 	print(classification_report(all_targets, all_preds,zero_division=0))
# 	print(confusion_matrix_df) 




with open('./stats.pkl', 'wb') as handle:
	pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)