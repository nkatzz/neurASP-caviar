import sys
sys.path.append('../')
import torch
from network import Lstm
import torch.nn.functional as F
from torchmetrics import F1



NUM_OF_CLASSES=3

def accuracy(y_pred, target):
    
    _, y_pred_tags = torch.max(y_pred, dim = 1)
    

    correct_pred = (y_pred_tags == target).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    return acc



def infer(model,dataList):

	all_preds=[]
	all_ground=[]


	with torch.no_grad():
		model.eval()

		for dataIdx, data  in enumerate(dataList):

			inputTensor = data["tensor"]
			labelTensor = data["labels"]

			
			out, (_, _)=model(inputTensor)

			out = F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1) 

			out=torch.log(out)

			_, y_pred_tags = torch.max(out, dim = 1)


			y_pred_tags=torch.flatten(y_pred_tags).tolist()
			y_ground=torch.flatten(labelTensor).tolist()

			for yp in y_pred_tags:
				all_preds.append(yp)

			for yg in y_ground:
				all_ground.append(yg)


	return [all_preds,all_ground]

def train(model,dataList_Train,optimizer,criterion,batchSize):

	model.train()

	training_acc=0
	training_loss=0

	validation_acc=0
	validation_loss=0


	for dataIdx, data  in enumerate(dataList_Train):


		inputTensor = data["tensor"]

		labelTensor = data["labels"]

		out , (_,_)=model(inputTensor)
		out = F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1) 
		out=torch.log(out)


		# s_acc= accuracy(out, labelTensor.view(-1))
		s_loss = criterion(out, labelTensor.view(-1))


		# training_acc+=s_acc.item()
		# training_loss+=s_loss.item()


		s_loss.backward(retain_graph=True) 



		if (dataIdx+1) % batchSize == 0:
			optimizer.step()
			optimizer.zero_grad()


	# training_acc=round(training_acc/len(dataList_Train),2)
	# training_loss=round(training_loss/len(dataList_Train),2)


	# with torch.no_grad():
	# 	model.eval()

	# 	for dataIdx, data  in enumerate(dataList_Val):

	# 		inputTensor = data["tensor"]
	# 		labelTensor = data["labels"]

	# 		out,(_,_)=model(inputTensor)
	# 		out = F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1) 
	# 		out=torch.log(out)

	# 		s_acc= accuracy(out, labelTensor.long().view(-1))
	# 		s_loss = criterion(out, labelTensor.long().view(-1))


	# 		validation_acc+=s_acc.item()
	# 		validation_loss+=s_loss.item()


	# 	validation_acc=round(validation_acc/len(dataList_Val),2)
	# 	validation_loss=round(validation_loss/len(dataList_Val),2)



	# return [training_acc,training_loss,validation_acc,validation_loss]
