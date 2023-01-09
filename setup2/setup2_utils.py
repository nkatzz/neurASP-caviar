import sys
sys.path.append('../')

import torch
from network import Lstm
import pickle5 as pickle
import time
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from clyngor import ASP, solve
from operator import itemgetter
import re
from torchmetrics import F1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_OF_CLASSES=4



def get_asp():

	asp_program = '''

holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "next.lp".

% Distance/orientation calculations:
#include "distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).


'''

	return asp_program




def infer_on_complex(model,data_to_infer,complex_label_encoder):


	domain=["active", "inactive", "running", "walking"]
	asp_program = get_asp()


	total_acc=0
	all_preds=[]
	all_targets=[]



	with torch.no_grad():
		model.eval()


	for d in data_to_infer:

		t1=d["tensors"]["p1_tensor"]
		t2=d["tensors"]["p2_tensor"]
		atoms=d["atoms"]
		labels=d["labels"]

		p1_out,(_,_)=model(t1)
		p2_out,(_,_)=model(t2)
		p1_out=F.softmax(p1_out.view(-1,NUM_OF_CLASSES), dim=1)
		p2_out=F.softmax(p2_out.view(-1,NUM_OF_CLASSES), dim=1)

		inference_filter=""
		results=""

		_, p1_out = torch.max(p1_out, dim = 1)
		p1_out = p1_out.tolist()


		_, p2_out = torch.max(p2_out, dim = 1)
		p2_out = p2_out.tolist()


		for tp in range(0,len(labels)):

			inference_filter+="complexEvent(no_interaction,{}) :- not holdsAt(meeting(_,_),{}) , not holdsAt(moving(_,_),{}).\n".format(tp,tp,tp)


			happens_atom_p1="happensAt({}({}),{}).".format(domain[p1_out[tp]],"p1",tp)
			happens_atom_p2="happensAt({}({}),{}).".format(domain[p2_out[tp]],"p2",tp)

			results+=happens_atom_p1+" "+happens_atom_p2+"\n"

		inference_filter+="complexEvent(meeting,T) :- holdsAt(meeting(_,_),T).\ncomplexEvent(moving,T) :- holdsAt(moving(_,_),T).\n#show complexEvent/2."
		infer_program=asp_program+"\n"+atoms+"\n"+results+"\n"+inference_filter

		answers = ASP(infer_program)
		output=[]
		output_dict={}


		for answer in answers.by_predicate:
			ans_lst=str(answer['complexEvent']).replace("frozenset(","")[:-1].replace("{","[").replace("}","]").replace("(","[").replace(")","]")

			matches = re.findall("\['(\w+)', (\d+)", ans_lst)
			ans_lst = [[m[0],int(m[1])] for m in matches]

			ans_lst=list(sorted(ans_lst, key=itemgetter(1)))

			for ans in ans_lst:
				if(ans[1] not in output_dict):
					output_dict[ans[1]]=ans[0]

			for fr in output_dict:
				output.append(complex_label_encoder[output_dict[fr]])


		output[0]=output[1]

		for l in output:
			all_preds.append(l)


		for l in labels:
			all_targets.append(l)





		acc=sum(1 for x,y in zip(labels,output) if x == y) / float(len(output))

		total_acc+=acc


	total_acc = round(total_acc/len(data_to_infer),2)

	return [total_acc,all_preds,all_targets]
	


def accuracy(y_pred, target):

	_, y_pred_tags = torch.max(y_pred, dim = 1)

	correct_pred = (y_pred_tags == target).float()
	acc = correct_pred.sum() / len(correct_pred)

	acc = torch.round(acc * 100)

	return acc


def train_setup2(nn_model,criterion,optimizer,dataList_Train,dataList_Val,batch_size):

	nn_model.train()
	optimizer=optimizer

	nn_model.train()


	training_acc=0
	training_loss=0

	validation_acc=0
	validation_loss=0



	for dataIdx, sample in enumerate(dataList_Train):

		input_tensor=sample["tensor"]

		labelTensor=sample["labels"]

		out, (_, _) = nn_model(input_tensor.to(device))

		out = F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1) 
		out=torch.log(out)


		# s_acc= accuracy(out, labelTensor.view(-1))

		s_loss = criterion(out, labelTensor.view(-1))

		s_loss.backward(retain_graph=True) 


		# training_acc+=s_acc.item()
		# training_loss+=s_loss.item()



		if (dataIdx+1) % batch_size == 0:
			optimizer.step()
			optimizer.zero_grad()


	# training_acc=round(training_acc/len(dataList_Train),2)
	# training_loss=round(training_loss/len(dataList_Train),2)


	# with torch.no_grad():
	# 	nn_model.eval()


	# 	for dataIdx, sample in enumerate(dataList_Val):

	# 		input_tensor=sample["tensor"]

	# 		labelTensor=sample["labels"]

	# 		out, (_,_) = nn_model(input_tensor.to(device))

	# 		out = F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1) 
	# 		out=torch.log(out)

	# 		s_acc= accuracy(out, labelTensor.view(-1))

	# 		s_loss = criterion(out, labelTensor.view(-1))


	# 		validation_acc+=s_acc.item()
	# 		validation_loss+=s_loss.item()


	# validation_acc=round(validation_acc/len(dataList_Val),2)
	# validation_loss=round(validation_loss/len(dataList_Val),2)



	return [training_acc,training_loss,validation_acc,validation_loss]


def infer_on_simple(nn_model,dataList):

	acc_on_simple=0
	loss_on_simple=0

	all_preds=[]
	all_ground=[]

	with torch.no_grad():
		nn_model.eval()

	for dataIdx, sample in enumerate(dataList):

		input_tensor=sample["tensor"]
		labelTensor=sample["labels"]

		person_out,(_,_)=nn_model(input_tensor)

		person_out=F.softmax(person_out.view(-1,NUM_OF_CLASSES), dim=1)

		train_out=torch.log(person_out)
		_, y_pred_tags = torch.max(train_out, dim = 1)
		y_pred_tags=torch.flatten(y_pred_tags).tolist()
		y_ground=torch.flatten(labelTensor).tolist()

		for yp in y_pred_tags:
		    all_preds.append(yp)

		for yg in y_ground:
		    all_ground.append(yg)


	return [
	    all_preds,
	    all_ground
	]



def acc_on_simple(nn_model,dataList):

	all_preds=[]
	all_targets=[]

	micro_f1=None
	m_f1_score = 0 
	acc_on_simple=0

	with torch.no_grad():
		nn_model.eval()



	for dataIdx, sample in enumerate(dataList):

		input_tensor=sample["tensor"]
		labelTensor=sample["labels"]

		out,(_,_)=nn_model(input_tensor)

		out=F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1)
		
		_, y_pred_tags = torch.max(out, dim = 1)
		# all_preds.append(y_pred_tags)

		for yp in y_pred_tags.tolist():
			all_preds.append(yp)




		# mf1s=mircro_f1(torch.tensor(y_pred_tags.tolist()),torch.tensor(torch.flatten(labelTensor).tolist()))
		# all_preds.append(x for x in y_pred_tags.tolist())

		y_ground=torch.flatten(labelTensor).tolist()
		# all_targets.append(y_ground)
		for yg in y_ground:
			all_targets.append(yg)

		# print(all_preds)
		# input()
		# all_targets.append(x for x in y_ground)


		out=torch.log(out)
		s_acc= accuracy(out, labelTensor.view(-1))
		acc_on_simple+=s_acc.item()
		# m_f1_score+=mf1s.item()


	# mircro_f1=None
	# if(2 in all_targets):
	# print(len(all_targets),len(all_preds))
	# 
	# else:
	# micro_f1 = F1(num_classes=3)	

	if(2 in y_ground):
		micro_f1 = F1(num_classes=4,average='macro')
	else:
		for i in range(len(all_targets)):
			if(all_targets[i]==3):
				all_targets[i]=2

			if(all_preds[i]==3):
				all_preds[i]=2

		micro_f1 = F1(num_classes=3,average='macro')	


	m_f1_score=micro_f1(torch.tensor(all_preds),torch.tensor(all_targets))	
	acc_on_simple=round(acc_on_simple/len(dataList),2)
	# m_f1_score=round(m_f1_score/len(dataList),2)

	return acc_on_simple , round(m_f1_score.item(),2)




def main():


	ws=24

	with open('../data/data_dict.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)


	domain=["active", "inactive", "running", "walking"]
	simple_label_encoder=data_dict["simple_label_encoder"]
	complex_label_encoder=data_dict["complex_label_encoder"]


	dprogram,asp_program = get_asp()

	setup="setup_2"

	training_data=data_dict[setup]["training"]
	testing_data=data_dict[setup]["testing"]


	train_inference_for_complex_events=[]


	train_datalist={
	    "tensorList":[],
	    "labelList":[]
	}



	for s in training_data["by_person"]:

	    if (len(s['models'])>0):
	        train_datalist["tensorList"].append(s["tensor"])
	        train_datalist["labelList"].append(s["labels"])



	for s in training_data["by_pair"]:

	    train_inference_for_complex_events.append(
	        {
	            "tensors":s["tensors"],
	            "labels":s["labels"],
	            "atoms_for_inference":s["atoms_for_inference"]
	        }
	    )


	


	testing_inference_for_complex_events=[]

	testing_datalist={
	    "tensorList":[],
	    "labelList":[]
	}


	for s in testing_data["by_person"]:

	    testing_datalist["tensorList"].append(s["tensor"])
	    testing_datalist["labelList"].append(s["labels"])


	for s in testing_data["by_pair"]:

	    testing_inference_for_complex_events.append(
	        {
	            "tensors":s["tensors"],
	            "labels":s["labels"],
	            "atoms_for_inference":s["atoms_for_inference"]
	        }
	    )


	nn_model = Lstm()
	nn_model.to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer=torch.optim.Adam(nn_model.parameters(), lr=0.0001)
	batch_size=32


	Best_acc=0
	keep_model=None

	idx2class = {simple_label_encoder[k]:k for k in simple_label_encoder.keys()}


	for i in range(0,3):

		nn_model=train_setup2(nn_model,criterion,optimizer,train_datalist,domain,batch_size,device)
		[all_preds_simple,all_targets_simple,acc,_]=infer_on_simple(nn_model,train_datalist,domain)

		confusion_matrix_df = pd.DataFrame(confusion_matrix(all_targets_simple, all_preds_simple)).rename(
		    columns=idx2class, 
		    index=idx2class
		)

		print(confusion_matrix_df)
		print(classification_report(all_targets_simple, all_preds_simple,zero_division=0))

		[all_preds_simple,all_targets_simple,vacc,_]=infer_on_simple(nn_model,testing_datalist,domain)


		total_acc,all_preds,all_targets=infer_on_complex(nn_model,asp_program,train_inference_for_complex_events,complex_label_encoder,ws,domain)


		if(Best_acc<acc):
			torch.save(nn_model,"n.pt")
			bb="Best Epoch {} tr acc {} val acc {}".format(i+1,acc,vacc)
			Best_acc=acc


		print("Epoch {} tr acc {} val acc {}".format(i+1,acc,vacc))	



	loaded_model = torch.load("./n.pt")
	print(bb)
	[all_preds_simple,all_targets_simple,acc,_]=infer_on_simple(loaded_model,testing_datalist,domain)

	if(simple_label_encoder["running"] not in all_targets_simple and simple_label_encoder["running"] not in all_preds_simple):
	    encoder_for_print={"active":0,"inactive":1,"walking":2}
	    idx2class = {encoder_for_print[k]:k for k in encoder_for_print }


	confusion_matrix_df = pd.DataFrame(confusion_matrix(all_targets_simple, all_preds_simple)).rename(
	    columns=idx2class, 
	    index=idx2class
	)

	print(confusion_matrix_df)
	print(classification_report(all_targets_simple, all_preds_simple,zero_division=0))


if __name__ == '__main__':
	main()