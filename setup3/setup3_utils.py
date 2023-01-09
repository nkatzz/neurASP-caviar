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
import torch
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


def infer_on_simple(nn_model,dataList):

	all_preds=[]
	all_ground=[]

	with torch.no_grad():
		nn_model.eval()

	for dataIdx, sample in enumerate(dataList):

		input_tensor=sample["tensor"]
		labelTensor=sample["labels"]

		out,(_,_)=nn_model(input_tensor)

		out=F.softmax(out.view(-1,NUM_OF_CLASSES), dim=1)

		_, y_pred_tags = torch.max(out, dim = 1)
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
	


