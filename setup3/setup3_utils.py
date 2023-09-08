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

'''
Use gpu if present
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
The number of simple event classes is 4 [active,inactive,walking,running]
'''
NUM_OF_CLASSES=4

'''
This asp code has all the definitions needed. For each sample
that needs to infered the outputs of the lstm on simple events
will be appended to the end alognside the aux atoms.

'''

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

'''
This method is actually the ASP-logic layer. Basically it calls clingo internally as NeurAsp.

'''
def infer_on_complex(model,data_to_infer,complex_label_encoder):


	domain=["active", "inactive", "running", "walking"]
	'''
	Get a copy of the asp code that contains all the definitions in order to append
	the classification outputs on simple events
	'''
	asp_program = get_asp()


	total_acc=0
	all_preds=[]
	all_targets=[]


	'''
	Pytorch needs this in order to evaluate the model
	'''
	with torch.no_grad():
		model.eval()

	'''
	For each testing sample get the tensors , the aux atoms and the complex event labels

	'''
	for d in data_to_infer:

		t1=d["tensors"]["p1_tensor"] # person1 tensor
		t2=d["tensors"]["p2_tensor"] # person2 tensor
		atoms=d["atoms"] # aux atoms for both persons extracted from the xml files
		labels=d["labels"] # complex event labels

		p1_out,(_,_)=model(t1) # results on simple events regarding person1
		p2_out,(_,_)=model(t2) # results on simple events regarding person2

		''''
		Softmax the output of the Lstm to get probabilities
		'''

		p1_out=F.softmax(p1_out.view(-1,NUM_OF_CLASSES), dim=1) 
		p2_out=F.softmax(p2_out.view(-1,NUM_OF_CLASSES), dim=1)

		inference_filter=""
		results=""

		'''
		Get the classification on every timestep with the greater probability
		'''

		_, p1_out = torch.max(p1_out, dim = 1)
		p1_out = p1_out.tolist()


		_, p2_out = torch.max(p2_out, dim = 1)
		p2_out = p2_out.tolist()

		'''
		Creation of asp atoms from the lstm outputs for both involved persons
		for each time-step
		'''
		for tp in range(0,len(labels)):
			'''
			This is needed in order to define the no interaction complex event.
			It is a simple rule that says if the complex event meeting or moving does not hold at that time point then
			there is no interaction between those persons. 
			'''
			inference_filter+="complexEvent(no_interaction,{}) :- not holdsAt(meeting(_,_),{}) , not holdsAt(moving(_,_),{}).\n".format(tp,tp,tp)


			happens_atom_p1="happensAt({}({}),{}).".format(domain[p1_out[tp]],"p1",tp)
			happens_atom_p2="happensAt({}({}),{}).".format(domain[p2_out[tp]],"p2",tp)

			results+=happens_atom_p1+" "+happens_atom_p2+"\n"

		inference_filter+="complexEvent(meeting,T) :- holdsAt(meeting(_,_),T).\ncomplexEvent(moving,T) :- holdsAt(moving(_,_),T).\n#show complexEvent/2."
		
		'''
		Append everything to the end of the initial asp program mentioned at line 33
		'''

		infer_program=asp_program+"\n"+atoms+"\n"+results+"\n"+inference_filter

		'''
		Run clingo and get inference results on complex events
		'''

		answers = ASP(infer_program)
		output=[]
		output_dict={}

		'''
		Parse the output of clingo and get the results using the label encoder
		'''
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


	'''
	Return a list of the encoded results and the targets
	'''

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
