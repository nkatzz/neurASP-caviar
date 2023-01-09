import pickle5 as pickle
import sys
import glob
import os
import sys
from itertools import islice , combinations , permutations
import re
from operator import itemgetter
from sklearn.preprocessing import normalize
import numpy as np
import subprocess


'''
This script generates asp code on pre-defined time windows.
The procedure is exectuted in two phases.
-During the first phase we provide the xml data with the annotated
 ground truth on simple events and we generate asp models that contain
 atoms that indeicate which complex event holds in what frame in the given
 time window.
-During the second phase we provide the generated models of the first phase
 as observations (constraints) in order to generate models that contain simple events.
 Those models are used for the NeurAsp framework in setup3.

The genetated asp files are stored in the asp_on_ws folder for both phases.
The two phases uses background knowledges with all the definitions of the event calculus
The BKs for both phases can be found at the end of the script    

'''


#clingo must be installed in your system
#this line of code will search for the clingo path. If not provide it hardcoded in the next commented line
clingo_path=str(subprocess.check_output(["which", "clingo"])).replace("\\n"," ")[1:].replace("\'","")
# clingo_path="/usr/bin/clingo "


def main():
	ws=24 #window size
	cov=0.5 #coverage 
	asp_on_ws_ground(ws,cov)



# This method created the sliding time window of the given size and coverage
def SlWindows(fseq, window_size, coverage):

	res=[]
	it = iter(fseq)
	result = list(islice(it, 0, window_size))
	res.append(result)
	step_size=max(1, int(round(window_size*coverage)))   
	while True:
		new_elements = list(islice(it, step_size))
		result = result[step_size:] + new_elements
		res.append(result)
		if len(new_elements) < step_size:
			res.append(result)
			return res

#This method is used to salvage some samples that do not contain enough frames
#It takes some consequtive frames from a previous slice.
def equalize_ws(slwindows,ws):

	eq_windows=[]

	prev_window_start=-1
	for i in range(0,len(slwindows)):

		if(slwindows[i][0]==prev_window_start):
			prev_window_start=slwindows[i][0]
			continue
		prev_window_start=slwindows[i][0]

		diff=ws-len(slwindows[i])

		if(diff):
			sl_start=slwindows[i][0]
			for j in range(1,diff+1):
				slwindows[i]=[sl_start-j]+slwindows[i]

		eq_windows.append(slwindows[i])


	return eq_windows

#This method is used to create the atoms taken directly from the data.
#In fact it creates the asp code needed for the time window.
def asp_person_atoms(info,frame_num,first_frame,person_list):


	asp_lines1=""
	asp_lines2=""
	asp_lines3=""


	map_persons={
		person_list[0]:"p1",
		person_list[1]:"p2"
	}



	[person_id,ap,ori,x,y,w,h,ground_simple] = [info['id'],info['ap'],info['ori'],info['x'],info['y'],info['w'],info['h'],info['label']]
	orientation_atom="orientation({},{},{}).".format(person_id,ori,frame_num)
	coords_atom="coords({},{},{},{}). ".format(person_id,x,y,frame_num)
	appearance_atom=""
	if(ap == 'visible'):
		appearance_atom = "visible({},{}).".format(person_id,frame_num)
	else:
		appearance_atom = "happensAt({}({}),{}).".format(ap,person_id,frame_num)

	simple_event_atom_ground = "happensAt({}({}),{}).".format(ground_simple,person_id,frame_num)

	orientation_atom3="orientation({},{},{}).".format(map_persons[person_id],ori,frame_num-first_frame)
	coords_atom3="coords({},{},{},{}). ".format(map_persons[person_id],x,y,frame_num-first_frame)


	appearance_atom3=""
	if(ap == 'visible'):
		appearance_atom3 = "visible({},{}).".format(map_persons[person_id],frame_num-first_frame)
	else:
		appearance_atom3 = "happensAt({}({}),{}).".format(ap,map_persons[person_id],frame_num-first_frame)



	asp_lines1+="{} {} {} {}".format(orientation_atom,coords_atom,appearance_atom,simple_event_atom_ground)
	asp_lines2+="{} {} {}".format(orientation_atom,coords_atom,appearance_atom)
	asp_lines3+="{} {} {}".format(orientation_atom3,coords_atom3,appearance_atom3)

	return [asp_lines1,asp_lines2,asp_lines3]


#this method calls internally clingo to generate the models of the first phase.
#the output is parsed on the fly using regexes and takes the output of the
#holdsAt atom and finally gets which complex event holds in that time window
#and which persons are involved.
def parse_phase1(clingo_output):


	complex_event_dict={}

	results = clingo_output.split(" ")
	for r in results:
		matches = re.findall("holdsAt\((\w*)\((\w*),(\w*)\),(\d*)", r)#parse holdsAt atom (get who is involved and which complex event holds)
		if(matches):
			complex_event=matches[0][0]

			person1=matches[0][1]
			person2=matches[0][2]
			frame_num=int(matches[0][3])
			source=person1.split("_")[0]
			if(complex_event not in complex_event_dict):
				complex_event_dict[complex_event]=[]

			if(frame_num not in complex_event_dict[complex_event]):
				complex_event_dict[complex_event].append(frame_num)

	return complex_event_dict


#this method generates the choice rules on simple events on the given time-range
#the return value of this method is appened to the phase2 asp files
def phase2_choice_rules(p1,p2,start,end):

	r1="1{simpleEvent(T,"+p1+",active); simpleEvent(T,"+p1+",inactive); simpleEvent(T,"+p1+",walking); simpleEvent(T,"+p1+",running)}1 :- T="+str(start)+".."+str(end)+".\n"
	r2="1{simpleEvent(T,"+p2+",active); simpleEvent(T,"+p2+",inactive); simpleEvent(T,"+p2+",walking); simpleEvent(T,"+p2+",running)}1 :- T="+str(start)+".."+str(end)+".\n"

	return r1+r2

#this method creates the observations using the output of the first phase
#in the given time window
def phase2_observations(p1,p2,complex_event_dict):

	obs=""
	first=True
	complex_ev_labels={}

	all_frames=[]
	for e in complex_event_dict:
		for fr in complex_event_dict[e]:
			all_frames.append([e,fr])

	all_frames=list(sorted(all_frames, key=itemgetter(1)))

	start_event=all_frames[0][0]
	start_frame=all_frames[0][1]-1

	end_event=all_frames[-1][0]
	end_frame=all_frames[-1][1]+1


	obs+=":- holdsAt({}({},{}),{}). :- holdsAt({}({},{}),{}).\n".format(start_event,p1,p2,start_frame,start_event,p2,p1,start_frame)
	prev_e=start_event
	for e,fr in all_frames:

		complex_ev_labels[fr]=e
		if(prev_e==e):
			obs+=":- not holdsAt({}({},{}),{}). :- not holdsAt({}({},{}),{}).\n".format(e,p1,p2,fr,e,p2,p1,fr)
		else:
			obs+=":- holdsAt({}({},{}),{}). :- holdsAt({}({},{}),{}).\n".format(prev_e,p1,p2,fr,prev_e,p2,p1,fr)
			obs+=":- holdsAt({}({},{}),{}). :- holdsAt({}({},{}),{}).\n".format(e,p1,p2,fr-1,e,p2,p1,fr-1)

		prev_e=e

	obs+=":- holdsAt({}({},{}),{}). :- holdsAt({}({},{}),{}).\n".format(end_event,p1,p2,end_frame,end_event,p2,p1,end_frame)

	if('meeting' in complex_event_dict):
		meeting_start=complex_event_dict["meeting"][0]-1
		meeting_end=complex_event_dict["meeting"][-1]
		# obs+=":- coords(P,X1,Y1,T1),coords(P,X2,Y2,T2), X1=X2 , Y1=Y2 ,T2=T1+1 ,T1 >{} ,T2<{} ,simpleEvent(T2,P,active).\n".format(meeting_start,meeting_end)
		obs+=":- simpleEvent(T,P,walking), T>{} , T<{}.\n".format(meeting_start,meeting_end)
		obs+=":- simpleEvent(T,P,running), T>{} , T<{}.\n".format(meeting_start,meeting_end)

	if('moving' in complex_event_dict):
		moving_start=complex_event_dict["moving"][0]-1
		moving_end=complex_event_dict['moving'][-1]
		obs+=":- simpleEvent(T,P,inactive), T>{} , T<{}.\n".format(moving_start,moving_end)
		obs+=":- simpleEvent(T,P,running), T>{} , T<{}.\n".format(moving_start,moving_end)





	return obs, [x[1] for x in all_frames] , complex_ev_labels


#this method calls clingo internally and parses its output using regexes
#in order to retrieve models on simple events. Those models are used to
#train the neural network using the NeurAsp framework (setup3)
def parse_phase2(clingo_output):


	models_list=[]
	model=[]
	model_count=0


	models_dict={
		"pair_models":[]
	}


	person_list=[]

	for line in clingo_output:

		if('Answer:' in line or 'SATISFIABLE' in line):

			if(len(model)):

				model_count+=1
				person_list=list(sorted(person_list))

				map_persons={
					person_list[0]:"p1",
					person_list[1]:"p2"
				}

				model=list(sorted(model, key=itemgetter(0)))


				if(person_list[0] not in models_dict):
					models_dict[person_list[0]]=[]

				if(person_list[1] not in models_dict):
					models_dict[person_list[1]]=[]


				cur_model_dict={

					"pair_model":[],
					person_list[0]:[],
					person_list[1]:[]
					
				}



				first_frame=model[0][0]

				for t in model:
					[fr,atom] = t
					
					re_index_frame = fr-first_frame
					matches = re.findall("simpleEvent\((\d*),(\w*),(\w*)\)", atom)#parse the person , the frame number and the simple event
					for m in matches:
						person=m[1]
						simple_event=m[2]
						actual_atom="simpleEvent({},{},{})".format(re_index_frame,map_persons[person],simple_event)
						actual_atom_for_person="simpleEvent({},p,{})".format(re_index_frame,simple_event)
					

					cur_model_dict[person].append(actual_atom_for_person)
					cur_model_dict["pair_model"].append(actual_atom)


				model=[]


				if(cur_model_dict["pair_model"] not in models_dict["pair_models"]):
					models_dict["pair_models"].append(cur_model_dict["pair_model"])


				if(cur_model_dict[person_list[0]] not in models_dict[person_list[0]]):
					models_dict[person_list[0]].append(cur_model_dict[person_list[0]])


				if(cur_model_dict[person_list[1]] not in models_dict[person_list[1]]):
					models_dict[person_list[1]].append(cur_model_dict[person_list[1]])



		matches = re.findall("simpleEvent\((\d*),(\w*),(\w*)\)", line)

		if(matches):
			for m in matches:
			
				frame=m[0]
				person=m[1]
				if(person not in person_list):
					person_list.append(person)

				simple_event=m[2]
				model.append([int(frame),"simpleEvent({},{},{})".format(frame,person,simple_event)])



	return models_dict


#this is the method that executes the phase1 and phase2 asp code generation
def asp_on_ws_ground(ws,cov):

	ws=ws+1
	tensors_with_models=0
	folder="asp_on_ws"

	#Clear folder before execution
	files = glob.glob(folder+"/*")
	for f in files:
		os.remove(f)


	with open('caviar_dict.pkl', 'rb') as handle:
		caviar_dict = pickle.load(handle)

	sample_dict={}

	for source in caviar_dict.keys():

		total_frames=caviar_dict[source]["total_frames"]
		
		for up in caviar_dict[source]['unique_pairs']:


			sample_dict[up]={
				"complex_event_distibution":{
					"moving":0,
					"meeting":0,
					"no_interaction":0
				},
				"simple_event_distribution":{
					"walking":0,
					"active":0,
					"inactive":0,
					"running":0
				},
				"samples":[]

			}

			cons_frames=caviar_dict[source]['frames_by_pair'][up]
			for cf in cons_frames:

				if(len(cf)<ws):
					continue

				slwindows=SlWindows(cf,ws,cov)
				slwindows=equalize_ws(slwindows,ws)
				for slw in slwindows:
					
					[person1,person2]=up.split("-")

					asp_phase1=get_bk_phase1()

					asp_phase2=get_bk_phase2()
					asp_phase2+="person({};{}).\n".format(person1,person2)

					f_name_phase1="{}/{}-{}_{}-{}.lp".format(folder,up,slw[0],slw[-1],"phase1")


					person_list=list(sorted(up.split("-")))
					
					pair_atoms_phase1=""
					pair_atoms_phase2=""
					pair_atoms_inference=""

					for frame_num in slw:
						info_p1 = caviar_dict[source][frame_num]['features_by_person'][person1]
						info_p1['id']=person1

						info_p2 = caviar_dict[source][frame_num]['features_by_person'][person2]
						info_p2['id']=person2

						[atoms_phase1_p1,atoms_phase2_p1,atoms_inference_p1] = asp_person_atoms(info_p1,frame_num,slw[0],person_list)
						[atoms_phase1_p2,atoms_phase2_p2,atoms_inference_p2] = asp_person_atoms(info_p2,frame_num,slw[0],person_list)

						pair_atoms_phase1+=atoms_phase1_p1+" "+atoms_phase1_p2+"\n"
						

						pair_atoms_phase2+=atoms_phase2_p1+" "+atoms_phase2_p2+"\n"
						
						pair_atoms_inference+=atoms_inference_p1+" "+atoms_inference_p2+"\n"

						# print(caviar_dict[source][frame_num]['pairs'][up])

					[_,atoms_phase2_p1,atoms_inference_p1] = asp_person_atoms(info_p1,frame_num+1,slw[0],person_list)
					[_,atoms_phase2_p2,atoms_inference_p2] = asp_person_atoms(info_p2,frame_num+1,slw[0],person_list)
					pair_atoms_phase2+=atoms_phase2_p1+" "+atoms_phase2_p2+"\n"
					pair_atoms_phase2+="happensAt(disappear({}),{}). happensAt(disappear({}),{}).\n".format(person1,frame_num,person2,frame_num)

					# print(pair_atoms_inference)
					# input()



					asp_phase1+=pair_atoms_phase1
					asp_phase1+="#show holdsAt/2."

					f = open(f_name_phase1, "w")
					f.write(asp_phase1)
					f.close()

					asp_phase2+=pair_atoms_phase2


					cmd=clingo_path+f_name_phase1

					clingo_output = os.popen(cmd).read()
					# sys.exit()
					complex_event_dict=parse_phase1(clingo_output)

					atoms_for_inference_in_window=[]

					if(len(complex_event_dict)==0):

						raw_features_p1=[]
						raw_features_p2=[]
						complex_ev_labels=[]
						p1_simple_event_labels=[]
						p2_simple_event_labels=[]


						

						frame_idx=0
						for frame_num in slw:
							info_p1 = caviar_dict[source][frame_num]['features_by_person'][person1]
							info_p2 = caviar_dict[source][frame_num]['features_by_person'][person2]
							raw_features_p1.append([info_p1['ori'],info_p1['x'],info_p1['y'],info_p1['w'],info_p1['h']])
							raw_features_p2.append([info_p2['ori'],info_p2['x'],info_p2['y'],info_p2['w'],info_p2['h']])

							complex_ev_labels.append("no_interaction")
							p1_simple_event_labels.append(info_p1['label'])
							p2_simple_event_labels.append(info_p2['label'])
							[_,_,atoms_inference_p1] = asp_person_atoms(info_p1,frame_idx,0,person_list)
							[_,_,atoms_inference_p2] = asp_person_atoms(info_p2,frame_idx,0,person_list)

							atoms_for_inference_in_window.append(atoms_inference_p1+" "+atoms_inference_p2)
							frame_idx+=1





						if(len(complex_ev_labels)>ws-1):
							complex_ev_labels=complex_ev_labels[:-1]
							raw_features_p1=raw_features_p1[:-1]
							raw_features_p2=raw_features_p2[:-1]
							p1_simple_event_labels=p1_simple_event_labels[:-1]
							p2_simple_event_labels=p2_simple_event_labels[:-1]
							atoms_for_inference_in_window=atoms_for_inference_in_window[:-1]


						# norm_features1 = normalize(raw_features_p1, axis=0, norm='max')
						# norm_features2 = normalize(raw_features_p2, axis=0, norm='max')


						for p_labels in [p1_simple_event_labels,p2_simple_event_labels]:
							for l in p_labels:
								sample_dict[up]['simple_event_distribution'][l]+=1

						for com_ev_l in complex_ev_labels:
							sample_dict[up]['complex_event_distibution'][com_ev_l]+=1


						pair_atoms_inference=""
						for ia in atoms_for_inference_in_window:
							pair_atoms_inference+=ia



						sample_dict[up]["samples"].append({

							"sample_flag":False,
							"p1_features":raw_features_p1,
							"p2_features":raw_features_p2,
							"p1_simple_event_labels":p1_simple_event_labels,
							"p2_simple_event_labels":p2_simple_event_labels,
							"p1_models":[],
							"p2_models":[],
							"pair_models":[],
							"complex_ev_labels":complex_ev_labels,
							"atoms_for_inference":pair_atoms_inference,

						})

						continue


					complex_ev_labels=[]
					[obs,model_frames,complex_ev_labels_dict] = phase2_observations(person1,person2,complex_event_dict)


					# print(complex_ev_labels)

					if(len(model_frames)<ws-1 or len(model_frames)>ws-1):
						continue


					meeting_flag=""
					moving_flag=""

					if("meeting" in complex_event_dict):
						meeting_flag="meeting"

					if("moving" in complex_event_dict):
						moving_flag="moving"


					choice_r = phase2_choice_rules(person1,person2,slw[0],slw[-1])

					asp_phase2+=choice_r

					asp_phase2+=obs

					asp_phase2+="\n#show simpleEvent/3."

					f_name_phase2="{}/{}-{}_{}-{}-{}-{}.lp".format(folder,up,slw[0],slw[-1],meeting_flag,moving_flag,"phase2")
					f = open(f_name_phase2, "w")
					f.write(asp_phase2)
					f.close()
					
					cmd=clingo_path+"{} {}".format(f_name_phase2,30)

					clingo_output = os.popen(cmd).read()
					models_dict=parse_phase2(clingo_output.split("\n"))
					
					tensors_with_models+=1
					print("{} created".format(f_name_phase2))

					raw_features_p1=[]
					raw_features_p2=[]




					# for com_ev_l in complex_ev_labels:
					# 	sample_dict[up]['complex_event_distibution'][com_ev_l]+=1

					p1_simple_event_labels=[]
					p2_simple_event_labels=[]


					# print(complex_ev_labels_dict.keys(),len(complex_ev_labels_dict))

					frame_idx=0
					for frame_num in model_frames:

						if(frame_num in complex_ev_labels_dict):

							info_p1 = caviar_dict[source][frame_num]['features_by_person'][person1]
							info_p2 = caviar_dict[source][frame_num]['features_by_person'][person2]

							p1_simple_event_labels.append(info_p1['label'])
							p2_simple_event_labels.append(info_p2['label'])



							com_ev_l = complex_ev_labels_dict[frame_num]
							sample_dict[up]['complex_event_distibution'][com_ev_l]+=1
							complex_ev_labels.append(com_ev_l)



							raw_features_p1.append([info_p1['ori'],info_p1['x'],info_p1['y'],info_p1['w'],info_p1['h']])
							raw_features_p2.append([info_p2['ori'],info_p2['x'],info_p2['y'],info_p2['w'],info_p2['h']])



							[_,_,atoms_inference_p1] = asp_person_atoms(info_p1,frame_idx,0,person_list)
							[_,_,atoms_inference_p2] = asp_person_atoms(info_p2,frame_idx,0,person_list)

							atoms_for_inference_in_window.append(atoms_inference_p1+" "+atoms_inference_p2)
							frame_idx+=1

	

					pair_atoms_inference=""
					for ia in atoms_for_inference_in_window:
						pair_atoms_inference+=ia


					for p_labels in [p1_simple_event_labels,p2_simple_event_labels]:
						for l in p_labels:
							sample_dict[up]['simple_event_distribution'][l]+=1


					s_flag=False
					if("meeting" not in f_name_phase2):
						s_flag=True




					sample_dict[up]["samples"].append({

						"sample_flag":s_flag,
						"p1_features":raw_features_p1,
						"p2_features":raw_features_p2,
						"p1_simple_event_labels":p1_simple_event_labels,
						"p2_simple_event_labels":p2_simple_event_labels,
						"p1_models":models_dict[person1],
						"p2_models":models_dict[person2],
						"pair_models":models_dict["pair_models"],
						"complex_ev_labels":complex_ev_labels,
						"atoms_for_inference":pair_atoms_inference,

					})


	for up in sample_dict:
		print(up,sample_dict[up]["complex_event_distibution"])

	#final dictionary output
	with open('sample_dict.pkl', 'wb') as handle:
		pickle.dump(sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



#background knowledge for phase1
def get_bk_phase1():
	bk="""
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

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

	"""

	return bk

#background knowledge for phase2
def get_bk_phase2():

	bk="""

holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	"""

	return bk






if __name__ == '__main__':
	main()