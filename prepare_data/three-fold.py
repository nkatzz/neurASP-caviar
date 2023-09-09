import pickle as pickle
import numpy as np
import torch
import random
from sklearn.preprocessing import normalize


simple_label_encoder={'active': 0, 'inactive': 1, 'running': 2, 'walking': 3}
complex_label_encoder={'no_interaction':0,'meeting': 1, 'moving': 2}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




fold_dict_test={
	"fold1":[
		"spgt_0-spgt_1",
		"mwt2gt_1-mwt2gt_2",
		"rsfgt_1-rsfgt_2",
		"br1gt_2-br1gt_3"
		"wk2gt_2-wk2gt_4",
		"lb1gt_0-lb1gt_2",
		"wk1gt_1-wk1gt_5",
		"mws1gt_1-mws1gt_5",
		"mws1gt_5-mws1gt_6",
		"lbbcgt_4-lbbcgt_6"
		]
	,
	"fold2":[
		"fra2gt_3-fra2gt_4",
		"mc1gt_1-mc1gt_3",
		"spgt_0-spgt_1",
		"fcgt_1-fcgt_2",
		"fcgt_3-fcgt_4",
		"lbgt_3-lbgt_4",
		"mws1gt_1-mws1gt_3",
		"mws1gt_5-mws1gt_6"

	],
	"fold3":[
		"spgt_0-spgt_1",
		"wk1gt_4-wk1gt_5",
		"ms3ggt_0-ms3ggt_1",
		"lb1gt_1-lb1gt_2",
		"mc1gt_1-mc1gt_3",
		"rsfgt_1-rsfgt_2",
		"rwgt_0-rwgt_2",
		"br3gt_1-br3gt_3",
		"mws1gt_1-mws1gt_6",
		"br3gt_2-br3gt_3",
		"mws1gt_6-mws1gt_7"

	]
}



fold_sample_dict={
	"simple_label_encoder":simple_label_encoder,
	"complex_label_encoder":complex_label_encoder,
	"fold1":{"train":[],"test":[],"dist_train":{'no_interaction': 0, 'meeting': 0, 'moving': 0},"dist_test":{'no_interaction': 0, 'meeting': 0, 'moving': 0}},
	"fold2":{"train":[],"test":[],"dist_train":{'no_interaction': 0, 'meeting': 0, 'moving': 0},"dist_test":{'no_interaction': 0, 'meeting': 0, 'moving': 0}},
	"fold3":{"train":[],"test":[],"dist_train":{'no_interaction': 0, 'meeting': 0, 'moving': 0},"dist_test":{'no_interaction': 0, 'meeting': 0, 'moving': 0}}
}




def create_label_tensors_simple(label_list,label_encoder):

	ll=[]

	for l in label_list:
		ll.append(label_encoder[l])

	ll = np.vstack(ll)
	label_tensor = torch.Tensor(ll).long().to(device)

	return label_tensor


def create_label_tensors_for_setup1(ll):

	ll=np.vstack(ll)
	label_tensor = torch.Tensor(ll).long().to(device)

	return label_tensor


def create_label_tensors_complex(label_list,label_encoder,complex_dist):

	ll=[]

	for l in label_list:
		ll.append(label_encoder[l])
		complex_dist[l]+=1


	return ll,complex_dist


def check_noise(s):


	std_features_p1=np.std(s['p1_features'],axis=0)
	is_all_zero_p1 = np.all((std_features_p1 == 0))

	check_inactive_p1 = all(element == 'inactive' for element in s["p1_simple_event_labels"])
	check_walking_p1 = all(element == 'walking' for element in s["p1_simple_event_labels"])


	std_features_p2=np.std(s['p2_features'],axis=0)
	is_all_zero_p2 = np.all((std_features_p2 == 0))


	check_inactive_p2 = all(element == 'inactive' for element in s["p2_simple_event_labels"])
	check_walking_p2 = all(element == 'walking' for element in s["p2_simple_event_labels"])


	if(is_all_zero_p1 and check_walking_p1):
		return True


	if(is_all_zero_p2 and check_walking_p2):
		return True


	if(is_all_zero_p1 and not check_inactive_p1):
		return True

	if(is_all_zero_p2 and not check_inactive_p2):
		return True


	return False



def add_distance_feature(p1_features,p2_features,concatenated_features):
	p1_positions=[(np.array((p1_features[i][1],p1_features[i][2]))) for i in range(0,len(p1_features))]
	p2_positions=[(np.array((p2_features[i][1],p2_features[i][2]))) for i in range(0,len(p2_features))]
	
	for tp  in range(0,len(p1_positions)):


		p1_pos=p1_positions[tp]

		p2_pos=p2_positions[tp]

		dist = np.linalg.norm(p1_pos - p2_pos)

		concatenated_features[tp].append(dist)

	return concatenated_features



def add_speed_feature(features):
	positions=[(np.array((features[i][1],features[i][2]))) for i in range(0,len(features))]
	for tp  in range(0,len(positions)):

		if(tp+1==len(positions)):
			features[tp].append(0.0)
			break

		a=positions[tp]
		b=positions[tp+1]

		dist = np.linalg.norm(a-b)

		if(tp==0):	
			features[tp].append(0.0)
		else:

			tp_speed=dist/tp
			features[tp].append(tp_speed)


	return features



def get_tensors_and_models_from_sample(s):
	p1_features=s['p1_features']
	p2_features=s['p2_features']

	complex_dist={'no_interaction': 0, 'meeting': 0, 'moving': 0}


	p1_labels=create_label_tensors_simple(s['p1_simple_event_labels'],simple_label_encoder) #use for setup2

	p2_labels=create_label_tensors_simple(s['p2_simple_event_labels'],simple_label_encoder) #use for setup2

	p1_models=s["p1_models"]
	p2_models=s["p2_models"]

	models=s['pair_models']
	complex_labels,complex_dist=create_label_tensors_complex(s["complex_ev_labels"],complex_label_encoder,complex_dist)
	atoms_for_inference=s["atoms_for_inference"]


	concatenated_features=[]
	for tp in range(0,len(p1_features)):
		concatenated_features.append(p1_features[tp]+p2_features[tp])

	concatenated_features = add_distance_feature(p1_features,p2_features,concatenated_features)




	concatenated_features = normalize(concatenated_features,axis=0,norm='max')

	concatenated_data_tensor = torch.Tensor(concatenated_features).float().view(1,concatenated_features.shape[0],concatenated_features.shape[1]).to(device)





	
	p1_features = normalize(p1_features, axis=0, norm='max')
	p2_features = normalize(p2_features, axis=0, norm='max')




	p1_data_tensor = torch.Tensor(p1_features).float().view(1,p1_features.shape[0],p1_features.shape[1]).to(device)
	p2_data_tensor = torch.Tensor(p2_features).float().view(1,p2_features.shape[0],p2_features.shape[1]).to(device)



	return [
		concatenated_data_tensor,
		complex_labels,
		p1_data_tensor,
		p2_data_tensor,
		p1_labels,
		p2_labels,
		p1_models,
		p2_models,
		models,
		atoms_for_inference,
		complex_dist
	]



no_interaction_to_use=[
	"mws1gt_3-mws1gt_6",
	"br1gt_2-br1gt_3"
	"wk2gt_2-wk2gt_4",
	"lb1gt_0-lb1gt_2",
	"wk1gt_1-wk1gt_5",
	"fcgt_3-fcgt_4",
	"lbbcgt_3-lbbcgt_5",
	"rsfgt_1-rsfgt_2",
	"rwgt_0-rwgt_2",
	"br3gt_1-br3gt_3",
	"mws1gt_1-mws1gt_6",
	"mws1gt_1-mws1gt_5",
	"lbbcgt_4-lbbcgt_6",
	"lbgt_0-lbgt_1"


	]

with open('sample_dict.pkl', 'rb') as handle:
	sample_dict = pickle.load(handle)



count=0


all_dist={
	"moving":0,
	"meeting":0,
	"no_interaction":0

}



skip_meeting_fold1_train=0
skip_meeting_fold2_train=0


skip_meeting_fold1_test=0
skip_meeting_fold3_test=0

skip_moving_fold1_test=0

skip_moving_fold2_test=0
skip_meeting_fold2_train=0

skip_moving_fold3_test=0
skip_meeting_fold3_test=0


skip_no_interaction_fold1_test=0
skip_no_interaction_fold1_train=0


skip_no_interaction_fold3_test=0
skip_no_interaction_fold3_train=0


skip_no_interaction_fold2_test=0
skip_no_interaction_fold2_train=0


pair_count=0

for up in sample_dict:


	if(sample_dict[up]["complex_event_distibution"]["meeting"] > 0 or sample_dict[up]["complex_event_distibution"]["moving"] > 0 or  (up in no_interaction_to_use)):

		print(up,sample_dict[up]["complex_event_distibution"])
		pair_count+=1


		all_dist['moving']+=sample_dict[up]["complex_event_distibution"]["moving"]
		all_dist['no_interaction']+=sample_dict[up]["complex_event_distibution"]["no_interaction"]
		all_dist['meeting']+=sample_dict[up]["complex_event_distibution"]["meeting"]

		for s in sample_dict[up]["samples"]:

			[concatenated_data_tensor,complex_labels,p1_data_tensor,p2_data_tensor,p1_labels,p2_labels,p1_models,p2_models,models,atoms_for_inference,complex_dist]=get_tensors_and_models_from_sample(s)


			record={
				"tag":0,
				"concat_tensor":concatenated_data_tensor,
				"complex_labels":create_label_tensors_for_setup1(complex_labels),
				"complex_labels_as_list":complex_labels,
				"p1_tensor":p1_data_tensor,
				"p2_tensor":p2_data_tensor,
				"p1_labels":p1_labels,
				"p2_labels":p2_labels,
				"p1_models":p1_models,
				"p2_models":p2_models,
				"models":models,
				"atoms":atoms_for_inference
			}


			

			for k in fold_dict_test:


				check_no_interaction = all(element == 'no_interaction' for element in s["complex_ev_labels"])
				check_moving = all(element == 'moving' for element in s["complex_ev_labels"])
				check_meeting = all(element == 'meeting' for element in s["complex_ev_labels"])

				skip=False
				if(up in fold_dict_test[k]):
					

					
					if(check_moving):
						record["tag"]=1



					if(k=="fold2" and check_moving and skip_moving_fold1_test<1):
						skip_moving_fold1_test+=1
						skip=True


					if(k=="fold1" and up=="spgt_0-spgt_1" and check_meeting and skip_meeting_fold1_test<2):
						skip_meeting_fold1_test+=1
						skip=True


					if(k=="fold3" and up=="wk1gt_4-wk1gt_5" and check_meeting and skip_meeting_fold3_test<20):
						skip_meeting_fold3_test+=1
						skip=True


					if(k=="fold2" and check_no_interaction and skip_no_interaction_fold2_test<19):
						skip_no_interaction_fold2_test+=1
						skip=True
					

					if(k=="fold3" and check_no_interaction and skip_no_interaction_fold3_test<6):
						skip_no_interaction_fold3_test+=1
						skip=True


					if(not skip):
						fold_sample_dict[k]["test"].append(record)

						for c in complex_dist:
							fold_sample_dict[k]["dist_test"][c]+=complex_dist[c]

				else:

					if(k=="fold1" and check_no_interaction and skip_no_interaction_fold1_train<19):
						skip_no_interaction_fold1_train+=1
						skip=True


					if(k=="fold3" and check_no_interaction and skip_no_interaction_fold3_train<13):
						skip_no_interaction_fold3_train+=1
						skip=True


					if(k=="fold1" and up=="wk2gt_1-wk2gt_2" and check_meeting and skip_meeting_fold1_train<18):
						skip_meeting_fold1_train+=1
						skip=True


					if(k=="fold2" and up=="wk2gt_1-wk2gt_2" and check_meeting and skip_meeting_fold2_train<21):
						skip_meeting_fold2_train+=1
						skip=True

					if(not skip):

						fold_sample_dict[k]["train"].append(record)
						for c in complex_dist:
							fold_sample_dict[k]["dist_train"][c]+=complex_dist[c]


random.shuffle(fold_sample_dict["fold1"]["train"])
random.shuffle(fold_sample_dict["fold2"]["train"])
random.shuffle(fold_sample_dict["fold3"]["train"])



with open('../data/3-fold_dict.pkl', 'wb') as handle:
	pickle.dump(fold_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



print("FOLD 1")
print(fold_sample_dict["fold1"]["dist_train"])
print(fold_sample_dict["fold1"]["dist_test"])

print("FOLD 2")
print(fold_sample_dict["fold2"]["dist_train"])
print(fold_sample_dict["fold2"]["dist_test"])

print("FOLD 3")
print(fold_sample_dict["fold3"]["dist_train"])
print(fold_sample_dict["fold3"]["dist_test"])


print("Total unique pairs",pair_count)
