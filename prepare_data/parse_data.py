import xml.etree.ElementTree as ET
import sys
from itertools import combinations , groupby 
import glob
import pickle
from operator import itemgetter

'''
This script is dedicated in parsing the caviar xml files
that are present in xml_caviar folder.
The data is organized in such way to facilitate the asp code generation
and the input data creation such as tensors and asp models.

The final output is a pickle file called caviar_dict.pkl which contains
the parsed data indexed by their source file. 
In each source file key of the dictionary there is an internal dictionary
which contains the features of each pair of persons mapped by their frame number.
The features of each pair is stored individually.

''' 



def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1])
    return sub_li


def get_consecutive_frames(l):
	ll=[]
	for k, g in groupby(enumerate(l), lambda i_x: i_x[0] - i_x[1]):
		ll.append(list(map(itemgetter(1), g)))

	return ll

def parse_all():

	file_list=glob.glob("xml_caviar/*.xml") #Read all xmls from all_xmls folder path
	unique_pairs=set()
	unique_persons=set()

	caviar_dict={ #create dictionary with the xml files name as key 
		k.replace("caviar/","").replace(".xml","").replace("xml_",""): #remove the folder name pre-fix

		{
			"frames_by_pair":{},
			"unique_persons":set(),
			"unique_pairs":set(),
			"total_frames":0
		} 
		for k in file_list

	} # iteration through the list containing xml file names
	
	
	for filename in file_list: #iteration through the current xml file

		tree = ET.parse(filename)
		root = tree.getroot()

		source=filename.replace("caviar/","").replace(".xml","").replace("xml_","") # get file name
		for f in root:
			ol=f.find('objectlist')
			objs=list(ol)
			frame_num=int(f.attrib['number']) 
			if(not objs):
				continue
			
			caviar_dict[source]['total_frames']=frame_num


			caviar_dict[source][frame_num]={
				"visible_persons":[],
				"features_by_person":{},
				"pairs":{}

			}
			visible_persons=[]
			for ob in objs:

				person_id="{}_{}".format(source,ob.attrib['id'])
				caviar_dict[source]["unique_persons"].add(person_id)
	
				visible_persons.append(person_id)


				ev=ob.find('hypothesislist')
				label=""
				for e in ev:
					label=e.find('movement').text


				box=ob.find('box')
				ori=int(ob.find('orientation').text)
				h=int(box.attrib['h'])
				w=int(box.attrib['w'])
				x=int(box.attrib['xc'])
				y=int(box.attrib['yc'])
				appearance=ob.find('appearance').text


				caviar_dict[source][frame_num]['features_by_person'][person_id]={
																		"ap":appearance,
																		"ori":ori,
																		"x":x,
																		"y":y,
																		"w":w,
																		"h":h,
																		"label":label
				}


			caviar_dict[source][frame_num]['visible_persons']=visible_persons
			groups=f.find('grouplist')
			visible_pairs=list(combinations(visible_persons,2))

			
			if(len(visible_pairs)):
				if(len(groups)==0):
					for vp in visible_pairs:

						p1_id=vp[0]
						p2_id=vp[1]

						pair_id = "{}-{}".format(p1_id,p2_id)
						label="no_interaction"
						caviar_dict[source][frame_num]['pairs'][pair_id] = {
																		"person1":p1_id,
																		"person2":p2_id,
																		"label":label
						}
						caviar_dict[source]["unique_pairs"].add(pair_id)
						if(pair_id not in caviar_dict[source]['frames_by_pair']):
							caviar_dict[source]['frames_by_pair'][pair_id]=[]


						if(frame_num not in caviar_dict[source]['frames_by_pair'][pair_id]):
							caviar_dict[source]['frames_by_pair'][pair_id].append(frame_num)

				else:
					for g in groups:
						members=["{}_{}".format(source,x) for x in g.find('members').text.split(',')]


						group_label=g.find('hypothesislist').find('hypothesis').find('situation').text
						if(group_label=='interacting'):
							group_label='meeting' 


						member_pairs=list(combinations(members,2))



						no_interaction_pairs = [x for x in visible_pairs if x not in member_pairs]
						
						for nip in no_interaction_pairs:

							p1_id=nip[0]
							p2_id=nip[1]

							pair_id = "{}-{}".format(p1_id,p2_id)
							label="no_interaction"							
							caviar_dict[source][frame_num]['pairs'][pair_id] = {
																			"person1":p1_id,
																			"person2":p2_id,
																			"label":label
							}

							caviar_dict[source]["unique_pairs"].add(pair_id)
							if(pair_id not in caviar_dict[source]['frames_by_pair']):
								caviar_dict[source]['frames_by_pair'][pair_id]=[]

							if(frame_num not in caviar_dict[source]['frames_by_pair'][pair_id]):
								caviar_dict[source]['frames_by_pair'][pair_id].append(frame_num)




						for ip in member_pairs:
							p1_id=ip[0]
							p2_id=ip[1]

							pair_id = "{}-{}".format(p1_id,p2_id)						
							caviar_dict[source][frame_num]['pairs'][pair_id] = {
																			"person1":p1_id,
																			"person2":p2_id,
																			"label":group_label
							}
							caviar_dict[source]["unique_pairs"].add(pair_id)


							if(pair_id not in caviar_dict[source]['frames_by_pair']):
								caviar_dict[source]['frames_by_pair'][pair_id]=[]

							if(frame_num not in caviar_dict[source]['frames_by_pair'][pair_id]):
								caviar_dict[source]['frames_by_pair'][pair_id].append(frame_num)


		caviar_dict[source]['last_frame'] = frame_num


	for source in caviar_dict.keys():

		for up in caviar_dict[source]['unique_pairs']:
			all_frames=caviar_dict[source]['frames_by_pair'][up]
			cons_frames=get_consecutive_frames(all_frames)
			
			caviar_dict[source]['frames_by_pair'][up] = cons_frames



	with open('caviar_dict.pkl', 'wb') as handle:
		pickle.dump(caviar_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
	parse_all()



if __name__ == '__main__':
	main()