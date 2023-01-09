import pickle
import re
import sys
import time
import os
import clingo
import torch
import numpy as np
import torch.nn as nn
from clyngor import ASP, solve
import json
import ast
from operator import itemgetter
import torch.nn.functional as F


from mvpp import MVPP


class NeurASP(object):
    def __init__(self, dprogram, nnMapping, optimizers, gpu=False):

        """
        @param dprogram: a string for a NeurASP program
        @param nnMapping: a dictionary maps nn names to neural networks modules
        @param optimizers: a dictionary maps nn names to their optimizers
        @param gpu: a Boolean denoting whether the user wants to use GPU for training and testing
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        self.dprogram = dprogram
        self.const = {} # the mapping from c to v for rule #const c=v.
        self.n = {} # the mapping from nn name to an integer n denoting the domain size; n would be 1 or N (>=3); note that n=2 in theorey is implemented as n=1
        self.e = {} # the mapping from nn name to an integer e
        self.domain = {} # the mapping from nn name to the domain of the predicate in that nn atom
        self.normalProbs = None # record the probabilities from normal prob rules
        self.nnOutputs = {}
        self.nnGradients = {}


        if gpu==True:
            self.nnMapping = {key : nn.DataParallel(nnMapping[key].to(self.device)) for key in nnMapping}
        else:
            self.nnMapping = nnMapping

        self.optimizers = optimizers
        # self.mvpp is a dictionary consisting of 4 keys: 
        # 1. 'program': a string denoting an MVPP program where the probabilistic rules generated from NN are followed by other rules;
        # 2. 'nnProb': a list of lists of tuples, each tuple is of the form (model, i ,term, j)
        # 3. 'atom': a list of list of atoms, where each list of atoms is corresponding to a prob. rule
        # 4. 'nnPrRuleNum': an integer denoting the number of probabilistic rules generated from NN
        self.mvpp = {'nnProb': [], 'atom': [], 'nnPrRuleNum': 0, 'program': ''}
        self.yoloInfo = [] # (beta version; used when e="yolo") a list for yolo neural network to make prediction, each element is of the form [m, t, domain]
        self.mvpp['program'], self.mvpp['program_pr'], self.mvpp['program_asp'] = self.parse(obs='')
        self.stableModels = [] # a list of stable models, where each stable model is a list

        self.dmvpp = MVPP(self.mvpp['program'])




    def constReplacement(self, t):
        """ Return a string obtained from t by replacing all c with v if '#const c=v.' is present

        @param t: a string, which is a term representing an input to a neural network
        """
        t = t.split(',')
        t = [self.const[i.strip()] if i.strip() in self.const else i.strip() for i in t]
        return ','.join(t)

    def nnAtom2MVPPrules(self, nnAtom):
        """
        @param nnAtom: a string of a neural atom
        @param countIdx: a Boolean value denoting whether we count the index for the value of m(t, i)[j]
        """

        # STEP 1: obtain all information
        regex = '^nn\((.+)\((.+)\),\((.+)\)\)$'
        out = re.search(regex, nnAtom)
        m = out.group(1)
        e, t = out.group(2).split(',', 1) # in case t is of the form t1,...,tk, we only split on the first comma
        domain = out.group(3).split(',')
        t = self.constReplacement(t)
        # check the value of e
        e = e.strip()
        if e != 'yolo':
            e = int(self.constReplacement(e))
        n = len(domain)
        if n == 2:
            n = 1
        self.n[m] = n
        self.e[m] = e
        self.domain[m] = domain
        if m not in self.nnOutputs:
            self.nnOutputs[m] = {}
            self.nnGradients[m] = {}
        if t not in self.nnOutputs[m]:
            self.nnOutputs[m][t] = None
            self.nnGradients[m][t] = None

        # STEP 2: generate MVPP rules
        mvppRules = []
        # if the neural network is yolo network
        if e == 'yolo':
            self.yoloInfo.append((m, t, domain))
        # if the neural network is a classification network that outputs probabilities only
        else:
            # we have different translations when n = 2 (i.e., n = 1 in implementation) or when n > 2
            if n == 1:
                for i in range(e):
                    rule = '@0.0 {}({}, {}, {}); @0.0 {}({}, {}, {}).'.format(m, i, t, domain[0], m, i, t, domain[1])
                    prob = [tuple((m, i, t, 0))]
                    atoms = ['{}({}, {}, {})'.format(m, i, t, domain[0]), '{}({}, {}, {})'.format(m, i, t, domain[1])]
                    mvppRules.append(rule)
                    self.mvpp['nnProb'].append(prob)
                    self.mvpp['atom'].append(atoms)
                    self.mvpp['nnPrRuleNum'] += 1

            elif n > 2:
                for i in range(e):
                    rule = ''
                    prob = []
                    atoms = []
                    for j in range(n):
                        atom = '{}({}, {}, {})'.format(m, i, t, domain[j])
                        rule += '@0.0 {}({}, {}, {}); '.format(m, i, t, domain[j])
                        prob.append(tuple((m, i, t, j)))
                        atoms.append(atom)
                    mvppRules.append(rule[:-2]+'.')
                    self.mvpp['nnProb'].append(prob)
                    self.mvpp['atom'].append(atoms)
                    self.mvpp['nnPrRuleNum'] += 1
            else:
                print('Error: the number of element in the domain %s is less than 2' % domain)
        return mvppRules


    def parse(self, obs=''):
        dprogram = self.dprogram + obs
        # 1. Obtain all const definitions c for each rule #const c=v.
        regex = '#const\s+(.+)=(.+).'
        out = re.search(regex, dprogram)
        if out:
            self.const[out.group(1).strip()] = out.group(2).strip()
        # 2. Generate prob. rules for grounded nn atoms
        clingo_control = clingo.Control(["--warn=none"])
        # 2.1 remove weak constraints and comments
        program = re.sub(r'\n:~ .+\.[ \t]*\[.+\]', '\n', dprogram)
        program = re.sub(r'\n%[^\n]*', '\n', program)
        # 2.2 replace [] with ()
        program = program.replace('[', '(').replace(']', ')')
        # 2.3 use MVPP package to parse prob. rules and obtain ASP counter-part
        mvpp = MVPP(program)
        if mvpp.parameters and not self.normalProbs:
            self.normalProbs = mvpp.parameters
        pi_prime = mvpp.pi_prime
        # 2.4 use clingo to generate all grounded NN atoms and turn them into prob. rules
        clingo_control.add("base", [], pi_prime)
        clingo_control.ground([("base", [])])
        symbols = [atom.symbol for atom in clingo_control.symbolic_atoms]
        mvppRules = [self.nnAtom2MVPPrules(str(atom)) for atom in symbols if atom.name == 'nn']
        mvppRules = [rule for rules in mvppRules for rule in rules]
        # 3. obtain the ASP part in the original NeurASP program
        lines = [line.strip() for line in dprogram.split('\n') if line and not line.startswith('nn(')]
        return '\n'.join(mvppRules + lines), '\n'.join(mvppRules), '\n'.join(lines)


    @staticmethod
    def satisfy(model, asp):
        """
        Return True if model satisfies the asp program; False otherwise
        @param model: a stable model in the form of a list of atoms, where each atom is a string
        @param asp: an ASP program (constraints) in the form of a string
        """
        asp_with_facts = asp + '\n'
        for atom in model:
            asp_with_facts += atom + '.\n'
        clingo_control = clingo.Control(['--warn=none'])
        clingo_control.add('base', [], asp_with_facts)
        clingo_control.ground([('base', [])])
        result = clingo_control.solve()
        if str(result) == 'SAT':
            return True
        return False

    

    def infer(self, dataDic, obs='', mvpp='', postProcessing=None):
        """
        @param dataDic: a dictionary that maps terms to tensors/np-arrays
        @param obs: a string which is a set of constraints denoting an observation
        @param mvpp: an MVPP program used in inference
        """

        mvppRules = ''
        facts = ''

        # Step 1: get the output of each neural network
        for m in self.nnOutputs:
            self.nnMapping[m].eval()
            for t in self.nnOutputs[m]:
                # if dataDic maps t to tuple (dataTensor, {'m': labelTensor})
                if isinstance(dataDic[t], tuple):
                    dataTensor = dataDic[t][0]
                # if dataDic maps t to dataTensor directly
                else:
                    dataTensor = dataDic[t]
                if self.e[m] == 'yolo':
                    if postProcessing:
                        self.nnOutputs[m][t] = postProcessing(self.nnMapping[m](dataTensor))
                        # Step 2: turn yolo outputs into a set of MVPP probabilistic rules
                        for idx, (label, x1, y1, x2, y2, conf) in enumerate(self.nnOutputs[m][t]):
                            # we only include the box if its label is within domain
                            if label in self.domain[m]:
                                facts += 'box({}, b{}, {}, {}, {}, {}).\n'.format(t, idx, x1, y1, x2, y2)
                                atom = '{}({}, b{}, {})'.format(m, t, idx, label)
                                mvppRules += '@{} {}; @{} -{}.\n'.format(conf, atom, 1-conf, atom)
                    else:
                        print('Error: the function for postProcessing yolo output is not specified.')
                        sys.exit()
                else:
                    self.nnOutputs[m][t] = self.nnMapping[m](dataTensor).view(-1).tolist()

        # Step 3: turn the NN outputs (from usual classification neurual networks) into a set of MVPP probabilistic rules
        for ruleIdx in range(self.mvpp['nnPrRuleNum']):
            probs = [self.nnOutputs[m][t][i*self.n[m]+j] for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]]
            if len(probs) == 1:
                mvppRules += '@{} {}; @{} {}.\n'.format(probs[0], self.mvpp['atom'][ruleIdx][0], 1 - probs[0], self.mvpp['atom'][ruleIdx][1])
            else:
                tmp = ''
                for atomIdx, prob in enumerate(probs):
                    tmp += '@{} {}; '.format(prob, self.mvpp['atom'][ruleIdx][atomIdx])
                mvppRules += tmp[:-2] + '.\n'

        # Step 3: find an optimal SM under obs
        dmvpp = MVPP(facts + mvppRules + mvpp)
        return dmvpp.find_one_most_probable_SM_under_obs_noWC(obs=obs)



    def accuracy(self,y_pred, target):
        
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        

        correct_pred = (y_pred_tags == target).float()
        acc = correct_pred.sum() / len(correct_pred)
        
        acc = torch.round(acc * 100)
        return acc


    def infer_on_complex(self,model,asp_program,data_to_infer,complex_label_encoder,ws,domain):



        total_acc=0
        all_preds=[]
        all_targets=[]



        with torch.no_grad():
            model.eval()


        for d in data_to_infer:

            t1=d["tensors"]["p1"]
            t2=d["tensors"]["p2"]
            atoms=d["atoms_for_inference"]
            labels=d["labels"]

            p1_out,(_,_)=model(t1)
            p2_out,(_,_)=model(t2)
            p1_out=F.softmax(p1_out.view(-1,len(domain)), dim=1)
            p2_out=F.softmax(p2_out.view(-1,len(domain)), dim=1)

            inference_filter=""
            results=""

            _, p1_out = torch.max(p1_out, dim = 1)
            p1_out = p1_out.tolist()


            _, p2_out = torch.max(p2_out, dim = 1)
            p2_out = p2_out.tolist()


            for tp in range(0,ws):

                inference_filter+="complexEvent(no_interaction,{}) :- not holdsAt(meeting(_,_),{}) , not holdsAt(moving(_,_),{}).\n".format(tp,tp,tp)


                happens_atom_p1="happensAt({}({}),{}).".format(domain[p1_out[tp]],"p1",tp)
                happens_atom_p2="happensAt({}({}),{}).".format(domain[p2_out[tp]],"p2",tp)

                results+=happens_atom_p1+" "+happens_atom_p2+"\n"

            inference_filter+="complexEvent(meeting,T) :- holdsAt(meeting(_,_),T).\ncomplexEvent(moving,T) :- holdsAt(moving(_,_),T).\n#show complexEvent/2."
            infer_program=asp_program+"\n"+d["atoms_for_inference"]+"\n"+results+"\n"+inference_filter

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

        return total_acc,all_preds,all_targets



    def train_setup3(self,dataList,alpha,batchSize,model_count):


        m="simpleEvent"
        nn_model=self.nnMapping[m].train()
        optimizer=self.optimizers[m]

        outputs_mapping=self.nnOutputs[m]
        gradientsBP=self.nnGradients[m]


        self.dmvpp = MVPP(self.mvpp['program_pr'])
        self.normalProbs = self.dmvpp.parameters

        for dataIdx, sample in enumerate(dataList):


            data={"p1":sample["tensors"]["p1"],"p2":sample["tensors"]["p2"]}


            models=sample["models"][:model_count]


            for key in list(data.keys()):
                data[self.constReplacement(key)] = data.pop(key)


            outputBP={}

            for t in outputs_mapping:

                out, (_,_) = nn_model(data[t].to(self.device))
                out=F.softmax(out.view(-1,self.n[m]), dim=1)


                outputBP[t]=out

                outputs_mapping[t] = out.view(-1).tolist()
                gradientsBP[t] = [0.0 for i in outputs_mapping[t]]


            for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]:
                    self.dmvpp.parameters[ruleIdx] = [outputs_mapping[t][i*self.n[m]+j] for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]]


            computed_gradients = self.dmvpp.mvppLearn(models)
            gradientsNN = computed_gradients[:self.mvpp['nnPrRuleNum']].tolist()


            for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                for probIdx, (m, i, t, j) in enumerate(self.mvpp['nnProb'][ruleIdx]):
                    gradientsBP[t][i*self.n[m]+j] = (alpha - 1) * gradientsNN[ruleIdx][probIdx]

            for t in outputs_mapping:
                outputBP[t].backward(torch.cuda.FloatTensor(np.reshape(np.array(gradientsBP[t]),outputBP[t].shape)), retain_graph=True)


            if (dataIdx+1) % batchSize == 0:
                optimizer.step()
                optimizer.zero_grad()


        return nn_model



    def neurAspLearn(self,dataList,domain,alpha,lr):
        m="simpleEvent"
        nn_model=self.nnMapping[m].train()
        optimizer=self.optimizers[m]


        self.dmvpp = MVPP(self.mvpp['program_pr'])
        self.normalProbs = self.dmvpp.parameters


        for dataIdx, data in enumerate(dataList['tensorList']):

            for key in list(data.keys()):
                data[self.constReplacement(key)] = data.pop(key)

            for person in data.keys():
                input_tensor = data[person]

                person_models = dataList["by_person_models"][dataIdx][person]

                person_out, (hn, cn) = nn_model(input_tensor.to(self.device))

                person_out=F.softmax(person_out.view(-1,len(domain)), dim=1)


                train_out=torch.log(person_out)
                _, y_pred_tags = torch.max(train_out, dim = 1)

                # person_out=torch.clamp(person_out, min=10e-8, max=1.-10e-8)
                self.nnOutputs[m]["p"] = person_out.view(-1).tolist()
                self.nnGradients[m]["p"] = [0.0 for i in self.nnOutputs[m]["p"]]


                for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                    for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]:
                        self.dmvpp.parameters[ruleIdx] = [self.nnOutputs[m][t][i*self.n[m]+j] for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]]


                # print(self.dmvpp.parameters)
                # input()

                for ruleIdx, step_probs in enumerate(self.normalProbs):
                    
                    for atom_idx ,prob in enumerate(step_probs):
                        self.dmvpp.parameters[ruleIdx][atom_idx] = prob




                # print(self.dmvpp.parameters)



                gradients = self.dmvpp.mvppLearn(person_models)
                
                gradientsNN = gradients[:self.mvpp['nnPrRuleNum']].tolist()


                for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                    for probIdx, (m, i, t, j) in enumerate(self.mvpp['nnProb'][ruleIdx]):
                        self.nnGradients[m][t][i*self.n[m]+j] = (alpha - 1) * gradientsNN[ruleIdx][probIdx]


                # gradientsNormal = gradients

                # for ruleIdx, ruleGradients in enumerate(gradientsNormal):
                #     for atomIdx, atomgrad in enumerate(ruleGradients):
                #         self.dmvpp.parameters[ruleIdx][atomIdx] += lr * atomgrad



                # self.dmvpp.normalize_probs()


                person_out.backward(torch.cuda.FloatTensor(np.reshape(np.array(self.nnGradients[m]["p"]),person_out.shape)), retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

        return nn_model




    def infer_on_simple(self,nn_model,dataList,domain):

        acc_on_simple=0
        loss_on_simple=0

        all_preds=[]
        all_ground=[]

        with torch.no_grad():
            nn_model.eval()

        for dataIdx, data in enumerate(dataList['tensorList']):
            labelTensor=dataList['labelList'][dataIdx]

            person_out,(_,_)=nn_model(data)

            person_out=F.softmax(person_out.view(-1,4), dim=1)

            train_out=torch.log(person_out)
            _, y_pred_tags = torch.max(train_out, dim = 1)
            y_pred_tags=torch.flatten(y_pred_tags).tolist()
            y_ground=torch.flatten(labelTensor).tolist()

            for yp in y_pred_tags:
                all_preds.append(yp)

            for yg in y_ground:
                all_ground.append(yg)

            

            s_acc= self.accuracy(train_out, labelTensor.long().view(-1))
 

            acc_on_simple+=s_acc.item()

        return [
            all_preds,
            all_ground,
            round(acc_on_simple/len(dataList["tensorList"]),2)
        ]



    def learn(self, dataList, obsList,factsList,epoch, alpha=0, lossFunc='cross', method='exact', lr=0.01, opt=False, storeSM=False, smPickle=None, accEpoch=0, batchSize=1):
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to either a tensor/np-array or a tuple (tensor/np-array, {'m': labelTensor})
        @param obsList: a list of strings, where each string is a set of constraints denoting an observation
        @param epoch: an integer denoting the number of epochs
        @param alpha: a real number between 0 and 1 denoting the weight of cross entropy loss; (1-alpha) is the weight of semantic loss
        @param lossFunc: a string in {'cross'} or a loss function object in pytorch
        @param method: a string in {'exact', 'sampling'} denoting whether the gradients are computed exactly or by sampling
        @param lr: a real number between 0 and 1 denoting the learning rate for the probabilities in probabilistic rules
        @param batchSize: a positive interger denoting the batch size, i.e., how many data instances do we use to update the NN parameters for once
        """
        assert len(dataList) == len(obsList), 'Error: the length of dataList does not equal to the length of obsList'
        assert alpha >= 0 and alpha <= 1, 'Error: the value of alpha should be within [0, 1]'

        # if the pickle file for stable models is given, we will either read all stable models from it or
        # store all newly generated stable models in that pickle file in case the pickle file cannot be loaded
        savePickle = False
        if smPickle is not None:
            storeSM = True
            try:
                with open(smPickle, 'rb') as fp:
                    self.stableModels = pickle.load(fp)
            except Exception:
                savePickle = True


        # get the mvpp program by self.mvpp, so far self.mvpp['program'] is a string
        if method == 'nn_prediction':
            dmvpp = MVPP(self.mvpp['program_pr'])
        elif method == 'penalty':
            dmvpp = MVPP(self.mvpp['program_pr'])
        else:
            dmvpp = MVPP(self.mvpp['program'])

        # we train all nerual network models
        for m in self.nnMapping:
            self.nnMapping[m].train()




        # we train for 'epoch' times of epochs
        for epochIdx in range(epoch):
            # for each training instance in the training data
            for dataIdx, data in enumerate(dataList):
                # data is a dictionary. we need to edit its key if the key contains a defined const c
                # where c is defined in rule #const c=v.


                for key in list(data.keys()):
                    data[self.constReplacement(key)] = data.pop(key)

                # Step 1: get the output of each neural network and initialize the gradients
                for m in self.nnOutputs:
                    nnOutput = {}
                    nnOutput[m] = {}
                    for t in self.nnOutputs[m]:
                        print(t)
                        labelTensor = None
                        # if data maps t to tuple (dataTensor, {'m': labelTensor})
                        if isinstance(data[t], tuple):
                            dataTensor = data[t][0]
                            if m in data[t][1]:
                                labelTensor = data[t][1][m]
                        # if data maps t to dataTensor directly
                        else:
                            dataTensor = data[t]

                        nnOutput[m][t] = self.nnMapping[m](dataTensor.to(self.device))


                        nnOutput[m][t] = torch.clamp(nnOutput[m][t], min=10e-8, max=1.-10e-8)


                        


                        self.nnOutputs[m][t] = nnOutput[m][t].view(-1).tolist()
                        # print(self.nnOutputs[m][t])
                        # sys.exit()
                        # initialize the semantic gradients for each output
                        self.nnGradients[m][t] = [0.0 for i in self.nnOutputs[m][t]]

                        # if alpha is greater than 0 and the labelTensor is given in dataList, we compute the nn gradients
                        if alpha > 0 and labelTensor is not None:
                            if isinstance(lossFunc, str):
                                if lossFunc == 'cross':
                                    criterion = torch.nn.NLLLoss()
                                    loss = alpha * criterion(torch.log(nnOutput[m][t].view(-1, self.n[m])), labelTensor.long().view(-1))
                            else:
                                criterion = lossFunc
                                loss = alpha * criterion(nnOutput[m][t].view(-1, self.n[m]), labelTensor)
                            loss.backward(retain_graph=True)
                # print(alpha)
                # sys.exit()
                # Step 2: if alpha is less than 1, we compute the semantic gradients
                if alpha < 1:
                    # Step 2.1: replace the parameters in the MVPP program with nn outputs
                    for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                        for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]:
                            print(ruleIdx,(m, i, t, j))
                            
                        dmvpp.parameters[ruleIdx] = [self.nnOutputs[m][t][i*self.n[m]+j] for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]]
                        if len(dmvpp.parameters[ruleIdx]) == 1:
                            dmvpp.parameters[ruleIdx] = [dmvpp.parameters[ruleIdx][0], 1-dmvpp.parameters[ruleIdx][0]]
                    
                    # print(dmvpp.parameters)
                    # print(len(dmvpp.parameters))
                    # sys.exit()
                    # Step 2.2: replace the parameters for normal prob. rules in the MVPP program with updated probabilities
                    if self.normalProbs:
                        for ruleIdx, probs in enumerate(self.normalProbs):
                            dmvpp.parameters[self.mvpp['nnPrRuleNum']+ruleIdx] = probs

                    # Step 2.3: compute the gradients
                    dmvpp.normalize_probs()
                    check = False
                    if storeSM:
                        try:
                            models = self.stableModels[dataIdx]
                            gradients = dmvpp.mvppLearn(models)
                        except:
                            if opt:
                                models = dmvpp.find_all_opt_SM_under_obs_WC(obsList[dataIdx])
                            else:
                                models = dmvpp.find_k_SM_under_obs(obsList[dataIdx], k=0)

                            
                            self.stableModels.append(models)
                            gradients = dmvpp.mvppLearn(models)
                    else:
                        #HERE
                        if method == 'exact':
                        
                            gradients = dmvpp.gradients_one_obs(obsList[dataIdx], opt=opt)
                            


                        elif method == 'sampling':
                            models = dmvpp.sample_obs(obsList[dataIdx], num=10)
                            gradients = dmvpp.mvppLearn(models)
                        elif method == 'nn_prediction': 
                            models = dmvpp.find_one_most_probable_SM_under_obs_noWC()
                            check = self.satisfy(models[0], self.mvpp['program_asp'] + obsList[dataIdx])
                            gradients = dmvpp.mvppLearn(models) if check else -dmvpp.mvppLearn(models)
                            if check:
                                continue
                        elif method == 'penalty':
                            models = dmvpp.find_all_SM_under_obs()
                            models_noSM = [model for model in models if not self.satisfy(model, self.mvpp['program_asp'] + obsList[dataIdx])]
                            gradients = - dmvpp.mvppLearn(models_noSM)
                        else:
                            print('Error: the method \'%s\' should be either \'exact\' or \'sampling\'', method)

                    # Step 2.4: update parameters in neural networks
                    gradientsNN = gradients[:self.mvpp['nnPrRuleNum']].tolist()

                    print(gradientsNN)
                    # input()

                    # print(gradientsNN)
                    # print(len(gradientsNN))
                    # input()
                    for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                        for probIdx, (m, i, t, j) in enumerate(self.mvpp['nnProb'][ruleIdx]):
                            print(ruleIdx,probIdx, (m, i, t, j),gradientsNN[ruleIdx][probIdx])
                            # input()
                            self.nnGradients[m][t][i*self.n[m]+j] = (alpha - 1) * gradientsNN[ruleIdx][probIdx]
                    # Step 2.5: backpropogate

                    # print(len(gradientsNN))
                    # print(gradientsNN)
                    # input()

                    print(nnOutput)
                    for m in nnOutput:
                        for t in nnOutput[m]:
                            # print(m,t)
                            if self.device.type == 'cuda':
                                # print(self.nnGradients[m][t],nnOutput[m][t])
                                # input("--")
                                nnOutput[m][t].backward(torch.cuda.FloatTensor(np.reshape(np.array(self.nnGradients[m][t]),nnOutput[m][t].shape)), retain_graph=True)
                            else:
                                nnOutput[m][t].backward(torch.FloatTensor(np.reshape(np.array(self.nnGradients[m][t]),nnOutput[m][t].shape)), retain_graph=True)


                    print(nnOutput)
                    sys.exit()

                # Step 3: update the parameters
                if (dataIdx+1) % batchSize == 0:
                    for m in self.optimizers:
                        self.optimizers[m].step()
                        self.optimizers[m].zero_grad()

                # Step 4: if alpha is less than 1, we update probabilities in normal prob. rules
                if alpha < 1:
                   
                    if self.normalProbs:
                        gradientsNormal = gradients[self.mvpp['nnPrRuleNum']:].tolist()
                        for ruleIdx, ruleGradients in enumerate(gradientsNormal):
                            ruleIdxMVPP = self.mvpp['nnPrRuleNum']+ruleIdx
                            for atomIdx, b in enumerate(dmvpp.learnable[ruleIdxMVPP]):
                                if b == True:
                                    dmvpp.parameters[ruleIdxMVPP][atomIdx] += lr * gradientsNormal[ruleIdx][atomIdx]
                        dmvpp.normalize_probs()
                        self.normalProbs = dmvpp.parameters[self.mvpp['nnPrRuleNum']:]

                # print(self.normalProbs)

                # Step 5: show training accuracy
                if accEpoch !=0 and (dataIdx+1) % accEpoch == 0:
                    print('Training accuracy at interation {}:'.format(dataIdx+1))
                    self.testConstraint(dataList, obsList, [self.mvpp['program']])

            # Step 6: save the stable models in a pickle file for potentially later usage
            if savePickle:
                with open(smPickle, 'wb') as fp:
                    pickle.dump(self.stableModels, fp)
                savePickle = False

    def testNN(self, nn, testLoader):
        """
        Return a real number in [0,100] denoting accuracy
        @nn is the name of the neural network to check the accuracy. 
        @testLoader is the input and output pairs.
        """
        self.nnMapping[nn].eval()
        # check if total prediction is correct
        correct = 0
        total = 0
        # check if each single prediction is correct
        singleCorrect = 0
        singleTotal = 0
        with torch.no_grad():
            for data, target in testLoader:
                output = self.nnMapping[nn](data.to(self.device))
                if self.n[nn] > 2 :
                    # print(output)
                    pred = output.argmax(dim=-1, keepdim=True) # get the index of the max log-probability
                    # print(pred,target)
                    
                    target = target.to(self.device).view_as(pred)
                    correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
                    # print(correctionMatrix)
                    # input()
                    correct += correctionMatrix.all(1).sum().item()
                    total += target.shape[0]
                    singleCorrect += correctionMatrix.sum().item()
                    singleTotal += target.numel()
                else: 
                    pred = np.array([int(i[0]<0.5) for i in output.tolist()])

                    target = target.numpy()
                    correct += (pred.reshape(target.shape) == target).sum()
                    total += len(pred)
        accuracy = 100. * correct / total
        singleAccuracy = 100. * singleCorrect / singleTotal
        return accuracy, singleAccuracy
    
    # We interprete the most probable stable model(s) as the prediction of the inference mode
    # and check the accuracy of the inference mode by checking whether the obs is satisfied by the prediction
    def testInferenceResults(self, dataList, obsList):
        """ Return a real number in [0,1] denoting the accuracy
        @param dataList: a list of dictionaries, where each dictionary maps terms to tensors/np-arrays
        @param obsList: a list of strings, where each string is a set of constraints denoting an observation
        """
        assert len(dataList) == len(obsList), 'Error: the length of dataList does not equal to the length of obsList'

        correct = 0
        for dataIdx, data in enumerate(dataList):
            models = self.infer(data, obs=':- mistake.', mvpp=self.mvpp['program_asp'])
            for model in models:
                if self.satisfy(model, obsList[dataIdx]):
                    correct += 1
                    break
        accuracy = 100. * correct / len(dataList)
        return accuracy


    def testConstraint(self, dataList, obsList, mvppList):
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to tensors/np-arrays
        @param obsList: a list of strings, where each string is a set of constraints denoting an observation
        @param mvppList: a list of MVPP programs (each is a string)
        """
        assert len(dataList) == len(obsList), 'Error: the length of dataList does not equal to the length of obsList'

        # we evaluate all nerual networks
        for func in self.nnMapping:
            self.nnMapping[func].eval()

        # we count the correct prediction for each mvpp program
        count = [0]*len(mvppList)

        for dataIdx, data in enumerate(dataList):
            # data is a dictionary. we need to edit its key if the key contains a defined const c
            # where c is defined in rule #const c=v.
            for key in list(data.keys()):
                data[self.constReplacement(key)] = data.pop(key)

            # Step 1: get the output of each neural network
            for model in self.nnOutputs:
                for t in self.nnOutputs[model]:
                    self.nnOutputs[model][t] = self.nnMapping[model](data[t].to(self.device)).view(-1).tolist()

            # Step 2: turn the NN outputs into a set of ASP facts
            aspFacts = ''
            for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                probs = [self.nnOutputs[m][t][i*self.n[model]+j] for (m, i, t, j) in self.mvpp['nnProb'][ruleIdx]]
                if len(probs) == 1:
                    atomIdx = int(probs[0] < 0.5) # t is of index 0 and f is of index 1
                else:
                    atomIdx = probs.index(max(probs))
                aspFacts += self.mvpp['atom'][ruleIdx][atomIdx] + '.\n'

            # Step 3: check whether each MVPP program is satisfied
            for programIdx, program in enumerate(mvppList):
                # if the program has weak constraints
                if re.search(r':~.+\.[ \t]*\[.+\]', program) or re.search(r':~.+\.[ \t]*\[.+\]', obsList[dataIdx]):
                    choiceRules = ''
                    for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                        choiceRules += '1{' + '; '.join(self.mvpp['atom'][ruleIdx]) + '}1.\n'
                    mvpp = MVPP(program+choiceRules)
                    models = mvpp.find_all_opt_SM_under_obs_WC(obs=obsList[dataIdx])
                    models = [set(model) for model in models] # each model is a set of atoms
                    targetAtoms = aspFacts.split('.\n')
                    targetAtoms = set([atom.strip().replace(' ','') for atom in targetAtoms if atom.strip()])
                    if any(targetAtoms.issubset(model) for model in models):
                        count[programIdx] += 1
                else:
                    mvpp = MVPP(aspFacts + program)
                    if mvpp.find_one_SM_under_obs(obs=obsList[dataIdx]):
                        count[programIdx] += 1
        for programIdx, program in enumerate(mvppList):
            print('The accuracy for constraint {} is {}'.format(programIdx+1, float(count[programIdx])/len(dataList)))
