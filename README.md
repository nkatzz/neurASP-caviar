# Neuro-Symbolic-Answer-Set-Programming-for-Complex-Event-Recognition
The purpose of this project is to investigate and present the capabilities of the NeurAsp Framework by applying an Activity Recognition Task. 
## INTRODUCTION
Given a set of video frames where persons are performing a set of actions (simple events) such as walking , running , remain idle etc we train a neural network that will recognize the interaction between pairs of persons and categorize them to complex events classes such as meeting , moving together etc. For this task we will need to create suitable input tensors that contain a sequence of features for a given number of frames (timepoints). Since sequential data is involved we have chosen to train a Bi-directional LSTM that will output classification results for each timepoint given as input. The dataset used for this thesis is the CAVIAR dataset. We are using the first section of video clips recorded in the entrance of the INRIA Labs at Grenoble, France.\
In order to have a complete overview of the capabilities of the NeurAsp Framework we have developed and compared three experimental setups:
* ***Only deep learning methods/setup1:*** In this setup we train a neural network to classify the complex events by giving as input a
single sequence tensor that contains features for two persons.
* ***Traditional neural network training and inference with logic/setup2:*** In this setup we train a neural network to classify simple events by giving as input individual sequence tensors that contain features for one person at the time. The ground truth for the simple events is provided directly from the CAVIAR dataset.The classification on complex events is achieved by applying ASP rules at the output layer.
* ***NeurAsp Integration/setup3:*** Finally we will use the neurAsp framework for training and inference. In this setup no ground truth will be provided for the simple events but rather stable ASP models that contain simple events. Again the classification on complex events is achieved by applying ASP rules at the output layer.
## SYSTEM REQUIREMENTS
* python3
* pip3
* clingo
## ENVIRONMENT SETUP
* ***clone repository:*** git clone https://github.com/AndreasOikonomakis/Neuro-Symbolic-Answer-Set-Programming-for-Complex-Event-Recognition.git
* ***install virtualenv:*** pip install virtualenv
* ***create virtual environment:*** python -m venv env_neurAsp
* ***activate/source environment:*** source env_neurAsp/bin/activate
* ***install python modules:*** pip install -r requirements.txt
### CREATE DATA
To start training we need to generate suitable data:
* ***nagivate to prepare data folder:*** cd prepare_data
* ***parse the xml files in xml_caviar folder***: python parse_data.py (a sample_dict.pkl will be generated in the same folder)
* ***create asp files ,asp models and tensors***: python generate_asp_on_ws.py (asp files will be generated in asp_on_ws folder)
* ***perform a three-fold data split***: python three-fold.py (a 3-fold_dict.pkl will be generated in the data folder (in root folder of the project))
### TRAIN AND TEST EXPERIMENT SETUPS
Every setup folder has the same structure:
* ***nagavigate to each setup folder***: cd setup1 | cd setup2 | cd setup3
* ***train setup*** : python train_setup1.py | python train_setup2.py | python train_setup3.py
* ***results of setup (after training)***: python results_setup1.py | python results_setup2.py | python results_setup3.py
