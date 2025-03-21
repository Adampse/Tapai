# Tapai
**TAPAI: Transcriptome Processing by Artificial Intelligence**

A program for training an A.I. model to classify protein sequences into several classes and for running inference with trained models. Users provide the fasta files for training with each fasta file being considered its own class. During inference each fasta file provided will be put through the model or models sequentially. The goal of this program is to provide those who lack experience with A.I. to build and apply basic models for custom classification tasks.

## Requirements
+ Keras >= 3.0.0

## Running
No installation is required, download tapai_program.py and run from the command line.


## Default Commands
These commands are used both during training of new models and inference:
+ `train` or `predict` REQUIRED: whether to run the program for training or for inference.
+ `-ff` REQUIRED: Path to the folder holding the fasta files for either training or inference.
+ `-mp` REQURIED: During training this is the path where the new model will be saved. During inference this a path to one or more models (see running inference for more detail).
+ `-bs` Batchsize of data during training and how to batch the data during inference. Defualts to 128 in both cases.
+ `-eb` The protein sequence embedding to be used, variable length embeddings should ha


## Training a New Model
TAPAI allows user to train new models on data for specific tasks. A Basic convolutional model will be built if no model is provided. A pre-built model can be supplied using the `-pm` command followed by the path to the model. ***The pre-built model must accept used one of the embedding options presented for input and have the same number of output nodes as classes***. Fasta files are used to provide the training data with each fasta file being a single class and all fasta files for training being located in a single folder. Training is meant for smaller datasets, < 10000 sequences, as larger datasets will consume large amounts of memory as all training data is loaded into memory at once as a numpy array.


### Training Example
A line for training a model will look like this: `python tapai_program.py train -ff training_data -mp my_model` The other arguments are used to augment training to increase performance of the model. For reference, training data is used to update the model weights during training and validations sequences are for testing the model performance on _unseen_ data.
+ Depending on the total amount of data avaliable using a batchsize of 32 may be preferred
+ The `-mi` command is a simple form of data stratification used to limit the maximum amount of training instances for single class. If you have 1000 examples of a class and do `mi 400`, the program will apply the validation split to the class giving 750 training and 250 validation examples with the default `-vr`, then take 350 examples from the training set to cap it at 400 instances and append those instances to the validation set yielding 600 validation examples. This is improtant if your class data is skewed. If there are 900 examples of a class A and only 100 examples of class B, the A.I. will _cheat_ by predicting any sequence as class A as the model still achieves a good score. With this command collected data can still be used for validation while the A.I. can no longer rely on the class distribution of data to acheive a low loss.
+ Dropout rate, `-do`, is the rate or chance at which a neuron in the A.I. has it's output set to 0. This is important to combat overfitting as it forces usefulness across many neurons which prevents memorization of a few key features that don't generalize well when applied to non training data. As a general rule, overfitting occurs if the training loss is very small with good accuracy while the validation loss is poor with poor accuracy.
+ An optimized training line may look like this: `python tapai_program.py train -ff training_data -mp my_model -bs 16 -mi 150 -do 0.35`
+ The `-do`, `-sl`, and `-ed` arguments are only used when using the program provided model. They have no effect when using a prebuilt model.
+ Pre-built models need to have their text embeddings set up prior to loading.


### Training Commands
These commands are used only during training:
+ `-pm` Optional: Path to the pre-built model.
+ `-mi` The maximum amount of instances for the training set for each class. Defaults to 1000.
+ `-lr` The learning rate of the model. Defaults to 0.001.
+ `-sl` The truncation or pad-to length of the sequences. Defaults to 128. 
+ `-ed` The amount of embedding dims used in the model. Defaults to 32.
+ `-do` The dropout rate of the model, needs to be between 0 and 1. Defaults to 0.2.
+ `-vr` The validation split rate. Defaults to 0.25.
+ `-ep` The number of epochs to train for. Defaults to 16.


## Running Inference
After training a model, or finding an appropriate one, TAPAI is able to run inference on unknown transcriptomes and sort out the sequences into various classes. The program takes in multiple fasta files using the `-ff` command and splits each fasta file into multiple fasta files with each one being a predicted class. TAPAI also allows for multiple models to be used in succession, with sequences predicted to be of a specified class being fed to the next model. TAPAI expects the user to know which output node of a model corresponds to which output class.
+ The command line for running a single model will look like this: `python tapai_program.py predict -ff target_data -mp my_model`
+ If node-class relationship is known using `-nn class_A class_B class_C ...` will label the output sequences of the 1st node class_A, 2nd node class_B, 3rd node class_C, and so on.
+ For example, I have a model to predict whether sequences are a non-toxin or a toxin and I know that the 0th node corresponds to the non-toxin class and the 1st node to the toxin class. The full command line will look like this `python tapai_program.py predict -ff target_data -mp nontoxin_toxin_model -nn non_toxin toxin`
+ If the node-class relationship of the model is not known `-nn` can be left out and TAPAI will assign a non-descriptive class name.


### Stacking models
Stacking models is a useful tool for helping to continually filter out sequences using multiple models. However, it requires knowledge of the node-class relationship of the used models. It works by taking the sequnces classified as a user specified class and feeding them to the next model in a continual cycle until the last model runs. As an example I have a scorpion transcriptome and I want to find calcium, potassium, and sodium channel toxins. I have two models, one to classify sequences as either non-toxin or toxin, and the other to classify the toxin sequences as either a calcium, potassium, or sodium channel toxin. I want to run the non-toxin/toxin model first to filter out sequences then run the toxin class model to further subdivide the toxin sequences.
+ The 0th node of the non-toxin/toxin model is for non-toxin sequences, and the 1st node for toxins
+ the 0th node of the toxin class model is for calcium channel, 1st for potassium channle, and 2nd for sodium channel toxins
+ The command line will look like this `python tapai_program.py predict -ff target_data -mp nontoxin_toxin_model toxin_class_model -cn 1 -nn calcium potassium sodium`
+ At `-mp` a path to the second model is added. ***Models execute in the order they are given in the line***
+  `-cn` tells TAPAI which class goes forward, here we want the toxin sequences corresponding to the 2nd node to go forward thus `-cn 1`. If we wanted the sequences of the 1st node to go forward `-cn 0` would be used. ***The node numbering system starts at 0***. If there were 3 models and we wanted the 3rd node of the 2nd model to go forward it would look like `-cn 1 2`.
+  `-nn` applies the names given in a direct manner. The 1st name corresponds to the 0th node of the ***last model*** in the stack, 2nd name to the 1st node, and so on
+  By default TAPAI will only save the output of the final model in the stack, us the `--save_all` flag to save the output of all models
+  When using `--save_all`, node names are gone through sequentially for each model.


### Inference Commands
Commands use donly during inference:
+ `--save_all` A flag for whether to save the output of all models or only that of the final model
+ `-cn` The node for each model that continues down the stack unti lthe last model.
+ `-nn` The node names for the model. If the 1st model has _X_ output nodes, the 1st _X_ names in `-nn` will correspond to nodes of that model, and so on for the other models
+ `-of` Output folder, save path for the program output
+ `-th` Threshold value for a sequence to continue forward, needs to be between 0 and 1. Expects models to use a softmax activation.


## General Tips
+ When training, name each fasta file so that it is instantly recognizable of what class it is. The names of the fasta files are used to represent the names of the classes.
+ Record the node-class dictionary printed out at the end of training for use when stacking models and naming nodes during prediction
+ Remember, the computer science convention of starting count from 0 is used to access nodes, so a model with 4 output nodes is accessed using 0-3.


## FAQ
### Can I use TAPAI with nucleotide sequences?
Yes, models can be trained using nucleotide sequences allowing for predicitons on nucleotide sequences. However, do not mix models trained on protein sequences and those trained on nucleotide sequences, ***they cannot be exchanged***, models trained on protein sequences should only predict protein sequences and vice versa for nucletotide sequences. In addition, due to nucleotide sequences being longer they take up more memory and should be accounted for in batch size and the amount of training sequences.

