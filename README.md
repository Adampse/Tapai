# Tapai
**TAPAI: Transcriptome Processing by Artificial Intelligence**

A program for training an A.I. model to classify protein sequences into several classes and for running inference with trained models. Users provide the fasta files for training with each fasta file being considered its own class. During inference each fasta file provided will be put through the model or models sequentially.

## Requirements
+ Keras >= 2.11
+ Tensorflow >= 2.11
+ Numpy >= 1.20.2
+ Python >= 3.8.12


## Default Commands
These commands are used both during training of new models and inference:
+ `train` or `predict` REQUIRED: whether to run the program for training or for inference.
+ `-ff` REQUIRED: Path to the folder holding the fasta files for either training or inference.
+ `-mp` REQURIED: During training this is the path where the new model will be saved. During inference this a path of one or more models (see running inference for more detail).
+ `-bs` Batchsize of data during training and how to batch the data during inference. Defualts to 128 in both cases.


## Training a New Model
TAPAI allows user to train new models on data for specific tasks. A Basic convolutional model will be built if no model is provided. A pre-built model can be supplied using the `-pm` command followed by the path to the model. ***The pre-built model must accept a whitespace delineated string as an input and have the same number of output nodes as classes***. Fasta files are used to provide the training data with each fasta file being a single class and all fasta files for training need to be located in a single folder. Training is meant for smaller datasets, < 5000 sequences, as larger datasets will consume large amounts of memory as all training data is loaded into memory at once as a numpy array.


### Training Example
A line for training a model will look like this: `python tapai_program.py train -ff training_data -mp my_model` The other arguments are used to augment training to achieve better performance. For reference, training data is used to update the model weights during training and validations sequences are for testing the model performance on _unseen_ data.
+ Depending on the total amount of data avaliable using a batchsize of 32 may be preferred
+ The `-mi` command is a simple form of data stratification used to limit the maximum amount of training instances for single class. If you have 1000 examples of a class and do `mi 400`, the program will apply the validation split to the class giving 750 training and 250 validation examples with the default `-vr`, then take 350 examples from the training set to cap it at 400 instances and append those instances to the validation set yielding 600 validation examples. This is improtant if your class data is skewed. Say if you have 900 examples of a class A and only 100 examples of class B, the A.I. will _cheat_ by predicting any sequence as class A as it still achieves a good score. With this command collected data can still be used for validation while the A.I. can no longer rely on the distribution of data to acheive a low loss.
+ Dropout rate, `-do`, is the rate or chance at which a neuron in the A.I. has it's output set to 0. This is important to combat overfitting as it forces usefulness across many neurons which prevents memorization of a few key features that don't generalize well when applied to non training data. As a genral rule you can tell if overfitting occurs if the training loss is very small with good accuracy while the validation loss is poor with poor accuracy.
+ An optimized training line may look like this: `python tapai_program.py train -ff training_data -mp my_model -bs 16 -mi 150 -do 0.35`


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
