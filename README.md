# Tapai
TAPAI: Transcriptome Processing by Artificial Intelligence

A program for training an A.I. model to classify sequences into several classes and for running predictions with trained models. Users provide the fasta files for training with each fasta file being considered its own class. During predictions each fasta file provided will be put through the model sequentially.

# Requirements
Keras >= 2.11
Tensorflow >= 2.11
Numpy >= 1.20.2
Python >= 3.8.12

# Training a New Model
Put all fasta files for training, with each fasta file being used for a whole class, into a singular folder. Run the program as >python tapai_program.py train -ff fasta_folder_name -mp model_save_path.
