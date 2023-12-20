import keras
import keras.layers as layers
import argparse
import tensorflow as tf
import numpy as np
from os import mkdir, path, listdir

# Full fledged program complete with the ability to retrain models
# and input new dictionaries

# arg parser help strings
func_help = "Whether to predict sequences, or train new models: choices: predict or train : defaults to predict"
ff_help = "REQUIRED! File path to the folder containing the fasta files wanted for processing"
mp_help = "REQUIRED! File path to the model, in training it's the save path, in prediction it's the load path"
bs_help = "Batchsize for the model during training or inference: Defaults to 128"
# args used only during inference
nn_help = "The node names for the model"
of_help = "Folder where the output fast files go: defaults to program_out in working directory"
th_help = "Threshold value for model outputs: defaults to 0 (see documentation for more detail)"
# args only used during training
pm_help = "A file path to a saved Keras model: defaults to None. ONLY USED WHEN TRAINING"
im_help = "The max amount of training instances for all classes. defaults to 1000. ONLY USED WHEN TRAINING"
lr_help = "The learning rate of the model: defaults to 1e-3. ONLY USED WHEN TRAINING"
sl_help = "The truncation or pad-to length of the sequences: defaults to 128. ONLY USED WHEN TRAINING"
ed_help = "The amount of embedding dims used in the model: defaults to 32. ONLY USED WHEN TRAINING"
do_help = "The dropout rate of the model: defaults to 0.2. ONLY USED WHEN TRAINING"
vr_help = "The validation split rate: defaults to 0.25. ONLY USED WHEN TRAINING"
ep_help = "The number of epochs to train for: defaults to 16. ONLY USED WHEN TRAINING"

#-----CL argparsing to set needed variables-----#
parser = argparse.ArgumentParser("TAPAI: Transcriptome Processing by Artificial Intelligence")
parser.add_argument("function", choices=["predict", "train"], type=str, help=func_help) # gets whether to train or predict
parser.add_argument("-ff", type=str, help=ff_help, required=True) # gets the folder with the fasta files
parser.add_argument("-mp", type=str, help=mp_help, required=True) # gets the model
parser.add_argument("-bs", type=int, help=bs_help, default=128) # gets batchsize
# args used only during inference
parser.add_argument("-nn", type=str, nargs="+", help=nn_help, default=None)
parser.add_argument("-of", type=str, help=of_help, default="program_out") # gets the folder to put outputs
parser.add_argument("-th", type=float, help=th_help, default=-1.0) # gets threshold
# args only used during training
parser.add_argument("-pm", type=str, help=pm_help, default=None)
parser.add_argument("-im", type=int, help=im_help, default=1000)
parser.add_argument("-lr", type=float, help=lr_help, default=1e-3)
parser.add_argument("-sl", type=int, help=sl_help, default=128)
parser.add_argument("-ed", type=int, help=ed_help, default=32)
parser.add_argument("-do", type=float, help=do_help, default=0.2)
parser.add_argument("-vr", type=float, help=vr_help, default=0.25)
parser.add_argument("-ep", type=int, help=ep_help, default=16)
args = parser.parse_args()

# get whether to predict or train
predict = True # the variable to determine whether to predict or train
if args.function == "train":
    predict = False

# batchsize for model execution whether for training or inference
batchsize = args.bs
assert batchsize >= 1, "Batchsize must be >= 1"

# get the fasta files within the fast folder
fasta_folder = args.ff # get the paths to the multiple fasta files
assert path.exists(fasta_folder), "Fasta folder is not found"
fasta_files = listdir(fasta_folder) # get the fasta files
# join the folder to the fasta file name for the full path
fasta_paths = [path.join(fasta_folder, fn) for fn in fasta_files] 


# TODO: redesign this function to avoid hogging loads of memory 
#   Instead may keep this for training, and use separate logic for
#   the prediction
def get_array_from_fasta(fasta_lines):
    """
    Takes lines from a fasta file and turns it into a numpy array

    Args : fasta_lines, list[str] : all lines of the fasta file

    Returns : a numpy array of dtype=str & shape (X,2) 
        where X is the number of instances in the fasta file
    """
    species_info = [] # stores the species info
    aa_seqs = [] # stores the amino acid sequences
    seq_line = [] # stores multiple lines of a single aa seq
    aa_chars = [] # stores all the characters of amino acid sequence

    def write_aa_seq(seq_line):
        """
        helper function, as the code below whas to be repeated twice
        turns the individual lines that have a single AA seq between
        them into a an appropriate input for the model

        args : seq_line, list of str : the lines that hold the AA sequence
            for an instance

        returns : None
        """
        for line in seq_line:
            for c in line:
                aa_chars.append(c) # append all otherwise
        
        aa_seqs.append(" ".join(aa_chars))
        aa_chars.clear()
        seq_line.clear()

    # iterate over fasta file
    for l in fasta_lines:
        l = l.strip() # strip trailing characters
        l = l.upper() # make all upper case

        if l[0] == ">": # if it is the beginning of a new instance
            species_info.append(l) # get the info
            # turn the previous amino acid lines into a complete sequence
            if seq_line:
                write_aa_seq(seq_line)

        else:
            seq_line.append(l)

    # get the final sequence
    write_aa_seq(seq_line)
    # convert to numpy arrays and concatenate
    info_np = np.asarray(species_info, dtype=str)[:,np.newaxis]
    aa_np = np.asarray(aa_seqs, dtype=str)[:,np.newaxis]
    assert info_np.shape[0] == aa_np.shape[0], "Mismatch of shapes, must be even!" + str(info_np.shape[0]) + " " + str(aa_np.shape[0])
    # yield automatically closes out files
    return np.concatenate((info_np,aa_np),axis=1)


def write_array(array, model_out, node_dict):
    if threshold > 0.0:
        out_indices = np.where(model_out.max(axis=-1)>=threshold)
        model_out = model_out[out_indices]

    # argmax output for the class
    model_out = np.argmax(model_out, axis=-1)

    # write output to files
    for key in node_dict.keys():
        # get the predicted classes for the key
        cls_index = np.where(model_out == key) 
        cls = array[cls_index]
        
        file_name = path.join(save_path, node_dict[key]) +".fasta"
        f = open(file_name, "a", encoding="utf8")

        # write the lines
        for i in range(cls.shape[0]):
            f.write(cls[i,0] + "\n")
            # get rid of spaces in aa seq
            aa = cls[i,1].replace(" ","")
            f.write(aa + "\n")
        f.close() # close file


# what to do during inference
if predict:
    node_names = args.nn # get the name of the models nodes
    save_path = args.of # folder you want the output files saved to
    model_path = args.mp # path to each model, order in list is execution order
    threshold = args.th # threshold for the sequence to be saved as a class

    # create the output folder if it doesn't exist already
    if not path.exists(save_path):
        mkdir(save_path)
    assert threshold <= 1.0 , "The Threshold value cannot be >= 1!"

    # load in the model
    model = keras.models.load_model(model_path)
    model.trainable = False
    out_shape = model.get_layer(index=-1).output_shape
    assert len(out_shape) == 2, "The output shape of the model should be (None, num_classes)"
    
    # get the node dictionary, if node names exist
    node_dict = dict({})
    if node_names:
        for i, name in enumerate(node_names):
            node_dict[i] = name
    else: # otherwise make one using the model output shape
        for i in range(out_shape[0]):
            node_dict[i] = "class_"+str(i)

    count = 0 # stores how many sequences have been found
    line_batch = [] # stores all the lines in the sequences
    # iterate thru the fasta files
    for fasta in fasta_paths:
        f = open(fasta, "r") # open the file

        line = next(f, None)
        while line is not None: # iterate thru the lines of the file
            # if its the info of a new sequence and line_batch isn't empty
            if line[0] == ">" and line_batch:
                # if there are enough sequences to make a batch
                if count >= batchsize: 
                    arr = get_array_from_fasta(line_batch) # get the batch array
                    # get the batch output, [:,1] removes the '>' line from the array
                    model_out = model.predict_on_batch(arr[:,1]) 
                    write_array(arr, model_out, node_dict) # write the output to files
                    count = 0 # reset count
                    line_batch.clear() # clear line batch
                
                # either continue with the batch or start a new one
                line_batch.append(line) # add to the line
                count += 1 # add one to count

            else: # otherwise just append the line to line_batch
                line_batch.append(line)
            line = next(f, None) # get the next line

        f.close() # close the file


# what to do during training
else:
    model_save_path = args.mp # get the save path for the model
    learning_rate = args.lr # get the learning rate of the model
    seq_len = args.sl # get the length to truncate or pad to
    embed_dim = args.ed # get the amount of embedding dims
    do_rate = args.do # get the dropout rate
    val_rate = args.vr # get the validation split rate
    epochs = args.ep # get the number of epochs to train for
    load_premade = args.pm is not None # whether to load a pretrained model or not
    premade_path = args.pm # path to the pretrined model
    max_instances = args.im # the max amount of isntances
    rng = np.random.default_rng()

    if load_premade:
        assert path.exists(premade_path), "Path to the premade model cannot be found"
    assert(len(model_save_path)) == 1, "Only one model can be trained at a time"

    # create a class dictionary
    class_dict = dict({})
    for i, f_f in enumerate(fasta_files):
        class_dict[i] = f_f
    print("\n Model Dictionary: ",class_dict,"\n")

    # read in the fasta files and assemble a dataset
    # will just load in all fasta files and concat as numpy arr's
    X_train, X_val = [], [] # store the training and validation array
    Y_train, Y_val = [], [] # store the training and validation answer
    for i, fp in enumerate(fasta_paths):
        # open the file, read the lines, and close
        f = open(fp)
        lines = f.readlines()
        f.close()
        # turn into array and get only the sequences at the 2nd column
        arr = get_array_from_fasta(lines)[:,1]
        del lines # free up some memory
        rng.shuffle(arr) # shuffle the array
        # get validation and training samples per class
        split = int(arr.shape[0] * val_rate)
        v = arr[:split] 
        t = arr[split:]
        # stratify based on max instances
        if t.shape[0] >= max_instances:
            z = t[max_instances:]
            t = t[:max_instances]
            v = np.concatenate([v, z], axis=0)
        # create the answer arrays
        v_y = np.repeat(i, v.shape[0])
        t_y = np.repeat(i, t.shape[0])
        # append
        X_train.append(t)
        Y_train.append(t_y)
        X_val.append(v)
        Y_val.append(v_y)
    # turn the lists of arrays into full arrays
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)

    # create the model based on the number of fasta files
    if load_premade:
        model = keras.models.load_model(premade_path)
        model.trainable = True
    else:
        model = keras.Sequential()
        model.add(layers.Input([1,], dtype=tf.string)) # needs an explicit input layer & dtype
        model.add(layers.TextVectorization(max_tokens=30,
                                            output_mode='int',
                                            output_sequence_length=seq_len))
        model.add(layers.Embedding(30, embed_dim))
        model.add(layers.Dropout(do_rate))
        for _ in range(3):
            model.add(layers.Conv1D(64,3,2))
            model.add(layers.LayerNormalization())
            model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dense(len(class_dict), activation='softmax'))
    
    model.summary() # output the model summary
    # adapt the TextVectorization layer
    model.get_layer(index=0).adapt(X_train)

    # train the model
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss_func, metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs,
            validation_data=(X_val, Y_val))
    
    # show the per class validation accuracy of the model
    for i in range(len(class_dict)):
        # get a specific class from the X val set
        cls_index = np.where(Y_val == i)
        cls = X_val[cls_index]
        # run it thru the model
        out = model.predict(cls)
        # argmax to get the predicted class number, 0 to 6
        out_index = np.argmax(out, axis=1)
        # subtract the predicted class number from the actual
        zeroed = i - out_index
        # count the non-zero elements, if an element is zero it is correct
        non_zeros = np.count_nonzero(zeroed)
        # calculate accuracy
        acc = 1.0 - (non_zeros/out.shape[0])
        print(class_dict[i], " Accuracy: ", acc)
    
    # save the model, it is in a list hence [0]
    model.save(model_save_path[0])

    




    








    



