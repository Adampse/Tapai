import tf_keras
import tf_keras.layers as layers
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
sa_help = "Whether to save all model outputs or only those of the last mode. defaults to False"
cn_help = "The node to continue in the model stack; used for multiple models"
nn_help = "Name of the output nodes of the model(s)"
of_help = "Folder where the output fast files go: defaults to program_out in working directory"
th_help = "Threshold value for model outputs: defaults to 0 (see documentation for more detail)"
# args only used during training
pm_help = "A file path to a saved Keras model: defaults to None. ONLY USED WHEN TRAINING"
mi_help = "The max amount of training instances for all classes. defaults to 1000. ONLY USED WHEN TRAINING"
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
parser.add_argument("-mp", type=str, nargs="+", help=mp_help, required=True) # gets the model(s)
parser.add_argument("-bs", type=int, help=bs_help, default=128) # gets batchsize
# args used only during inference
parser.add_argument("--save_all", action="store_true", help=sa_help, default=False)
parser.add_argument("-nn", type=str, nargs="+", help=nn_help, default=None)
parser.add_argument("-cn", type=int, nargs="+", help=cn_help, default=None)
parser.add_argument("-of", type=str, help=of_help, default="program_out") # gets the folder to put outputs
parser.add_argument("-th", type=float, help=th_help, default=-1.0) # gets threshold
# args only used during training
parser.add_argument("-pm", type=str, help=pm_help, default=None)
parser.add_argument("-mi", type=int, help=mi_help, default=1000)
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


def write_array(array, model_out, node_dict, fasta_index):
    fasta_file = fasta_files[fasta_index]
    sp = path.join(save_path, fasta_file)
    if not path.exists(sp):
        mkdir(sp)

    # write output to files
    for key in node_dict.keys():
        # get the predicted classes for the key
        cls_index = np.where(model_out == key) 
        cls = array[cls_index]
        
        file_name = path.join(sp, node_dict[key])+".fasta"
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
    save_all = args.save_all # whether to save all model outputs or only that of the last
    node_names = args.nn # get the name of the models nodes
    cont_nodes = args.cn # get the nodes to continue
    save_path = args.of # folder you want the output files saved to
    model_path = args.mp # path to each model, order in list is execution order
    threshold = args.th # threshold for the sequence to be saved as a class

    # create the output folder if it doesn't exist already
    if not path.exists(save_path):
        mkdir(save_path)
    assert threshold <= 1.0 , "The Threshold value cannot be >= 1!"

    # check inputs
    assert len(model_path) >= 1 and model_path[0] is not None, "A path to a model must be provided"
    if len(model_path) > 1:
        assert len(cont_nodes) == len(model_path)-1, "There should be one fewer continue nodes than models"
    else:
        assert cont_nodes is None, "The -cn argument should not be used with only 1 model for predicting"
    
    for mp in model_path:
        assert path.exists(mp), "Cannot find model with path: " + str(mp)

    # load in the models
    model_list = [] # stores the models
    node_dicts = [] # stores the node dict for each model
    nn_index = 0 # stores the current index at node names
    for index, mp in enumerate(model_path):
        model = tf_keras.models.load_model(mp)
        model.trainable = False
        #layer = layers.TFSMLayer(mp, call_endpoint="serving_default")
        out_shape = model.get_layer(index=-1).output_shape
        assert len(out_shape) == 2, "The output shape of the model should be (None, num_classes)"
        model_list.append(model)

        nd = dict({}) 
        # only get node names of all models if --save_all
        # otherwise just get the node names for the last model
        if index == len(model_path)-1 or save_all: 
            # get the node dictionary, if node names exist
            if nn_index < len(node_names):
                for i in range(out_shape[1]):
                    nd[i] = node_names[nn_index]
                    nn_index += 1
            else: # otherwise make one using the model output shape
                for i in range(out_shape[1]):
                    nd[i] = "model_"+str(index)+"_class_"+str(i)
            node_dicts.append(nd) # append the node dict for the model

    count = 0 # stores how many sequences have been found
    line_batch = [] # stores all the lines in the sequences
    # iterate thru the fasta files
    for fasta_index, fasta in enumerate(fasta_paths):
        f = open(fasta, "r") # open the file

        line = next(f, None)
        while line is not None: # iterate thru the lines of the file
            # if its the info of a new sequence and line_batch isn't empty
            if line[0] == ">" and line_batch:
                # if there are enough sequences to make a batch
                if count >= batchsize: 
                    arr = get_array_from_fasta(line_batch) # get the batch array
                    model_in = arr[:,1] # [:,1] removes the '>' line column from the array

                    for i, model in enumerate(model_list): # iterate thru the models
                        model_out = model.predict_on_batch(model_in) # get the batch output
                        out_ag = np.argmax(model_out, axis=-1) # argmax the output

                        if i == len(model_list)-1 or save_all: # write the output to files
                            if save_all: nd = node_dicts[i] # if --save_all, get the node dict that matches the model
                            else: nd = node_dicts[0] # otherwise just get the only node dict for the final model
                            write_array(arr, out_ag, nd, fasta_index) 

                        # get the next model input using cont_nodes if there are more models to go
                        if i != len(model_list)-1:
                            if threshold > 0.0: # apply threshold if needed
                                ti = np.where(model_out.max(axis=-1)>=threshold)
                                out_ag = model_out[ti]
                            # get the needed indices to continue
                            out_indices = np.where(out_ag == cont_nodes[i]) 
                            arr = arr[out_indices] # reduce the array for the next pass thru
                            model_in = arr[:,1]

                    count = 0 # reset count
                    line_batch.clear() # clear line batch
                
                # either continue with the batch or start a new one
                line_batch.append(line) # add to the line
                count += 1 # add one to count

            else: # otherwise just append the line to line_batch
                line_batch.append(line)
            line = next(f, None) # get the next line
            
        # if there are no sequences left but there is stuff in line batch
        if line_batch:
            arr = get_array_from_fasta(line_batch) # get the batch array
            model_in = arr[:,1] # [:,1] removes the '>' line column from the array

            for i, model in enumerate(model_list): # iterate thru the models
                model_out = model.predict_on_batch(model_in) # get the batch output
                out_ag = np.argmax(model_out, axis=-1) # argmax the output

                if i == len(model_list)-1 or save_all: # write the output to files
                    if save_all: nd = node_dicts[i] # if --save_all, get the node dict that matches the model
                    else: nd = node_dicts[0] # otherwise just get the only node dict for the final model
                    write_array(arr, out_ag, nd, fasta_index) 

                # get the next model input using cont_nodes if there are more models to go
                if i != len(model_list)-1:
                    if threshold > 0.0: # apply threshold if needed
                        ti = np.where(model_out.max(axis=-1)>=threshold)
                        out_ag = model_out[ti]
                    # get the needed indices to continue
                    out_indices = np.where(out_ag == cont_nodes[i]) 
                    arr = arr[out_indices] # reduce the array for the next pass thru
                    model_in = arr[:,1]
            line_batch.clear()
                    
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
    max_instances = args.mi # the max amount of isntances
    rng = np.random.default_rng()

    if load_premade:
        assert path.exists(premade_path), "Path to the premade model cannot be found"
    assert(len(model_save_path)) == 1, "Only one model can be trained at a time"
    model_save_path = model_save_path[0]

    # create a class dictionary
    class_dict = dict({})
    for i, f_f in enumerate(fasta_files):
        class_dict[i] = f_f

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
        model = tf_keras.models.load_model(premade_path)
        model.trainable = True
        out_shape = model.get_layer(index=-1).output_shape
        assert len(out_shape) == 2, "The output shape of the model should be (None, num_classes)"
        assert out_shape[1] == len(fasta_files), "The number of output nodes must be equal to the number of classes/fasta files"
    else:
        model = tf_keras.Sequential()
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
        # adapt the TextVectorization layer
        model.get_layer(index=0).adapt(X_train)
    
    model.summary() # output the model summary

    # train the model
    loss_func = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate), loss=loss_func, metrics=["accuracy"])
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

    # print out the model dictionary
    print("\n Model Dictionary")
    for key in class_dict.keys():
        print("Node ",key," = ", class_dict[key])
    # save the model
    model.save(model_save_path)

    




    








    





    




    








    



