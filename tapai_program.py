import keras
from keras import layers
import argparse
import numpy as np
from os import mkdir, path, listdir

# New edition for keras 3.0 and up
# Full fledged program complete with the ability to retrain models
# and input new dictionaries

# arg parser help strings
func_help = "Whether to predict sequences, or train new models: choices: predict or train : defaults to predict"
ff_help = "REQUIRED! File path to the folder containing the fasta files wanted for processing."
mp_help = "REQUIRED! File path to the model, in training it's the save path, in prediction it's the load path"
bs_help = "Batchsize for the model during training or inference: Defaults to 128"
eb_help = "The embedding scheme used to encode sequences. Options: AF, SR, AT, CO, HP. Defaults to AF"
sl_help = "The truncation or pad-to length of the sequences. Defaults to 128. ONLY USED WITH AT, CO, & HP EMBEDDINGS"
sd_help = "An integer seed for the random number generator. Defaults to 42."
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
do_help = "The dropout rate of the model: defaults to 0.2. ONLY USED WHEN TRAINING"
vr_help = "The validation split rate: defaults to 0.25. ONLY USED WHEN TRAINING"
ep_help = "The number of epochs to train for: defaults to 16. ONLY USED WHEN TRAINING"

#-----CL argparsing to set needed variables-----#
parser = argparse.ArgumentParser("TAPAI: Transcriptome Processing by Artificial Intelligence")
parser.add_argument("function", choices=["predict", "train"], type=str, help=func_help) # gets whether to train or predict
parser.add_argument("-ff", type=str, help=ff_help, required=True) # gets the folder with the fasta files
parser.add_argument("-mp", type=str, nargs="+", help=mp_help, required=True) # gets the model(s)
parser.add_argument("-bs", type=int, help=bs_help, default=128) # gets batchsize
parser.add_argument("-eb", choices=["AF","SR","AT","CO","HP"], nargs="+", help=eb_help, default="AF")
parser.add_argument("-sl", type=int, help=sl_help, default=128)
parser.add_argument("-sd", type=int, help=sd_help, default=42)
# args used only during inference
parser.add_argument("--save_all", action="store_true", help=sa_help, default=False)
parser.add_argument("-nn", type=str, nargs="+", help=nn_help, default=None)
parser.add_argument("-cn", type=int, nargs="+", help=cn_help, default=None)
parser.add_argument("-of", type=str, help=of_help, default="Tapai_out") # gets the folder to put outputs
parser.add_argument("-th", type=float, help=th_help, default=-1.0) # gets threshold
# args only used during training
parser.add_argument("-pm", type=str, help=pm_help, default=None)
parser.add_argument("-mi", type=int, help=mi_help, default=1000)
parser.add_argument("-lr", type=float, help=lr_help, default=1e-3)
parser.add_argument("-do", type=float, help=do_help, default=0.2)
parser.add_argument("-vr", type=float, help=vr_help, default=0.25)
parser.add_argument("-ep", type=int, help=ep_help, default=16)
args = parser.parse_args()


# redesigned to read in lines one at a time
# the arrays will still hog memory but there isn't much to do about that
def get_array_from_fasta(fasta_path):
    """
    Takes lines from a fasta file and turns it into a numpy array

    Args : fasta_path, str : path to the fasta file

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
        
        aa_seqs.append("".join(aa_chars))
        aa_chars.clear()
        seq_line.clear()

    # iterate over fasta file
    fasta = open(fasta_path, "r")
    for l in fasta:
        l = l.strip() # strip trailing characters
        l = l.upper() # make all upper case

        if l[0] == ">": # if it is the beginning of a new instance
            species_info.append(l) # get the info
            # turn the previous amino acid lines into a complete sequence
            if seq_line:
                write_aa_seq(seq_line)
        else:
            seq_line.append(l)
    fasta.close()
    # get the final sequence
    write_aa_seq(seq_line)
    # convert to numpy arrays and concatenate
    info_np = np.asarray(species_info, dtype=str)[:,np.newaxis]
    aa_np = np.asarray(aa_seqs, dtype=str)[:,np.newaxis]
    assert info_np.shape[0] == aa_np.shape[0], "Mismatch of shapes, must be even!" + str(info_np.shape[0]) + " " + str(aa_np.shape[0])
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
        f = open(file_name, "w", encoding="utf8")

        # write the lines
        for i in range(cls.shape[0]):
            f.write(cls[i,0] + "\n")
            # get rid of spaces in aa seq
            aa = cls[i,1].replace(" ","")
            f.write(aa + "\n")
        f.close() # close file


# Fixed length embedding
def align_free_embed(seq, eps=1e-5):
    amino_acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    I = np.identity(len(amino_acids))

    # get the total amount of each amino acid and cumulative series
    cumulative_arr = []
    for k, c in enumerate(seq):
        if c in amino_acids:
            i = amino_acids.index(c)
            x = I[i]
        else:
            x = np.zeros([len(amino_acids)])
        
        if cumulative_arr:
            current = cumulative_arr[k-1] + x
        else:
            current = x
        cumulative_arr.append(current)
    cumulative_arr = np.asarray(cumulative_arr)
    total = cumulative_arr[-1]

    # get the avg position, add 1e-5 to avoid /0
    avg_pos = np.divide(np.sum(cumulative_arr, axis=0), total + eps)

    # calculate variance and covariance
    mean = np.divide(np.sum(cumulative_arr, axis=0), len(seq))
    num = cumulative_arr - mean
    num = np.dot(num.T, num)
    total = total[:,np.newaxis]
    den = np.dot(total, total.T) + eps
    var_and_cov = np.divide(num, den)

    # get the diagonal and all entries above or below the diagonal
    final = []
    for i in range(var_and_cov.shape[0]):
        final.append(np.diagonal(var_and_cov, -i))
    final = np.concatenate(final, axis=0)

    out = np.concatenate([np.squeeze(total), np.squeeze(avg_pos), np.squeeze(final)], axis=0)
    return out


# Fixed length embedding
def spectral_embed(seq, eps=1e-6):
    values = {'A': 6.04,'C': 2.5,'D': 3.02,'E': 2.46,'F': 3.26,'G': 8.13,'H': 2.93,
              'I': 5.15,'K': 5.1,'L': 7.42,'M': 2.24,'N': 3.26,'P': 8.13,'Q': 2.46,
              'R': 8.32,'S': 5.19,'T': 3.15,'V': 3.61,'W': 2.24,'Y': 3.26}
    
    seq_len = len(seq)
    num_seq = []
    for s in seq:
        if s in values:
            num_seq.append(values[s])
    
    avg_sr = sum(num_seq)/seq_len # get the average spectral radius
    avg_sr = np.asarray(avg_sr)[np.newaxis]
    
    # get the distribution of spectral radii and spectral radii transitions
    #intervals = [2,2.5,3.15,5,6,7,8,8.3,9]
    F = [0 for _ in range(8)]
    transitions = np.zeros([8,8],dtype="float32")
    previous = -1
    for ns in num_seq:
        if 2 <= ns and ns <= 2.5:
            index = 0
        elif 2.5 < ns and ns <= 3.15:
            index = 1
        elif 3.15 < ns and ns <= 5:
            index = 2
        elif 5 < ns and ns <= 6:
            index = 3
        elif 6 < ns and ns <= 7:
            index = 4
        elif 7 < ns and ns <= 8:
            index = 5
        elif 8 < ns and ns <= 8.3:
            index = 6
        elif 8.3 < ns and ns <= 9:
            index = 7
 
        if previous != -1:
            transitions[previous, index] += 1
        previous = index
        F[index] += 1

    F = np.asarray(F, dtype="float32")
    F /= seq_len
    A = np.divide(transitions, seq_len-1).flatten(order="F")

    # Get the distribution and transition of 10 functional groups of amino acids
    functional = {'F':1,'Y':1,'W':1,'D':2,'E':2,'H':3,'K':4,'R':5,
                  'C':6,'M':7,'Q':8,'N':8,'S':9,'T':9,'A':10,'G':10,
                  'I':10,'L':10,'V':10,'P':10}
    G = [0 for _ in range(10)]
    transitions = np.zeros([10,10])
    previous = -1
    for s in seq:
        if s in functional:
            index = functional[s]-1
            G[index] += 1
            if previous != -1:
                transitions[previous, index] += 1
            previous = index
    G = np.asarray(G, dtype='float32')
    G /= seq_len
    B = np.divide(transitions, seq_len-1).flatten(order="F")
    
    # Get the fluctuation complexity
    amino_acids = ["A","C","D","E","F","G","H","I","K",
            "L","M","N","P","Q","R","S","T","V","W","Y"]
    aa_dict = dict({})
    for i, aa in enumerate(amino_acids):
        aa_dict[aa] = i
    del amino_acids

    pi = [0 for _ in range(20)]
    pij = np.zeros([20,20])
    previous = -1
    for s in seq:
        if s in aa_dict:
            index = aa_dict[s]
            pi[index] += 1
            if previous != -1:
                pij[previous, index] += 1
            previous = index
    pi = np.asarray(pi, dtype='float32')[:,np.newaxis]
    pi /= seq_len
    pi += eps # avoids /0
    pij = np.divide(pij, seq_len-1)
    inverse_pi = np.divide(np.ones_like(pi, dtype='float32'),pi) 
    pi_pj = np.dot(pi, inverse_pi.T)
    C = np.sum(pij * np.square(np.log(pi_pj)),axis=1)
    C = np.sum(C, axis=0, keepdims=True)
    
    # Concatenate all calculated properties into a vector
    vector = np.concatenate([avg_sr,F,A,G,B,C],axis=0)
    return vector


# Variable length embedding
def atchley_embed(seq, length):
    """
    Embeds seqs either as plain integers or with atchley matrix
    Pads sequences with 0's
    """
    atchley_matrix = {'A': [-0.59145974, -1.30209266, -0.7330651,  1.5703918, -0.14550842],
        'C': [-1.34267179,  0.46542300, -0.8620345, -1.0200786, -0.25516894],
        'D': [1.05015062,  0.30242411, -3.6559147, -0.2590236, -3.24176791],
        'E': [1.35733226, -1.45275578,  1.4766610,  0.1129444, -0.83715681],
        'F': [-1.00610084, -0.59046634,  1.8909687, -0.3966186,  0.41194139],
        'G': [-0.38387987,  1.65201497,  1.3301017,  1.0449765,  2.06385566],
        'H': [0.33616543, -0.41662780, -1.6733690, -1.4738898, -0.07772917],
        'I': [-1.23936304, -0.54652238,  2.1314349,  0.3931618,  0.81630366],
        'K': [1.83146558, -0.56109831,  0.5332237, -0.2771101,  1.64762794],
        'L': [-1.01895162, -0.98693471, -1.5046185,  1.2658296, -0.91181195],
        'M': [-0.66312569, -1.52353917,  2.2194787, -1.0047207,  1.21181214],
        'N': [0.94535614,  0.82846219,  1.2991286, -0.1688162,  0.93339498],
        'P': [0.18862522,  2.08084151, -1.6283286,  0.4207004, -1.39177378],
        'Q': [0.93056541, -0.17926549, -3.0048731, -0.5025910, -1.85303476],
        'R': [1.53754853, -0.05472897,  1.5021086,  0.4403185,  2.89744417],
        'S': [-0.22788299,  1.39869991, -4.7596375,  0.6701745, -2.64747356],
        'T': [-0.03181782,  0.32571153,  2.2134612,  0.9078985,  1.31337035],
        'V': [-1.33661279, -0.27854634, -0.5440132,  1.2419935, -1.26225362],
        'W': [-0.59533918,  0.00907760,  0.6719274, -2.1275244, -0.18358096],
        'Y': [0.25999617,  0.82992312,  3.0973596, -0.8380164, 1.5115095]}
    
    if len(seq) > length:
        seq = seq[:length]

    d = atchley_matrix
    e = [0,0,0,0,0]
    
    v = []
    for c in seq:
        if c in d:
            v.append(d[c])
        else:
            v.append(e)
    vector = np.asarray(v)
    pad = length-vector.shape[0]
    if pad > 0:
        z = np.zeros([pad, 5])
        vector = np.concatenate([vector,z], axis=0)
    return vector


# Variable length embedding
def hydropathy_embed(seq, length):
    values = {'A': 1.8,'C': 2.5,'D': -3.5,'E': -3.5,'F': 2.8,'G': -0.4,'H': -3.2,
              'I': 4.5,'K': -3.9,'L': 3.8,'M': 1.9,'N': -3.5,'P': -1.6,'Q': -3.5,
              'R': -4.5,'S': -0.8,'T': -0.7,'V': 4.2,'W': -0.9,'Y': -1.3}
    if len(seq) > length:
        seq = seq[:length]
    v = []
    for c in seq:
        if c in values:
            v.append(values[c])
        else:
            v.append(0)
    vector = np.asarray(v)
    pad = length-vector.shape[0]
    if pad > 0:
        z = np.zeros([pad])
        vector = (np.concatenate([vector,z], axis=0))
    return vector


# Variable length embedding
def cysteine_embed(seq, length):
    if len(seq) > length:
        seq = seq[:length]
    v = []
    for i in range(length):
        if i < len(seq):
            if seq[i] == 'C':
                v.append(1)
            else:
                v.append(0)
        else:
            v.append(0)
    vector = np.asarray(v)
    return vector


def embed(sequences, embedding, length=-1):
    if length == -1:
        x = [embedding(sequences[i]) for i in range(sequences.shape[0])]
    else:
        x = [embedding(sequences[i], length) for i in range(sequences.shape[0])]
    return np.asarray(x)


# get whether to predict or train
predict = True # the variable to determine whether to predict or train
if args.function == "train":
    predict = False

# set the rng seed for repeatability
rng = np.random.default_rng(seed=args.sd)

# batchsize for model execution whether for training or inference
batchsize = args.bs
assert batchsize >= 1, "Batchsize must be >= 1"

# File handling parameters
in_path = args.ff # get the paths to the multiple fasta files
assert path.exists(in_path), "File path is not found"
if path.isdir(in_path): # if the path is a folder, do this
    fasta_files = listdir(in_path) # get the fasta files
    # join the folder to the fasta file name for the full path
    fasta_paths = [path.join(in_path, fn) for fn in fasta_files]
    write_path = args.of
    if not path.exists(write_path):
        mkdir(write_path)
    save_file = False
else: # otherwise just get the singular file
    assert predict == True, "For training the given sequences need to be in seperate fasta files representing their classes"
    fasta_files = [in_path]
    fasta_paths = [in_path]
    write_path = args.of
    save_file = True

# get the seq to length
seq_length = args.sl
assert seq_length > 0, "The Sequence length must be greater than 0!"

# get the embedding scheme
embedding_dictionary = {"AF":align_free_embed, "SR":spectral_embed,
                        "AT":atchley_embed, "CO":cysteine_embed,
                        "HP": hydropathy_embed}
one_dim = True
if not predict:
    assert len(args.eb) == 1, "During training, only 1 embedding option should be given."
for e in args.eb:
    fixed_len = False
    if e == "AF":
        embedding = align_free_embed
        fixed_len = True
        embed_shape = [250]
    elif e == "SR":
        embedding = spectral_embed
        fixed_len = True
        embed_shape = [184]
    elif e == "AT":
        embedding = atchley_embed
        embed_shape = [seq_length, 5]
        one_dim = False
    elif e == "CO":
        embedding = cysteine_embed
        embed_shape = [seq_length]
    elif e == "HP":
        embedding = hydropathy_embed
        embed_shape = [seq_length]
    else:
        assert False, "Unidentified Embedding "+str(e)+". Options are: AF, SR, AT, CO, and HP"


# what to do during inference
if predict:
    embeddings = args.eb # the list of embeddings to be used
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
    node_dicts = [] # stores the node dictionary for each model
    nn_index = 0 # stores the current index at node names
    for index, mp in enumerate(model_path):
        model = keras.models.load_model(mp)
        model.trainable = False
        out_shape = model.output_shape
        assert len(out_shape) == 2, "The output shape of the model should be (None, *num classes*)"
        model_list.append(model)

        nd = dict({}) 
        # only get node names of all models if --save_all
        # otherwise just get the node names for the last model
        if index == len(model_path)-1 or save_all: 
            # get the node dictionary, if node names exist
            if node_names != None and nn_index < len(node_names):
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
        whole_arr = get_array_from_fasta(fasta)
        for i, model in enumerate(model_list): # iterate thru the models
            arr = whole_arr[:,1]
            # get the embedding for the model
            if len(embeddings) == 1:
                emb = embedding_dictionary[embeddings[0]]
            else:
                emb = embedding_dictionary[embeddings[i]]
            # perform the sequence embedding
            if fixed_len:
                model_in = embed(arr, emb)
            else:
                model_in = embed(arr, emb, length=seq_length)
            model_out = model.predict(model_in, batch_size=batchsize) # get the batch output
            # apply threshold if needed and argmax the output
            if threshold > 0.0: 
                ti = np.where(model_out.max(axis=-1)>=threshold)
                out_ag = model_out[ti] 
            else:
                out_ag = np.argmax(model_out, axis=-1) 

            if i == len(model_list)-1 or save_all: # write the output to files
                if save_all: nd = node_dicts[i] # if --save_all, get the node dict that matches the model
                else: nd = node_dicts[0] # otherwise just get the only node dict for the final model
                write_array(whole_arr, out_ag, nd, fasta_index)
            # if not the last model, get the outputs from the cont node
            if i < len(model_list) - 1:
                keep_indices = np.asarray(out_ag==cont_nodes[i]).nonzero()
                whole_arr = whole_arr[keep_indices]


# what to do during training
else:
    model_save_path = args.mp # get the save path for the model
    learning_rate = args.lr # get the learning rate of the model
    do_rate = args.do # get the dropout rate
    val_rate = args.vr # get the validation split rate
    epochs = args.ep # get the number of epochs to train for
    load_premade = args.pm is not None # whether to load a pretrained model or not
    premade_path = args.pm # path to the pretrined model
    max_instances = args.mi # the max amount of instances

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
        # open the file, read the lines and turn into array
        whole_arr = get_array_from_fasta(fp)
        arr = whole_arr[:,1] # get only the protein sequences
        # embed the sequences
        if fixed_len:
            arr = embed(arr, embedding) 
        else:
            arr = embed(arr, embedding, length=seq_length)
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
    print(X_train.shape, X_val.shape)

    # create the model based on the number of fasta files
    if load_premade:
        model = keras.models.load_model(premade_path)
        model.trainable = True
        out_shape = model.output_shape
        in_shape = model.input_shape
        err_str = "Mismatch between the input shape of the model and the embedding shape. Input shape "+in_shape+", "+"embedding shape"+embed_shape
        assert in_shape[1:] == embed_shape, err_str
        assert len(out_shape) == 2, "The output shape of the model should be (None, num_classes)"
        assert out_shape[1] == len(fasta_files), "The number of output nodes must be equal to the number of classes/fasta files"
    
    else:
        model = keras.Sequential()
        model.add(layers.Input(shape=embed_shape))
        # add a dimension for convolutions if a 1 dimensional embedding is used
        if one_dim: 
            model.add(layers.Reshape([embed_shape[0],1]))
        model.add(layers.Conv1D(64, 4, input_shape=embed_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(do_rate))
        for _ in range(3):
            model.add(layers.Conv1D(64,3,2))
            model.add(layers.LayerNormalization())
            model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dense(len(class_dict), activation='softmax'))
    
    model.summary() # output the model summary

    # train the model
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss_func, metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs,
            validation_data=(X_val, Y_val))
    
    # show the per class validation accuracy of the model
    # TODO: replace with MCC, F1 score, and write out a confusion matrix
    num_classes = len(class_dict)
    # Rows are the true class, columns are predicted class
    confusion_matrix = np.zeros(shape=[num_classes, num_classes])
    for i in range(num_classes):
        # get a specific class from the X val set
        cls_index = np.where(Y_val == i)
        cls = X_val[cls_index]
        # run it thru the model
        out = model.predict(cls)
        # argmax to get the predicted class number
        out_index = np.argmax(out, axis=1)
        # calculate the current row of the confusion matrix
        for j in range(num_classes):
            zeroed = j - out_index
            non_zeros = np.count_nonzero(zeroed)
            confusion_matrix[i,j]  = out.shape[0] - non_zeros

    # using the confusion matrix, generate the f1 score for each class
    # can't do it in the above loop as we don't know the false positives
    total_f1_scores = 0
    for i in range(num_classes):
        true_pos = confusion_matrix[i,i]
        false_pos = np.sum(confusion_matrix[:,i]) - true_pos
        false_neg = np.sum(confusion_matrix[i,:]) - true_pos
        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        f1 = (2*precision*recall)/(precision+recall)
        total_f1_scores += f1
        acc = true_pos / np.sum(confusion_matrix[i])
        print(class_dict[i], "Accuracy:", acc, "F1 score:", f1, confusion_matrix[i])
    print("Macro F1 score:",total_f1_scores/num_classes)
        
    # write out the confusion matrix to a csv file    
    labels = [class_dict[k] for k in class_dict.keys()]
    conf_mat = confusion_matrix.astype(dtype=str)
    cm = np.column_stack([np.asarray(labels), conf_mat])
    labels = ["columns are model predictions"] + labels
    cm = np.concatenate([np.expand_dims(np.asarray(labels),axis=0), cm],axis=0)
    cm_write_path = model_save_path+"_validation_confusion_matrix.csv"
    cm_file = open(cm_write_path, "w")
    for k in range(cm.shape[0]):
        line = ",".join([s for s in cm[k]])
        cm_file.write(line+"\n")


    # print out the model dictionary
    print("\n Model Dictionary")
    for key in class_dict.keys():
        print("Node ",key," = ", class_dict[key])
    # save the model
    model.save(model_save_path+".keras")


