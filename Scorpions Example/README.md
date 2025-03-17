# Example Data & Models
Used to predict sequences from Scorpion transcriptomes. Both models are small Convolutional networks that pad or truncate input sequences to a length of 128 with each being under 31,000 parameters.

## Housekeeping Data & Model
Used to train the housekeeping model to differentiate between toxin (positive) and non-toxin (negative) sequences. The positive set is a fasta file of all toxins used for training the channel model and the negative set is a drosophila transcriptome.The housekeeping model was trained with a .35 and .95 validation split for the positive and negative set respectively, achieving 94.44% validation accuracy on the negative set and 99.46% valdiation accuracy on the postivie set.
+ Instances of the positive set: 1,589
+ Instances of the negative set: 31,159

## Channel Data & Model
Used to train the channel model to sort positive seqeunces as classified by the housekeeping model into 1 of 4 classes; Calcium channel, Potassium channel, Sodium channel, and Venom (a miscellaneous category). The channel model was trained with a validation split of 0.25 and max instances set to 150, `-mi 150`. This model achieves 80%, 85.11%, 93.88%, and 92.86% validation accuracy for  Calcium channel, Potassium channel, Sodium channel, and Venom classes respectively.
+ Calcium Instances: 89
+ Potassium Instances: 620
+ Sodium Instances: 702
+ Venom Instances: 167
