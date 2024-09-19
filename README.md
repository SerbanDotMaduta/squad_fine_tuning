### LLM fine-tuning on SQUAD dataset

#### Intro
The purpose is to obtain a fine-tuned version of a pre-trained LLM on the [SQUAD dataset](https://huggingface.co/datasets/rajpurkar/squad).
The LLM to be fine tuned on the dataset is ```distilbert-base-uncased```. DistilBERT is a lighter version of BERT.
Focus was to create a starting point for the actual fine-tuning process because achieving best results would take a significant more amount
of time.

#### Data Description
SQUAD dataset is a set of Q&As which contains 100k examples of ("context", "question", "answer") pairs. For the sake of our experiment, I took only the first **5k examples** and ran experiments on it.

20% of the training data was used for validation resulting in a split of 4k samples for training and 1k samples for testing.

#### Model Fine-tuning Approach
Two approaches were selected for fine tuning: Full fine-tuning and a PEFT technique called LoRA.
1. **Full fine-tuning**: It is a procedure in which all the parameters of the original network are updated during the backprop phase.
    * Advantages: \
        a. Theoretically, a better score metric could be achieved due to a larger set of parameters.\
        b. Could work well with a limited dataset
    * Disadvantages
        a. Slower training due to a higher number of parameters which need to be updated.\
        b. Computational expensive. \
        c. Risk of "catastrophic forgetting", since all the parameters of the network get updated, there is a risk to mess up the existing relations between tokens.  
1. **LoRA**: Stands for low-rank adaptation. It is a more modern approach to fine tuning. Only a small subset of parameters is introduced between specific layers of the network and this subset get updated during training.
    * Advantages: \
        a. Cheaper in terms of resources than full fine-tuning.\
        b. Usually faster.
    * Disadvantages
        a. High chance of not being able to achieve the best end result but the trade-off usually is convenient.\
        b. Extra set of optimal training parameters need to be found.

#### Experimental Setup
Experiments were ran on an on-prem environment with the following specifications:
* OS: Ubuntu 22.04 LTS
* GPU: Nvidia Geforce RTX 3080 Ti 16 GB memory
* Driver version 535.183, CUDA version 12.2

#### Results
The target metric is the one suggested by the squad dataset which is a combination between the ```Exact match``` score and the ```F1``` score.


|  | Full-fine tuning  | LoRA |
| -------------| ------------- | ------------- |
**Number of epochs**| 3  | 10  |
**Number of trained params**| 66,364,418  | 1,538  | 
**Time to train**| 3m 49s  | 2m 19s  |
**GPU memory usage during training**| 4.6 GB  | 1 GB  |
**Initial loss**| 2.29  | 3.7  |
**Final scores**| 'exact_match': 50.2, 'f1': 63.0  | 'exact_match': 12.1, 'f1': 18.6  |


#### Discussion
Although full fine-tuning approach in this case appears to have way better results, LoRA approach has it's own advantages. Keep in mind that very little time was allocated to find an optimal set of parameters for the LoRA training. We see a big difference between the initial losses of the models. This is normal due to the new parameters introduced in the model. Also the fact that the model converges so fast in the full fine-tuning approach means that it saw "similar" data to the current training data, so the fact that we had to stop after only 3 epochs is more like a happy/lucky outcome. In practice, you may end up needing to train the model for a significant more amount of time. We can also observe the difference in the resources needed for the two approaches, LoRA taking a significant lower amount of GPU memory while training, this can be a decisive factor when training large LLMs, keep in mind that DistilBERT is already a lightweight LLM.