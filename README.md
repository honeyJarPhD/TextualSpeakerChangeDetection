# TextualSpeakerChangeDetection

*************************************
Prerequisities:
*************************************
1. Python 3.5 (or higher)
2. Python Dependencies: PyTorch (torch), pandas, numpy, gensim, sklearn
3. Pre-trained word-embedding model

*************************************
Running:
*************************************
1. Raw-Data Conversion: Run data_to_pkl.py. 
It will create a pickle file (data_to_vectors_conversion_df.pkl) that contains a dataframe, 
followed by each file_id of your dataset, and the conversion of each sliding window to a feature vector.

2. Vectors Creation - Run vectors_creator.py.
The pickle file from previous step would be transformed into a list of vectors (stored in a dataframe),
so that each word is converted to a 300-dimensional word wmbedding, using the pre-trained word-embedding model.

3. Training - Run train.py.
The training process is based on a Neural Network infrasturcture, that can be found at nn_definition.py.
One can control the batch size, as well as other hyper-parameters such as learning rate, steps, optimizer, etc (nn_definition.py).
Every K steps (a pre-defined number), a check-point is saved to the "Models" folder (must be created a-priorily)

4. Testing - Run test_models.py.
The testing process reads a set of models that were created by the train.py script, 
and then calculates Precision, Recall and F1-Score measures for each model. 


*************************************
Citations:
*************************************
Our paper is cited as:

@misc{anidjar2020thousand,
        title={A Thousand Words are Worth More Than One Recording: NLP Based Speaker Change Point Detection},
        author={Anidjar, Or Haim and Hajaj, Chen and Dvir, Amit and Gilad, Issachar},
        journal={arXiv preprint arXiv:2006.01206},
        year={2020}
}
