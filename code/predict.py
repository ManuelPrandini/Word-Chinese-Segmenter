from argparse import ArgumentParser
import model as md
import os
import preprocess as pre
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()

def result_from_prediction(model,test_x_uni,test_x_bi):
    '''
    Method used to return the result of the input file prediction 
    :param test_x_uni the unigrams of the input file
    :param test_x_bi the bigrams of the input file
    :return an array containing the result of the prediction in the BIES format
    '''
    BIES = {0:'B', 1:'I', 2:'E', 3:'S'}
    result = []                  
    for i in range(len(test_x_uni)):
        predictions = model.predict([[test_x_uni[i]],[test_x_bi[i]]])
        s = ""
        pred = predictions.argmax(-1)[0]
        for p in pred:
            s+=BIES[p]
        predict_t = "".join(s)
        result.append(predict_t)
    return result

def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    
    #load the training file with the sentences of all the four sub-dataset to make the vocab 
    concat = pre.read_file(os.path.join(resources_path,"tensor_concat_train.utf8"))
    #create the vocab with the concatenation of all training files
    vocab_unigrams = pre.make_vocab(concat,1)
    vocab_bigrams = pre.make_vocab(concat,2)

    #load the test file from input
    test_tensor_lines = pre.read_file(input_path)

    #convert the input array into index 
    test_x_uni = pre.word_to_index(test_tensor_lines,vocab_unigrams,1)
    test_x_bi = pre.word_to_index(test_tensor_lines,vocab_bigrams,2)


    #DEFINE SOME COSTANTS
    VOCAB_SIZE_UNI = len(vocab_unigrams)
    VOCAB_SIZE_BI = len(vocab_bigrams)
    CHAR_EMBEDDING_SIZE = 32
    BIGRAM_EMBEDDING_SIZE = [16, 32, 64]
    LEARNING_RATE = [0.04, 0.035, 0.03, 0.02,0.009]
    HIDDEN_SIZE = 256
    INPUT_DROPOUT = [0,0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    LSTM_DROPOUT = [0,0.1, 0.2, 0.3, 0.4]

    #BUILD THE MODEL
    model = md.create_keras_model_parallel(VOCAB_SIZE_UNI,VOCAB_SIZE_BI,CHAR_EMBEDDING_SIZE,BIGRAM_EMBEDDING_SIZE[1]
                                            ,HIDDEN_SIZE,INPUT_DROPOUT[2],LSTM_DROPOUT[2])
    print("Load the weights...")                                        
    #load the better weights
    model.load_weights(os.path.join(resources_path,"weights.hdf5"))
    print("Predict...")
    #calculate the result from prediction
    result = result_from_prediction(model,test_x_uni,test_x_bi)
    print("Save the result on ",output_path)
    #create the output file with result from prediction
    pre.create_file(output_path,result)
    print("Done!")
    


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
