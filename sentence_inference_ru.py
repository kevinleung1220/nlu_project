import csv
import pandas as pd
import h5py
import numpy as np
import collections

#from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Embedding, Dropout
#from keras.optimizers import Adam
#from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ModelCheckpoint
#from keras.preprocessing.sequence import pad_sequences


'''
    Function to load input file for the sentence permutations
    
    Params:
        file = path of input csv
    Returns:
        data = csv formatted as data frame
'''

def convert_permutations_to_list(file):
    data = [] # create empty array to fill with data
    with open(file, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',') 
        for row in spamreader: 
            data.append(row) 
    data = pd.Series(data)
    return data


'''
    Loads corpus file
    
    Params:
        filename = path of input txt file
    
    Returns:
        text = opened txt file
'''
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding = 'utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


'''
    Create language model architecture
    
    Params: self-explanatory
    
    Returns: the language model
'''

def build_language_model(input_length,
                         embedding_size,
                         LSTM_size,
                         dropout_prob, 
                         vocab_size,
                         learning_rate):
    
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_length))
    model.add(Dropout(dropout_prob))
    model.add(LSTM(LSTM_size))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    return model


'''
    Loads model weights into model architecture
    
    Params:
        model = model architecture
        weights_path = hdf5 file path for weights
    
    Returns:
        model with trained weights
'''
def load_trained_model(model, weights_path):
    model.load_weights(weights_path)
    return model


'''
    Encodes scrambled sentences using the integer map
    
    Params:
        data = input data with 2 columns: 1st column = true sentence, 2nd column = all permutations
        mapping = character mapping
        chars = list of characters in the mapping
'''

def encode_scrambled_sentences(data, mapping, chars):
    # Create empty lists for the original and scrambled sentences
    encoded_label = []
    encoded_scrambled_sentences = []
    
    # Iterate over the data
    for i in range(len(data)):
        # Encode the original sentence to integers 
        labeled_sentence = data.iloc[i,0]
        encoded_sentence = [mapping[char] if char in chars else 90 for char in labeled_sentence]
        encoded_label.append(encoded_sentence)
        # List for the scrambled sentences
        filler = []
        scrambled_sentences = data.iloc[i,1]
        if scrambled_sentences[3] != '':
            for j in scrambled_sentences:
                # remove the parentheses
                if len(j) > 0:
                    if j[0] == "\"": 
                        j = j[1:] 
                    if j[-1] == "\"":
                        j = j[:-1]
                    # Encode each scrambled sentence
                    encoded = [mapping[char] if char in chars else 90 for char in j]
                    filler.append(encoded)
        encoded_scrambled_sentences.append(filler)
    # Convert to pandas dataframe
    encoded_label = pd.Series(encoded_label)
    encoded_scrambled_sentences = pd.Series(encoded_scrambled_sentences)
    encoded_data = pd.concat([encoded_label, encoded_scrambled_sentences], axis=1)   
    return encoded_data


'''
    Computes the log probability of each sequence
    
    Params:
        model = model with trained weights
        encoded_data = data frame for the encoded sentences and scrambled BoNgram sentences
    
    Returns:
        probabilities = data frame of log probabilities for each sequence
'''
'''
    Computes the log probability of each sequence
    
    Params:
        model = model with trained weights
        encoded_data = data frame for the encoded sentences and scrambled BoNgram sentences
    
    Returns:
        probabilities = data frame of log probabilities for each sequence
'''
def get_probability(model, encoded_data, num):
    # Create lists for the original sentence probability and scrambled sentence probabilities
    true_probability = []
    scrambled_probability_all = []
    
    # Iterate over each row in the encoded data frame
    for i in range(len(encoded_data)):
        print(str(num) + '-gram: True Sentence ' + str(i))
        prob = 0
        sentence = encoded_data.iloc[i, 0]
        sentence_scrambled_prob = []
        if len(encoded_data.iloc[i, 1]) != 0:
            for j in range(1, len(sentence)):
                if j < 100:
                    seq = np.array([sentence[0:j]])
                    seq = pad_sequences(seq, 100, value=25)
                else:
                    seq = np.array([sentence[(j-100):j]])
                prob += np.log(model.predict(seq)[0,sentence[j]])
            t = 0
            for k in encoded_data.iloc[i, 1]:
                print(str(num) + '-gram: Sentence ' + str(i) + ' Permutation number ' + str(t))
                t += 1
                temp_prob = 0
                for m in range(1, len(k)):
                    if m < 100:
                        temp_seq = np.array([k[0:m]])
                        temp_seq = pad_sequences(temp_seq, 100, value=25)
                    else:
                        temp_seq = np.array([k[(m-100):m]])
                    temp_prob += np.log(model.predict(temp_seq)[0,k[m]])
                sentence_scrambled_prob.append(temp_prob)

        true_probability.append(prob)
        scrambled_probability_all.append(sentence_scrambled_prob)
        
    true_probability = pd.Series(true_probability)
    scrambled_probability_all = pd.Series(scrambled_probability_all)
    probabilities = pd.concat([true_probability, scrambled_probability_all], axis=1)     
    
    return probabilities


if __name__ == '__main__':
    # Read in original sentences
    true_sentences = pd.read_csv('russian_test.csv', encoding='utf-8')
    print('True Sentences Loaded')
    # Load txt file
    in_filename = 'Data/GlobalVoices_ru.txt' # change this file for other languages
    raw_text = load_doc(in_filename)
    raw_text = raw_text.lower() # convert to lowercase

    chars = []
    for i in collections.Counter(raw_text).most_common()[:90]:
        chars.append(i[0])
    mapping = dict((c, i) for i, c in enumerate(chars))
    print('Character mapping created:')
    #print(mapping)

    missing_val = len(mapping)
    print(missing_val)

    '''rus_model_v1 = build_language_model(input_length=100, 
                                        embedding_size = 16,
                                        LSTM_size=256, 
                                        dropout_prob=.2, 
                                        vocab_size=len(mapping)+1,
                                        learning_rate =.0001)
    print('Language Model Built')
    new_model = load_trained_model(rus_model_v1, 
                                   'rus_model//weights-05-1.372-0.000.hdf5')
    print('Language Model Loaded')

    
    permute_9 = convert_permutations_to_list('permutations/russian_permutations_9.csv')
    print('Permutation dataset created')
    data_for_inference_9 = pd.concat([true_sentences, permute_9], axis=1)
    permute_9_encode = encode_scrambled_sentences(data_for_inference_9, mapping, chars)
    print('Permutation dataset encoded')
    permute_9_prob = get_probability(new_model, permute_9_encode.iloc[:500,:], 9)
    permute_9_prob.to_csv('probabilities/new_prob/ru_probabilities_9gram.csv')


    permute_8 = convert_permutations_to_list('permutations/russian_permutations_8.csv')
    print('Permutation dataset created')
    data_for_inference_8 = pd.concat([true_sentences, permute_8], axis=1)
    permute_8_encode = encode_scrambled_sentences(data_for_inference_8, mapping, chars)
    print('Permutation dataset encoded')
    permute_8_prob = get_probability(new_model, permute_8_encode.iloc[:500,:], 8)
    permute_8_prob.to_csv('probabilities/new_prob/ru_probabilities_8gram.csv')
    
    permute_9_prob = get_probability(new_model, permute_9_encode.iloc[500:1000,:], 9)
    permute_9_prob.to_csv('probabilities/new_prob/ru_probabilities_9gram2.csv')

    permute_8_prob = get_probability(new_model, permute_8_encode.iloc[500:1000,:], 8)
    permute_8_prob.to_csv('probabilities/new_prob/ru_probabilities_8gram2.csv')'''


    '''permute_7 = convert_permutations_to_list('permutations/russian_permutations_7.csv')
    print('Permutation dataset created')
    data_for_inference_7 = pd.concat([true_sentences, permute_7], axis=1)
    permute_7_encode = encode_scrambled_sentences(data_for_inference_7, mapping, chars)
    print('Permutation dataset encoded')
    permute_7_prob = get_probability(new_model, permute_7_encode.iloc[:500,:], 7)
    permute_7_prob.to_csv('C:/Users/kl2596/Google Drive/Mathematics and Data Science Classes/Natural Language Understanding/Project/Code/probabilities/new_prob/ru_probabilities_7gram.csv')


    permute_6 = convert_permutations_to_list('permutations/russian_permutations_6.csv')
    print('Permutation dataset created')
    data_for_inference_6 = pd.concat([true_sentences, permute_6], axis=1)
    permute_6_encode = encode_scrambled_sentences(data_for_inference_6, mapping, chars)
    print('Permutation dataset encoded')
    permute_6_prob = get_probability(new_model, permute_6_encode.iloc[:500,:], 6)
    permute_6_prob.to_csv('C:/Users/kl2596/Google Drive/Mathematics and Data Science Classes/Natural Language Understanding/Project/Code/probabilities/new_prob/ru_probabilities_6gram.csv')


    permute_7_prob = get_probability(new_model, permute_7_encode.iloc[500:1000,:], 7)
    permute_7_prob.to_csv('C:/Users/kl2596/Google Drive/Mathematics and Data Science Classes/Natural Language Understanding/Project/Code/probabilities/new_prob/ru_probabilities_7gram2.csv')

    permute_6_prob = get_probability(new_model, permute_6_encode.iloc[500:1000,:], 6)
    permute_6_prob.to_csv('C:/Users/kl2596/Google Drive/Mathematics and Data Science Classes/Natural Language Understanding/Project/Code/probabilities/new_prob/ru_probabilities_6gram.csv')

    
    permute_5 = convert_permutations_to_list('permutations/russian_permutations_5.csv')
    print('Permutation dataset created')
    data_for_inference_5 = pd.concat([true_sentences, permute_5], axis=1)
    permute_5_encode = encode_scrambled_sentences(data_for_inference_5, mapping, chars)
    print('Permutation dataset encoded')
    permute_5_prob = get_probability(new_model, permute_5_encode.iloc[:1000,:], 5)
    permute_5_prob.to_csv('probabilities/ru_probabilities_5gram.csv')


    permute_4 = convert_permutations_to_list('permutations/russian_permutations_4.csv')
    print('Permutation dataset created')
    data_for_inference_4 = pd.concat([true_sentences, permute_4], axis=1)
    permute_4_encode = encode_scrambled_sentences(data_for_inference_4, mapping, chars)
    print('Permutation dataset encoded')
    permute_4_prob = get_probability(new_model, permute_4_encode.iloc[:1000,:], 4)
    permute_4_prob.to_csv('probabilities/ru_probabilities_4gram.csv')   
    '''