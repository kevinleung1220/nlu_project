# Code from https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/

import numpy as np
import pandas as pd
from nltk.util import ngrams
import collections
import itertools
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.optimizers import Adam


def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding = 'utf8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


def build_language_model(X_data, y_data, embedding_size, LSTM_size, learning_rate, dropout_prob, vocab_size):
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size, input_length=X_data.shape[1]))
	model.add(Dropout(dropout_prob))
	model.add(LSTM(LSTM_size))
	model.add(Dense(vocab_size, activation='softmax'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
	# fit model
	model.fit(X_data, y_data, epochs=100, verbose=2, validation_split=.1)


def main():

	# Load data file and convert to lowercase
	in_filename = 'Data/GlobalVoices_en.txt'
	raw_text = load_doc(in_filename)
	raw_text = raw_text.lower()
	lines = raw_text.split('\n')

	# integer encode sequences of characters
	# hyperparam for the length of the input sequences
	length = 10

	# Generate dataset for character-level language modeling
	sequences = list()
	k = 0
	j = 0
	for line in lines: 
		if j < 25000:
			k += 1
			if line[-1:] == '.' or line[-1:] == '?' or line[-1:] == '!' or line[-1:] == '"':
				j += 1
				for i in range(length, len(line)):
					seq = raw_text[i-length:i+1]
					sequences.append(seq)

	print('Length of Original Corpus: %s' % len(lines))
	print('Length of Tokenized Sequences in LM Corpus: %s' % len(sequences))
	print('Sentence line we stop at: %s' % k)
	print('Number of Sentences in our dataset: %s' % j)

	# Create map for character -> integer
	chars = sorted(list(set(raw_text)))
	chars = chars[0:66]
	mapping = dict((c, i) for i, c in enumerate(chars))
	print(mapping)

	# Encode characters as integers
	encoded_sequences = list()
	for row in sequences:
	    # integer encode line; any characters outside of the first 66 are encoded as something else
		encoded_seq = [mapping[char] if char in chars else 66 for char in row ]
		# store
		encoded_sequences.append(encoded_seq)

	# Character-level vocabulary size
	vocab_size = len(mapping) + 1
	print('Vocabulary Size: %d' % vocab_size)

	# separate into input and output
	encoded_sequences = np.array(encoded_sequences) #, shape = (len(sequences),length))
	X, y = encoded_sequences[:,:-1], encoded_sequences[:,-1]
	#new_sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
	#X = np.array(new_sequences)
	y_temp = to_categorical(y, num_classes=vocab_size)
	
	build_language_model(X[:1000], 
						 y_temp[:1000], 
						 embedding_size=16, 
						 LSTM_size=256, 
						 learning_rate=.001, 
						 dropout_prob=.2, 
						 vocab_size=vocab_size)


if __name__ == "__main__":
	main()