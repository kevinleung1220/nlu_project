# Code from https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/

import numpy as np
import pandas as pd
from nltk.util import ngrams
import collections
import itertools
import matplotlib.pyplot as plt
import h5py
import os
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences


# Set tf flags to be used with bash files in order to adjust hyperparameters
tf.app.flags.DEFINE_string('language', 'English', 'Language that our model is for')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('drop_rate', 0.0, 'Dropout rate when training.')
tf.app.flags.DEFINE_integer('input_size', 10, 'Num of Inputs')
tf.app.flags.DEFINE_integer('embedding_size', 16, 'Size of embedding layer')
tf.app.flags.DEFINE_integer('LSTM_size', 256, 'Size of LSTM layer')
tf.app.flags.DEFINE_integer('GPU_num', 1, 'Which GPU is used')
#tf.app.flags.DEFINE_string('file_path', '/data/denizlab/OA_dataset/MRI Models/', 'Main Folder to Save outputs')

FLAGS = tf.app.flags.FLAGS

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


def build_language_model(X_train, 
						 y_train,
						 X_val,
						 y_val, 
						 embedding_size, 
						 LSTM_size, 
						 learning_rate, 
						 dropout_prob, 
						 vocab_size, 
						 path):
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size, input_length=X_train.shape[1]))
	model.add(Dropout(dropout_prob))
	model.add(LSTM(LSTM_size))
	model.add(Dense(vocab_size, activation='softmax'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
	# fit model
	# Early Stopping callback that can be found on Keras website
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
    
	# Create path to save weights with model checkpoint
	weights_path = path + '/weights-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5' #'best_weights.hdf5'
	model_checkpoint = ModelCheckpoint(weights_path, monitor = 'val_loss', save_best_only = True, save_weights_only=True,
                                       verbose=0, period=1)

	# Save loss and accuracy curves using Tensorboard
	tensorboard_callback = TensorBoard(log_dir = path, 
										histogram_freq = 0, 
										write_graph = False, 
										write_grads = False, 
										write_images = False)
										#batch_size = batch_size)

	callbacks_list = [early_stopping, model_checkpoint, tensorboard_callback]
	model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks_list)


def main(argv=None):

	# Load data file and convert to lowercase
	in_filename = 'Data/GlobalVoices_en.txt' # change this file for other languages
	raw_text = load_doc(in_filename)
	raw_text = raw_text.lower()
	lines = raw_text.split('\n')

	# hyperparam for the length of the input sequences
	#length = 10

	# Number of sentences in training and validation sets
	train_size = 75000
	val_size = 7500

	# Generate dataset for character-level language modeling
	train_sequences = list()
	val_sequences = list()
	k = 0 # index of lines that we iterate over
	t = 0 # index for where the training set ends
	j = 0 # index for where test set ends

	# Iterate over each line of input document
	for line in lines: 

		# Build training set
		if j < train_size:
			k += 1
			# End of Sentence tokens
			if line[-1:] == '.' or line[-1:] == '?' or line[-1:] == '!' or line[-1:] == '"':
				j += 1
				t = j
				# Break sentence into list of characters of the specified length
				for i in range(1, len(line)):
					if i < FLAGS.input_size:
						seq = line[0:i+1]
					else:
						seq = line[i-FLAGS.input_size:i+1]
					train_sequences.append(seq)

		# Build validation set
		elif j >= train_size and j < (train_size + val_size):

			if j == train_size: print('Training Set Created')
			k += 1
			# End of Sentence tokens
			if line[-1:] == '.' or line[-1:] == '?' or line[-1:] == '!' or line[-1:] == '"':
				j += 1
				# Break sentence into list of characters of the specified length
				for i in range(1, len(line)):
					if i < FLAGS.input_size:
						val_seq = line[0:i+1]
					else:
						val_seq = line[i-FLAGS.input_size:i+1]

					val_sequences.append(seq)

	print('Validation Set Created')

	print('Length of Original Corpus: %s' % len(lines))
	print('Length of Tokenized Sequences in train Corpus: %s' % len(train_sequences))
	print('Length of Tokenized Sequences in val Corpus: %s' % len(val_sequences))
	print('Number of Sentences in train set: %s' % t)
	print('Number of Sentences in validation set: %s' % (j-t))

	# Create map for character -> integer
	chars = sorted(list(set(raw_text)))
	# For English we only take the top 256 characters (there are way too many because of the presence of other languages)
	chars = chars[0:66] # change this line for other languages
	mapping = dict((c, i) for i, c in enumerate(chars))
	print('Character mapping created:')
	print(mapping)

	# Encode characters as integers
	encoded_train_sequences = list()
	for row in train_sequences:
	    # integer encode line; any characters outside of the first 256 are encoded as something else
		encoded_seq = [mapping[char] if char in chars else 66 for char in row ] # change this line for other languages
		# store
		encoded_train_sequences.append(encoded_seq)

	print('Training Set Encoded')

	encoded_val_sequences = list()
	for row in val_sequences:
	    # integer encode line; any characters outside of the first 256 are encoded as something else
		encoded_seq = [mapping[char] if char in chars else 66 for char in row ] # change this line for other languages
		# store
		encoded_val_sequences.append(encoded_seq)

	print('Validation Set Encoded')

	# Character-level vocabulary size
	vocab_size = len(mapping) + 1
	print('Vocabulary Size: %d' % vocab_size)

	max_len = FLAGS.input_size + 1

	encoded_train_sequences = pad_sequences(encoded_train_sequences, max_len)
	encoded_val_sequences = pad_sequences(encoded_val_sequences, max_len)

	# separate into input and output
	encoded_train_sequences = np.array(encoded_train_sequences) #, shape = (len(sequences),length))
	X_train, y_train = encoded_train_sequences[:,:-1], encoded_train_sequences[:,-1]
	y_train = to_categorical(y_train, num_classes=vocab_size)

	encoded_val_sequences = np.array(encoded_val_sequences) #, shape = (len(sequences),length))
	X_val, y_val = encoded_val_sequences[:,:-1], encoded_val_sequences[:,-1]
	y_val = to_categorical(y_val, num_classes=vocab_size)

	path = os.getcwd()
	#print(path)
	# Generate path to save weights
	model_path = path + '/%s_lr%s_dr%s_inputsize%s_embeddingsize%s_LSTMsize%s_gpu%s/' % (FLAGS.language, FLAGS.learning_rate, FLAGS.drop_rate, FLAGS.input_size, FLAGS.embedding_size, FLAGS.LSTM_size, FLAGS.GPU_num)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	#print(model_path)
	build_language_model(X_train=X_train, 
						 y_train=y_train,
						 X_val = X_val,
						 y_val = y_val, 
						 embedding_size=FLAGS.embedding_size, 
						 LSTM_size=FLAGS.LSTM_size, 
						 learning_rate=FLAGS.learning_rate, 
						 dropout_prob=FLAGS.drop_rate, 
						 vocab_size=vocab_size,
						 path = model_path)


if __name__ == "__main__":
	tf.app.run()

