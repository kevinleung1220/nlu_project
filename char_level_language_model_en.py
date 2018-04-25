# Code from https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/

import numpy as np
import pandas as pd
from nltk.util import ngrams
import collections
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ModelCheckpoint


# Set tf flags to be used with bash files in order to adjust hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('drop_rate', 0.0, 'Dropout rate when training.')
tf.app.flags.DEFINE_integer('input_size', 10, 'Num of Inputs')
tf.app.flags.DEFINE_integer('embedding_size', 16, 'Size of embedding layer')
tf.app.flags.DEFINE_integer('LSTM_size', 256, 'Size of LSTM layer')
tf.app.flags.DEFINE_integer('GPU_num', 1, 'Which GPU is used')
tf.app.flags.DEFINE_string('file_path', '/data/denizlab/OA_dataset/MRI Models/', 'Main Folder to Save outputs')


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


def build_language_model(X_data, y_data, embedding_size, LSTM_size, learning_rate, dropout_prob, vocab_size, path):
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size, input_length=X_data.shape[1]))
	model.add(Dropout(dropout_prob))
	model.add(LSTM(LSTM_size))
	model.add(Dense(vocab_size, activation='softmax'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
	# fit model
	# Early Stopping callback that can be found on Keras website
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    
	# Create path to save weights with model checkpoint
	weights_path = path + 'weights-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5' #'best_weights.hdf5'
	model_checkpoint = ModelCheckpoint(weights_path, monitor = 'val_loss', save_best_only = True, save_weights_only=True,
                                       verbose=0, period=1)

	# Save loss and accuracy curves using Tensorboard
	tensorboard_callback = TensorBoard(log_dir = path, 
										histogram_freq = 0, 
										write_graph = False, 
										write_grads = False, 
										write_images = False,
										batch_size = batch_size)

	callbacks_list = [early_stopping, model_checkpoint, tensorboard_callback]
	model.fit(X_data, y_data, epochs=100, verbose=2, validation_split=.1)


def main():

	# Load data file and convert to lowercase
	in_filename = 'GlobalVoices_en.txt'
	raw_text = load_doc(in_filename)
	raw_text = raw_text.lower()
	lines = raw_text.split('\n')

	# integer encode sequences of characters
	# hyperparam for the length of the input sequences
	length = 10
	num_of_sentences = 250000

	# Generate dataset for character-level language modeling
	sequences = list()
	k = 0
	j = 0
	for line in lines: 
		if j < num_of_sentences:
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
	chars = chars[0:256]
	mapping = dict((c, i) for i, c in enumerate(chars))
	print(mapping)

	# Encode characters as integers
	encoded_sequences = list()
	for row in sequences:
	    # integer encode line; any characters outside of the first 66 are encoded as something else
		encoded_seq = [mapping[char] if char in chars else 256 for char in row ]
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
	

	model_path = FLAGS.file_path + '%s_lr%s_dr%s_inputsize%s_embeddingsize%s_LSTMsize%s_gpu%s/' % (FLAGS.learning_rate, FLAGS.drop_rate, FLAGS.input_size, FLAGS.embedding_size, FLAGS.LSTM_size, FLAGS.GPU_num)
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	build_language_model(X, 
						 y_temp, 
						 embedding_size=FLAGS.embedding_size, 
						 LSTM_size=FLAGS.LSTM_size, 
						 learning_rate=FLAGS.learning_rate, 
						 dropout_prob=FLAGS.drop_rate, 
						 vocab_size=vocab_size,
						 path = model_path)


if __name__ == "__main__":
	main()

