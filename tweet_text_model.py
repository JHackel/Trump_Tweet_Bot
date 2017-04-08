import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Masking
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

with open('trimmed_tweets.txt', 'r') as myfile:
    text=myfile.read()

# to_delete = chars[90:]
# for char in to_delete:
#     text = text.replace(char,"")
# text = text.replace('/n','\n')
# with open('trimmed_tweets.txt','w') as text_file:
#     text_file.write(text)

# create mapping of unique chars to integers
text_as_list = text.split("\n")
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# Padding preparation
pad_char = '{'
max_len = len(max(text_as_list,key=len))
print(max_len)
text_as_list = text_as_list[:-1]
text_as_list = [pad_char * (max_len - len(tweet)) + tweet for tweet in text_as_list]
char_to_int[pad_char] = n_vocab
int_to_char[n_vocab] = pad_char

seq_length = max_len // 2
dataX = []
dataY = []
for tweet in text_as_list:
    for i in range(0,len(tweet) - seq_length, 1):
        seq_in = tweet[i:i + seq_length]
        if(seq_in.replace(pad_char,'') == ''):
            continue
        seq_out = tweet[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(Masking(mask_value=1., input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

def train(epochs=10,weights_file_to_load=None):
    if(weights_file_to_load):
        model.load_weights(weights_file_to_load)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # define the checkpoint
    file_path = "2-lstm-layer/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs=epochs, batch_size=128, callbacks=callbacks_list)

def generate_text(file_path,number_of_tweets=5):
    # load the network weights
    model.load_weights(file_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # generate characters
    for j in  range(number_of_tweets):
        tweet_i = np.random.randint(0, len(dataX) - 1)
        seed = dataX[tweet_i]
        print("Seed: \"", ''.join([int_to_char[value] for value in seed]), "\"")
        output = ""
        for i in range(140):
            x = np.reshape(seed, (1, len(seed), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_to_char[index]
            output += result
            seed.append(index)
            seed = seed[1:len(seed)]
        print(output,"\n")
    print("Done.")

train(50)
# generate_text("2-lstm-layer/weights-improvement-00-2.1947.hdf5",100)
