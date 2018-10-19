
from keras.models import Model
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,RepeatVector, TimeDistributed, Bidirectional
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from importlib import reload

num_words = 2000

input_tokenizer = Tokenizer(num_words=num_words, lower=False, split=' ', filters='')
input_tokenizer.fit_on_texts(tokenized_xtext)
input_word_index = input_tokenizer.word_index
target_tokenizer = Tokenizer(num_words=num_words, lower=False, split=' ', filters='')
target_tokenizer.fit_on_texts(tokenized_ytext)
target_word_index = target_tokenizer.word_index
text_tokenizer = Tokenizer(num_words=num_words, lower=False, split=' ', filters='')
text_tokenizer.fit_on_texts(tokenized_xtext + tokenized_ytext)
word_index = text_tokenizer.word_index
print(len(word_index))
num_encoder_tokens =len(input_word_index)
num_decoder_tokens = len(target_word_index)

print(num_encoder_tokens)
print(num_decoder_tokens)
max_encoder_seq_length = max([len(txt) for txt in tokenized_xtext])
max_decoder_seq_length = max([len(txt) for txt in tokenized_ytext])

max_len = max(max_encoder_seq_length, max_decoder_seq_length)

#print(len(tokenized_xtext))

encoder_input_data = pad_sequences(input_tokenizer.texts_to_sequences(tokenized_xtext), maxlen=max_len)
decoder_input_data = pad_sequences(target_tokenizer.texts_to_sequences(tokenized_ytext), maxlen=max_len)
decoder_target_data = np.zeros(
    (len(tokenized_ytext), max_len, num_decoder_tokens),
    dtype='float32')


for i, target_text in enumerate(tokenized_ytext):
   # print(i,target_text)
    for t, word in enumerate(target_text):
        #print(i,word)
#        w2idx = 0
#        if word in target_word_index:
#            w2idx = target_word_index[word]
#        decoder_input_data[i, t, w2idx] = 1
        if t>0:
            decoder_target_data[i, t-1, target_word_index[word] -1] = 1
  

max_nb_words  = 2000
embedding_dim = 300
validation_split = 0.2
vocab_size = len(word_index) + 1

#encoder_input_data.shape

nb_words = max(max_nb_words, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, embedding_dim))

for word, i in word_index.items():
    print(word_index.items())
    if word in word2Vecmodel.wv.vocab:
        #print(word)
        embedding_matrix[i] = word2Vecmodel[word]
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis = 1) == 0))

embedding_layer = Embedding(input_dim = nb_words, output_dim = embedding_dim, trainable = True)
embedding_layer.build((None,))
embedding_layer.set_weights([embedding_matrix])

encoder_inputs = Input(shape = (None,))
encoder_input_embeddings = embedding_layer(encoder_inputs)
encoder_lstm_layer =LSTM(embedding_dim, return_state = True)
encoder_lstm, state_h, state_c = encoder_lstm_layer(encoder_input_embeddings)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape = (None,))
decoder_input_embeddings = embedding_layer(decoder_inputs)
decoder_lstm_layer = LSTM(embedding_dim, return_state = True, return_sequences = True)
decoder_lstm,_,_ = decoder_lstm_layer(decoder_input_embeddings,initial_state = encoder_states)
decoder_dense_layer = Dense(num_decoder_tokens, activation = 'softmax')
output = decoder_dense_layer(decoder_lstm)

dense = TimeDistributed(Dense(vocab_size, activation = 'softmax'))

model = Model([encoder_inputs,decoder_inputs],output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit([encoder_input_data, decoder_input_data],decoder_target_data, epochs = 1000, batch_size = 128)

#decoder model

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape =(embedding_dim,))
decoder_state_input_c = Input(shape = (embedding_dim,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs,state_h,state_c = decoder_lstm_layer(decoder_input_embeddings, initial_state =
                                               decoder_input_states)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense_layer(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_input_states,
                      [decoder_outputs] + decoder_states)

#model visualization


from keras.utils.vis_utils import plot_model
plot_model(model, to_file = 'model.png',show_shapes = True, show_layer_names = True)

from IPython.display import Image
Image('model.png')

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


