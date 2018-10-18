
from keras.models import Model
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,RepeatVector, TimeDistributed, Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from importlib import reload


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


