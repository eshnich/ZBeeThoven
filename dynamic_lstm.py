import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import music21 as m21
import parse
import random
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

# ---------- PARAMETERS ----------
switch = False
irish = True

# network architecture
peephole = False
dropout = False
dropoutkeepprob =0.5
num_layers = 2

# hyperparameters
learning_rate = 0.01
sample_length = 48
lstm_size = 128

# data
beat = False
transposetoc = False

# training
num_epochs = 1000
batch_size = 1

# validation
validate = True
kfold_k = 2


# ---------- LOADING DATA ----------
print('Loading data')

train_data = []
dictionary_data = []

count = 0
if irish:
    for fn in os.listdir('Connolly_MusicMID/'):
        if count >= 10000:
            break
        count += 1
        data = parse.parse_music('Connolly_MusicMID/{}'.format(fn),transpose_to_c=transposetoc, include_beat=beat)
        if data == None:
            continue
        train_data.append(data)
        dictionary_data.extend(data)
else:
    for fn in os.listdir('beethoven_midis/'):
        if fn[-4:] == '.mid':
            t_data, d_data = parse.parse_music('beethoven_midis/{}'.format(fn), irish=False, transpose_to_c=transposetoc, include_beat=beat)
            train_data.extend(t_data)
            dictionary_data.extend(d_data)
print('Finished loading data')

print('Building vec_to_num and num_to_vec')

# hash which converts (note, duration) pairs to indices and vice versa
vec_to_num,num_to_vec = parse.build_dataset(dictionary_data)

# vec_size is basically the length of the input vectors, and out_size is the length of the output vectors
vocab_size = len(vec_to_num)
vec_size = vocab_size
out_size = vec_size
print('Finished building dictionaries')


# ---------- HELPER FUNCTIONS ----------
def make_feature_vec(point):
    vec = np.zeros(vec_size)
    vec[point] = 1.0
    return vec

def sftmax(z):
    ez = np.exp(z-np.max(z))
    return ez/ez.sum()

def get_a_cell(n_hidden):
    cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True,use_peepholes=peephole)
    if dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropoutkeepprob)
    return cell

def get_random_track(t):
    # chooses a random track in train_seq
    n = len(t)
    while True:
        seed = random.randint(0, n - 1)
        #print(seed)
        voice_track = t[seed]
        if len(voice_track) > sample_length:
            return [make_feature_vec(voice_track[i]) for i in range(len(voice_track))]

# convert to music
stream = m21.stream.Stream()

#model input and output
x = tf.placeholder("float", [batch_size,None,vec_size])
y = tf.placeholder("float", [batch_size,None,out_size]) #one-hot vector


# ---------- INITIALIZING THE LSTM ----------
print('Initializing the LSTM')

if switch:
    lstm = tf.contrib.rnn.MultiRNNCell([get_a_cell(lstm_size) for i in range(num_layers)],state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell=lstm,inputs=x,dtype=tf.float32)
    W = tf.Variable(tf.random_normal([lstm_size, out_size]))
    b = tf.Variable(tf.zeros([out_size]))
    outputs = tf.reshape(outputs,[-1,lstm_size])
    logits = tf.matmul(outputs,W)+b
    logits = tf.reshape(logits,[batch_size,-1,out_size])
else:
    W = {'out':tf.Variable(tf.random_normal([lstm_size, out_size]))}
    b = {'out':tf.Variable(tf.zeros([out_size]))}
    def RNN(x,W,b):
        lstm = tf.contrib.rnn.MultiRNNCell([get_a_cell(lstm_size) for i in range(num_layers)],state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cell=lstm,inputs=x, dtype=tf.float32)
        outputs = tf.reshape(outputs,[-1,lstm_size])
        return tf.matmul(outputs,W['out'])+b['out']
    logits = RNN(x,W,b)
    logits = tf.reshape(logits,[batch_size,-1,out_size])

softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_sum(softmax)
tf.summary.scalar('Loss',loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

train_seq = []
for voice in train_data:
    voice_track = []
    for v in voice:
        voice_track += [vec_to_num[v]]
    train_seq.append(voice_track)


# ---------- TESTING THE LSTM ----------
print('Testing the LSTM')

if switch:
    start_vec = tf.placeholder(tf.float32,[1, 1, vec_size])
    state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, batch_size, lstm_size])
    l = tf.unstack(state_placeholder, axis=0)
    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])for idx in range(num_layers)])
    output, new_state = tf.nn.dynamic_rnn(cell=lstm,inputs = start_vec,initial_state=rnn_tuple_state,dtype=tf.float32)
    output = tf.reshape(output,[1,lstm_size])
    out_logits = tf.matmul(output,W) + b

merged = tf.summary.merge_all() 
init = tf.global_variables_initializer()

# ---------- TRAINING ----------
print('Starting training')

with tf.Session() as session:
    kf = KFold(n_splits=kfold_k,shuffle=True)
    valid_loss_all = []
    for train_index, valid_index in kf.split(train_seq):
        session.run(init)
        writer = tf.summary.FileWriter("output",session.graph)

        if validate:
            training = [train_seq[i] for i in train_index]
            validation = [[make_feature_vec(j) for j in train_seq[i]] for i in valid_index]
        else:
            training = train_seq

        iterations = []
        losses = []
        valid_iterations = []
        valid_losses = []

        for epoch_id in range(num_epochs):
            data = get_random_track(training)
            length = len(data)
            seed = random.randint(0, length - sample_length - 1)
            x_data = []
            y_data = []
            x_data.append(data[seed : seed + sample_length])
            y_data.append(data[seed + 1 : seed + 1 + sample_length])
            x_data = np.array(x_data)
            y_data = np.array(y_data)

            _merged,_, _loss = session.run([merged,optimizer, loss], feed_dict = {x: x_data, y: y_data})
            
            averaged_training_loss = _loss/sample_length
            iterations.append(epoch_id)
            losses.append(averaged_training_loss)

            if epoch_id % 100 == 0:
                #use this if we wanna generate a plot of loss vs. epoch
                print("Loss for epoch %d: %f" % (epoch_id, averaged_training_loss))
                if validate:
                    cur_valid_loss = []

                    for data in validation:
                        x_valid = np.array([data[:-1]])
                        y_valid = np.array([data[1:]])
                        next_loss = session.run(loss,feed_dict = {x:x_valid,y:y_valid})
                        cur_valid_loss.append(next_loss/y_valid.shape[1])

                    averaged_valid_loss = sum(cur_valid_loss)/len(cur_valid_loss)

                    print("Valuation loss for epoch %d: %f" % (epoch_id, averaged_valid_loss))

                    valid_iterations.append(epoch_id)
                    valid_losses.append(averaged_valid_loss)

            writer.add_summary(_merged,epoch_id)

        if validate:
            valid_loss = []
            for data in validation:
                x_valid = np.array([data[:-1]])
                y_valid = np.array([data[1:]])
                next_loss = session.run(loss,feed_dict={x:x_valid,y: y_valid})
                valid_loss.append(next_loss/y_valid.shape[1])
            valid_loss_all.append(sum(valid_loss)/len(valid_loss))
        else:
            break

    print("Done Training")

    if validate:
        print("K-fold CV loss = {}".format(sum(valid_loss_all)/len(valid_loss_all)))

# ---------- MUSIC GENERATION ----------
    print('Generating music')
    generate_sample = True
    if generate_sample:
        seq = train_seq[-4][:sample_length]

        if not switch:
            for i in range(100):
                x_init = np.reshape([make_feature_vec(i) for i in seq[-sample_length:]],[1,sample_length,vec_size])
                pred_logits = session.run(logits,feed_dict={x:x_init})
                dist = sftmax(pred_logits[0][-1])
                index =np.random.choice(len(dist),p=dist)
                seq.append(index)
        else:
            new_state_gen = np.zeros((num_layers, 2, batch_size, lstm_size))
            x_init = np.reshape([make_feature_vec(i) for i in seq],[1,len(seq),vec_size])
            if len(seq) > 1:
                x_init = np.reshape([make_feature_vec(i) for i in seq[:-1]],[1,len(seq)-1,vec_size])
                new_state_gen = session.run(state,feed_dict = {x:x_init})
            start = np.reshape(make_feature_vec(seq[-1]),[1,1,vec_size])
            for i in range(100):
                new_out_logits, new_state_gen = session.run([out_logits,new_state], feed_dict={start_vec:start,state_placeholder:new_state_gen})

                dist = sftmax(new_out_logits[0])
                
                print(sorted(dist,reverse=True)[:5])
                index =np.random.choice(len(dist),p=dist)
                seq.append(index)
                start = np.reshape(make_feature_vec(index),[1,1,vec_size])

        current = seq
        current_converted = [num_to_vec[current[i]] for i in range(len(current))]

        print('Final Sequence: {}'.format(current_converted))

        for note in current_converted:
            n = m21.note.Note(note[0])
            n.duration = m21.duration.Duration(note[1])
            stream.append(n)
        stream.show()

# ---------- PLOTTING ----------
    plt.plot(iterations, losses, c='green')
    plt.plot(valid_iterations, valid_losses, c='red')
    plt.title('Evolution of SGD Training Loss using LSTM')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.show()


