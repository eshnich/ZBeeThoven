import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import music21 as m21
import parse
import random
import matplotlib.pyplot as plt
import os

test_midi = 'http://kern.ccarh.org/cgi-bin/ksdata?l=cc/bach/cello&file=bwv1007-01.krn&f=xml'
train_data = []
dictionary_data = []
switch = True
irish = False

# # add the midis to our data
print('Loading data')
if irish:
    for fn in os.listdir('Connolly_MusicMID/'):
        data = parse.parse_music('Connolly_MusicMID/{}'.format(fn))
        train_data.append(data)
        dictionary_data.extend(data)
else:
    for fn in os.listdir('beethoven_midis/'):
        if fn[-4:] == '.mid':
            t_data, d_data = parse.parse_music('beethoven_midis/{}'.format(fn), irish=False)
            train_data.extend(t_data)
            dictionary_data.extend(d_data)
print('Finished loading data')

# hash which converts (note, duration) pairs to indices and vice versa
print('Building vec_to_num and num_to_vec')
vec_to_num,num_to_vec = parse.build_dataset(dictionary_data)

# vec_size is basically the length of the input vectors, and out_size is the length of the output vectors
vocab_size = len(vec_to_num)
vec_size = vocab_size
out_size = vec_size

# duration in number to words
def duration(num):
    if num == 1:
        return 'quarter'
    elif num == 2:
        return 'half'
    elif num == 0.5:
        return 'eighth'
    elif num == 0.25:
        return '16th'
    return 'whole'

# one-hot encodes
def make_feature_vec(point):
    vec = np.zeros(vec_size)
    vec[point] = 1.0
    return vec

# parameters
learning_rate = 0.001
num_epochs = 1000
sample_length = 48
batch_size = 1

def sftmax(z):
    ez = np.exp(z-np.max(z))
    return ez/ez.sum()

# convert to music
stream = m21.stream.Stream()

#model input and output
#x = tf.placeholder("float", [batch_size,sample_length,vec_size])
x = tf.placeholder("float", [batch_size,None,vec_size])
y = tf.placeholder("float", [batch_size,None,out_size]) #one-hot vector

lstm_size = 128

# Initializing THE LSTM --------------------

if switch:
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size,state_is_tuple=False)
    outputs, state = tf.nn.dynamic_rnn(cell=lstm,inputs=x, dtype=tf.float32)
    W = tf.Variable(tf.random_normal([lstm_size, out_size]))
    b = tf.Variable(tf.zeros([out_size]))
    outputs = tf.reshape(outputs,[-1,lstm_size])
    logits = tf.matmul(outputs,W)+b
    logits = tf.reshape(logits,[batch_size,-1,out_size])
else:
    W = {'out':tf.Variable(tf.random_normal([lstm_size, out_size]))}
    b = {'out':tf.Variable(tf.zeros([out_size]))}
    def RNN(x,W,b):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size,state_is_tuple=False)
        outputs, state = tf.nn.dynamic_rnn(cell=lstm,inputs=x, dtype=tf.float32)
        print("outputs",outputs)
        outputs = tf.reshape(outputs,[batch_size*sample_length,lstm_size])
        return tf.matmul(outputs,W['out']+b['out'])
    logits = RNN(x,W,b)
    logits = tf.reshape(logits,[batch_size,sample_length,out_size])

# outputs = tf.reshape(outputs,[-1,lstm_size])
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
tf.summary.scalar('Loss',loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

train_seq = []
for voice in train_data:
    voice_track = []
    for v in voice:
        voice_track += [vec_to_num[v]]
    train_seq.append(voice_track)


# TESTING THE LSTM -----------------------
if switch:
    start_vec = tf.placeholder(tf.float32,[1, 1, vec_size])
    in_state = tf.placeholder(tf.float32,[1, 2 * lstm_size])
    output, new_state = tf.nn.dynamic_rnn(cell=lstm,inputs = start_vec,initial_state=in_state,dtype=tf.float32)
    output = tf.reshape(output,[1,lstm_size])
    out_logits = tf.matmul(output,W) + b

def get_random_track():
    # chooses a random track in train_seq
    n = len(train_seq)
    while True:
        seed = random.randint(0, n - 1)
        voice_track = train_seq[seed]
        if len(voice_track) > sample_length:
            return [make_feature_vec(voice_track[i]) for i in range(len(voice_track))]

merged = tf.summary.merge_all() 
init = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init)
    writer = tf.summary.FileWriter("output",session.graph)

    #Training
    print('Starting Training')
    iterations = []
    losses = []
    for epoch_id in range(num_epochs):
        data = get_random_track()
        length = len(data)
        #print(length)
        #print([len(i) for i in train_seq])
        seed = random.randint(0, length - sample_length - 1)
        x_data = []
        y_data = []
        x_data.append(data[seed : seed + sample_length])
        y_data.append(data[seed + 1 : seed + 1 + sample_length])
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        _merged,_, _loss = session.run([merged,optimizer,loss], feed_dict = {x: x_data, y: y_data})
        if(epoch_id % 100 == 0):
            print("Loss for epoch %d: %f" % (epoch_id, _loss)) #use this if we wanna generate a plot of loss vs. epoch
            iterations.append(epoch_id)
            losses.append(_loss)
        writer.add_summary(_merged,epoch_id)
    print("Done Training")

    seq = train_seq[-4][-sample_length:]
    #seq = [0]
    if not switch:
        for i in range(100):
            x_init = np.reshape([make_feature_vec(i) for i in seq[-sample_length:]],[1,sample_length,vec_size])
            pred_logits = session.run(logits,feed_dict={x:x_init})
            dist = sftmax(pred_logits[0][-1])
            index =np.random.choice(len(dist),p=dist)
            seq.append(index)
            print("NEW SEQ",seq)
    else:
        new_state_gen = np.zeros([1,2*lstm_size])
        x_init = np.reshape([make_feature_vec(i) for i in seq],[1,len(seq),vec_size])
        if len(seq) > 1:
            x_init = np.reshape([make_feature_vec(i) for i in seq[:-1]],[1,len(seq)-1,vec_size])
            new_state_gen = session.run(state,feed_dict = {x:x_init})
        start = np.reshape(make_feature_vec(seq[-1]),[1,1,vec_size])
        for i in range(100):
            new_out_logits, new_state_gen = session.run([out_logits,new_state], feed_dict={start_vec:start,in_state:new_state_gen})
            dist = sftmax(new_out_logits[0])
            
            print(sorted(dist,reverse=True)[:5])
            index =np.random.choice(len(dist),p=dist)
            #index = int(tf.argmax(new_out_logits, 1).eval())
            #print(dist[index])
            seq.append(index)
            start = np.reshape(make_feature_vec(index),[1,1,vec_size])
            #print("NEW SWITCH", seq)

    current = seq
    print('Final: {}'.format(current))
    current_converted = [num_to_vec[current[i]] for i in range(len(current))]
    print('Final converted: {}'.format(current_converted))
    for note in current_converted:
        n = m21.note.Note(note[0])
        n.duration = m21.duration.Duration(note[1])
        stream.append(n)
        #stream.append(m21.note.Note(note[0], type=duration(note[1])))
    stream.show()

    plt.plot(iterations, losses, c='green')
    plt.title('Evolution of SGD Training Loss using LSTM')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

'''

    

    #MUSIC GENERATION
    
#    new_state_gen = np.zeros([1,2*lstm_size])
    seq = [6] #the initial sequence we feed the LSTM
    x_init = np.reshape([make_feature_vec(i) for i in seq],[1,len(seq),vec_size])
    if len(seq) > 1:
#        x_init = np.reshape([make_feature_vec(i) for i in seq[:-1]],[1,len(seq)-1,vec_size])
#        new_state_gen = session.run(state,feed_dict = {x:x_init})
        pass

    start = np.reshape(make_feature_vec(seq[-1]),[1,1,vec_size])
    for i in range(20):
#        new_out_logits, new_state_gen = session.run([out_logits,new_state], feed_dict={start_vec:start,in_state:new_state_gen})
        new_logits = session.run([logits],feed_dict={x:x_init})

        index = int(tf.argmax(new_out_logits, 1).eval())
        #change this to be random
        seq.append(index)
        start = np.reshape(make_feature_vec(index),[1,1,vec_size])

    current = seq
    print('Final: {}'.format(current))
    current_converted = [num_to_vec[current[i]] for i in range(len(current))]
    print('Final converted: {}'.format(current_converted))
    print("adding")
    for note in current_converted:
        
        stream.append(m21.note.Note(note[0], type=duration(note[1])))
    print("note")
    stream.show()
'''
