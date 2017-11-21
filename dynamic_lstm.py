import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import music21 as m21
import parse
import random

test_midi = 'http://kern.ccarh.org/cgi-bin/ksdata?l=cc/bach/cello&file=bwv1007-01.krn&f=xml'

vec_to_num = {('C4', 1): 0, ('C4', 4): 1, ('D4', 1): 2, ('D4', 2): 3, ('E4', 1): 4, ('E4', 2): 5, ('G4', 1): 6, ('G4', 2): 7}
num_to_vec = dict(zip(vec_to_num.values(), vec_to_num.keys()))
print('Vector to Num: {}'.format(vec_to_num))
print('Num to Vector: {}'.format(num_to_vec))

def duration(num):
    if num == 1:
        return 'quarter'
    elif num == 2:
        return 'half'
    return 'whole'

def make_feature_vec(point):
    vec = []
    for i in range(vec_size):
        if i == point:
            vec.append(1.0)
        else:
            vec.append(0.0)
    return vec

# Parameters
learning_rate = 0.001
training_iters = 200
display_step = 1000
num_epochs = 100
n_input = 8
vocab_size = 8

batch_size = 1
sample_length = 7
vec_size = 8
out_size = 8

# convert to music
stream = m21.stream.Stream()
final_music = [('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 2), ('D4', 1), ('D4', 1), ('D4', 2), ('E4', 1), ('G4', 1), ('G4', 2), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 1), ('C4', 4), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 2), ('D4', 1), ('D4', 1), ('D4', 2), ('E4', 1), ('G4', 1), ('G4', 2), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 1), ('C4', 4), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 2), ('D4', 1), ('D4', 1), ('D4', 2), ('E4', 1), ('G4', 1), ('G4', 2), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 1), ('C4', 4)]

#model input and output
#x = tf.placeholder("float", [batch_size,sample_length,vec_size])
x = tf.placeholder("float", [batch_size,None,vec_size])
y = tf.placeholder("float", [batch_size,sample_length,out_size]) #one-hot vector

lstm_size = 256
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size,state_is_tuple=False)

#Initializing THE LSTM --------------------

outputs, state = tf.nn.dynamic_rnn(cell=lstm,inputs=x, dtype=tf.float32)

W = tf.Variable(tf.random_normal([lstm_size, out_size]))
b = tf.Variable(tf.zeros([out_size]))

outputs = tf.reshape(outputs,[batch_size*sample_length,lstm_size])
logits = tf.matmul(outputs,W)+b

logits = tf.reshape(logits,[batch_size,sample_length,out_size])

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)




# train_seq = [3,2,1,2,3,3,3,3,2,2,2,2,3,5,5,5,3,2,1,2,3,3,3,3,2,2,3,2,1,1,1,1]
train_seq = [4, 2, 0, 2, 4, 4, 5, 2, 2, 3, 4, 6, 7, 4, 2, 0, 2, 4, 4, 4, 4, 2, 2, 4, 2, 1]
print('Training sequence: {}'.format(train_seq))
print('Training sequence converted: {}'.format([num_to_vec[train_seq[i]] for i in range(len(train_seq))]))


#TESTING THE LSTM -----------------------

start_vec = tf.placeholder(tf.float32,[1,1,vec_size])
in_state = tf.placeholder(tf.float32,[1,2*lstm_size])
output, new_state = tf.nn.dynamic_rnn(cell=lstm,inputs = start_vec,initial_state=in_state,dtype=tf.float32)
output = tf.reshape(output,[1,lstm_size])
out_logits = tf.matmul(output,W)+b


def get_data():
    start = random.randint(0,len(train_seq)-sample_length-1)
    print(start)
    return [make_feature_vec(train_seq[i]) for i in range(start,start+sample_length+1)]

init = tf.global_variables_initializer()

with tf.Session() as session:
    
    session.run(init)
    step = 0
    offset = 0

    #Training
    for epoch_id in range(num_epochs):
        x_data = []
        y_data = []

        for i in range(batch_size):
            data = get_data()
            x_data.append(data[:-1])
            y_data.append(data[1:])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        _, _loss, _state= session.run([optimizer,loss,state], feed_dict = {x:x_data,y:y_data})
        print("Loss for epoch %d = %f" % (epoch_id,_loss)) #use this if we wanna generate a plot of loss vs. epoch
    print("Done Training")

    #MUSIC GENERATION
    
    new_state_gen = np.zeros([1,2*lstm_size])
    seq = [6] #the initial sequence we feed the LSTM
    if len(seq) > 1:
        x_init = np.reshape([make_feature_vec(i) for i in seq[:-1]],[1,len(seq)-1,vec_size])
        new_state_gen = session.run(state,feed_dict = {x:x_init})
    start = np.reshape(make_feature_vec(seq[-1]),[1,1,vec_size])
    for i in range(20):
        new_out_logits, new_state_gen = session.run([out_logits,new_state], feed_dict={start_vec:start,in_state:new_state_gen})
        index = int(tf.argmax(new_out_logits, 1).eval())
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