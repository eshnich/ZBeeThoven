import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import music21 as m21
import parse

test_midi = 'http://kern.ccarh.org/cgi-bin/ksdata?l=cc/bach/cello&file=bwv1007-01.krn&f=xml'

# data = parse.parse_music(test_midi)
# vec_to_num, num_to_vec = parse.build_dataset(data)
# print('Data: {}'.format(data))
# print('Vector to Num: {}'.format(vec_to_num))
# print('Num to Vector: {}'.format(num_to_vec))

vec_to_num = {('C4', 1): 0, ('C4', 4): 1, ('D4', 1): 2, ('D4', 2): 3, ('E4', 1): 4, ('E4', 2): 5, ('G4', 1): 6, ('G4', 2): 7}
num_to_vec = dict(zip(vec_to_num.values(), vec_to_num.keys()))
print('Vector to Num: {}'.format(vec_to_num))
print('Num to Vector: {}'.format(num_to_vec))

# Parameters
learning_rate = 0.001
training_iters = 200
display_step = 1000
n_input = 8
vocab_size = 8

def duration(num):
    if num == 1:
        return 'quarter'
    elif num == 2:
        return 'half'
    return 'whole'

# convert to music
stream = m21.stream.Stream()
final_music = [('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 2), ('D4', 1), ('D4', 1), ('D4', 2), ('E4', 1), ('G4', 1), ('G4', 2), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 1), ('C4', 4), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 2), ('D4', 1), ('D4', 1), ('D4', 2), ('E4', 1), ('G4', 1), ('G4', 2), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 1), ('C4', 4), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 2), ('D4', 1), ('D4', 1), ('D4', 2), ('E4', 1), ('G4', 1), ('G4', 2), ('E4', 1), ('D4', 1), ('C4', 1), ('D4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 1), ('C4', 4)]
# for note in final_music:
#     stream.append(m21.note.Note(note[0], type=duration(note[1])))
# stream.show()

# number of units in RNN cell
n_hidden = 512

#model input and output
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size]) #one-hot vector

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

print('Weights: {}'.format(weights))
print('Biases: {}'.format(biases))

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    # print('x: {}'.format(x))
    x = tf.reshape(x, [-1, n_input])
    # print('x after tf.reshape: {}'.format(x))

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)
    # print('x after tf.split: {}'.format(x))

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
probs = tf.nn.softmax(pred)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# train_seq = [3,2,1,2,3,3,3,3,2,2,2,2,3,5,5,5,3,2,1,2,3,3,3,3,2,2,3,2,1,1,1,1]
train_seq = [4, 2, 0, 2, 4, 4, 5, 2, 2, 3, 4, 6, 7, 4, 2, 0, 2, 4, 4, 4, 4, 2, 2, 4, 2, 1]
print('Training sequence: {}'.format(train_seq))
print('Training sequence converted: {}'.format([num_to_vec[train_seq[i]] for i in range(len(train_seq))]))

with tf.Session() as session:
    session.run(init)
    step = 0 
    offset = 0
    while step < training_iters:
        if step % 50 == 0:
            print(step)
        offset = offset % len(train_seq)-n_input
        x_in = [train_seq[i] for i in range(offset,offset+n_input)]
        x_in = np.reshape(np.array(x_in), [-1, n_input, 1])
        print('It\'s happening! with step: {}'.format(step))
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        print(symbols_out_onehot)
        symbols_out_onehot[train_seq[offset+n_input]] = 1.0
        print(symbols_out_onehot)
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        print(symbols_out_onehot)
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                    feed_dict={x: x_in, y: symbols_out_onehot})

        step += 1
        offset += 1
    print("Done Training")

    # current = [3,2,1,2,3,3,3,2]
    current = [4, 2, 0, 2, 4, 4, 5, 2, 2, 3, 4, 6, 7]
    print('Current: {}'.format(current))
    print('Current converted: {}'.format([num_to_vec[current[i]] for i in range(len(current))]))
    # for i in range(100):
    for i in range(48):
        next_vals = current[-n_input:]
        keys = np.reshape(np.array(next_vals), [-1, n_input, 1])
        onehot_pred = session.run(pred, feed_dict={x: keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        current.append(onehot_pred_index)

    print('Final: {}'.format(current))
    current_converted = [num_to_vec[current[i]] for i in range(len(current))]
    print('Final converted: {}'.format(current_converted))
    print("adding")
    for note in current_converted:
        
        stream.append(m21.note.Note(note[0], type=duration(note[1])))
    print("note")
    stream.show()


'''
# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, 5, 2, 4]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
'''
