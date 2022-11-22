import tensorflow as tf 
# tf.disable_v2_behavior() 
# tf.disable_v2_behavior()
from tensorflow.python.ops import variable_scope
from keras.layers import RNN
import tensorflow_addons as tfa

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import numpy as np

class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = RNN.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = RNN.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = RNN.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.compat.v1.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.compat.v1.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.compat.v1.variable_scope('rnnlm'):
            softmax_w = tf.compat.v1.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.compat.v1.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.compat.v1.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(params=embedding, ids=self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, rate=1 - (args.output_keep_prob))

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(input=prev, axis=1))
            return tf.nn.embedding_lookup(params=embedding, ids=prev_symbol)

        print(inputs)

        def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):

                with tf.compat.v1.variable_scope(scope or "rnn_decoder"):
                    state = initial_state
                    outputs = []
                    prev = None
                    for i, inp in enumerate(decoder_inputs):
                        print("DOING IT")
                        print("prev",prev)
                        if loop_function is not None and prev is not None:
                            print("NOT NONE")

                            with tf.compat.v1.variable_scope("loop_function", reuse=True):
                                inp = loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        output, state = cell(inp, state)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = output

                return outputs, state

        print("TRAINING",training)

        # training = False

        outputs, last_state = rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')

        # outputs, last_state, _ = decoder(
        #     inputs,
        #     initial_state=self.initial_state,
        #     training=False,
        # )

        print("OUTPUTS",outputs)

        output = tf.compat.v1.reshape(tf.compat.v1.concat(outputs, 1), [-1, args.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):

            if len(targets) != len(logits) or len(weights) != len(logits):
                raise ValueError("Lengths of logits, weights, and targets must be the same "
                                "%d, %d, %d." % (len(logits), len(weights), len(targets)))

            with ops.name_scope(name, "sequence_loss_by_example",
                                logits + targets + weights):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    if softmax_loss_function is None:
                        target = array_ops.reshape(target, [-1])
                        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                            labels=target, logits=logit)
                    else:
                        crossent = softmax_loss_function(labels=target, logits=logit)
                    log_perp_list.append(crossent * weight)
                log_perps = math_ops.add_n(log_perp_list)
                if average_across_timesteps:
                    total_size = math_ops.add_n(weights)
                    total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
                    log_perps /= total_size
            return log_perps

        loss = sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])

        with tf.compat.v1.name_scope('cost'):
            self.cost = tf.reduce_sum(input_tensor=loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ys=self.cost, xs=tvars),
                args.grad_clip)
        with tf.compat.v1.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.compat.v1.summary.histogram('logits', self.logits)
        tf.compat.v1.summary.histogram('loss', loss)
        tf.compat.v1.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret