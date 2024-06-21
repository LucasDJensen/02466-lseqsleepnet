import tensorflow as tf
from nn_basic_layers import *
from filterbank_shape import FilterbankShape
from ops import *

# L-SeqSleepNet
class LSeqSleepNet(object):

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None, self.config.nsubseq, self.config.sub_seq_len, self.config.frame_seq_len, self.config.ndim, self.config.nchannel], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.nsubseq, self.config.sub_seq_len, self.config.nclasses_data], name="input_y")

        self.dropout_rnn = tf.placeholder(tf.float32, name="dropout_rnn")
        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization
        # sequence length for epoch-wise modelling (the number of image columns in one 1 sleep epoch)
        self.frame_seq_len = tf.placeholder(tf.int32, [None])
        # sequence length for inter-subsequence modelling
        self.inter_subseq_len = tf.placeholder(tf.int32, [None])
        # sub-sequence length for intra-subsequence modelling
        self.sub_seq_len = tf.placeholder(tf.int32, [None])
        # filter banks
        self.filtershape = FilterbankShape()

        x = tf.reshape(self.input_x, [-1, self.config.ndim, self.config.nchannel])
        # apply the filter bank layers
        processed_x = self.preprocessing(x)

        # epoch-wise modelling
        processed_x = tf.reshape(processed_x, [-1, self.config.frame_seq_len, self.config.nfilter*self.config.nchannel])
        epoch_x = self.epoch_encoder(processed_x)

        # long sequence modelling with dual encoder (i.e. intra-epoch encoder and inter-epoch encoder)
        epoch_x = tf.reshape(epoch_x, [-1, self.config.nsubseq, self.config.sub_seq_len, self.config.nhidden1*2])
        seq_x = self.dual_sequence_encoder(epoch_x, self.config.dualrnn_blocks)

        # fully connected layer for classification
        with tf.variable_scope("output_layer"):
            X_out = tf.reshape(seq_x, [-1, self.config.nhidden1*2])
            fc1 = fc(X_out, self.config.nhidden1*2, self.config.fc_size, name="fc1", relu=True)
            fc1 = dropout(fc1, self.dropout_rnn)
            fc2 = fc(fc1, self.config.fc_size, self.config.fc_size, name="fc2", relu=True)
            fc2 = dropout(fc2, self.dropout_rnn)
            self.score = fc(fc2, self.config.fc_size, self.config.nclasses_model, name="output", relu=False)

            if self.config.nclasses_model > 1:
                self.prediction = tf.argmax(self.score, 1, name="pred")
            elif self.config.nclasses_model == 1:
                prediction = tf.math.greater(tf.nn.sigmoid(self.score), 0.5)
                prediction = tf.where(prediction, tf.ones(tf.shape(self.score)), tf.zeros(tf.shape(self.score)), name="pred") # boolean to binary
                self.prediction = tf.cast(prediction, dtype=tf.int64)
            self.score = tf.reshape(self.score, [-1, self.config.nsubseq, self.config.sub_seq_len, self.config.nclasses_model])
            self.prediction = tf.reshape(self.prediction, [-1, self.config.nsubseq, self.config.sub_seq_len])
            # extract embeddings
            self.embeddings = fc2

        # calculate sequence cross-entropy output loss
        with tf.name_scope("output-loss"):
            if self.config.loss_type == 'normal_ce':
                input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers
                scores = tf.reshape(self.score, [-1, self.config.nclasses_model])

                if self.config.nclasses_model > 1:
                    input_y_categorical = tf.reshape(input_y_categorical, [-1]) # in categorical ce labels need to be 1-dimensional and scores 2-dimensional
                    scores = tf.nn.softmax(scores)

                    if self.config.mask_artifacts == False:
                        cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                        n_elements_in_batch = tf.cast(tf.size(cce), dtype=tf.float32)
                    elif self.config.mask_artifacts == True:
                        artifacts_column = tf.zeros([tf.shape(scores)[0],1])
                        scores = tf.concat([scores, artifacts_column], 1)

                        artifact_mask = tf.not_equal(input_y_categorical, self.config.artifacts_label) # artifact mask (boolean)
                        artifact_mask = tf.where(artifact_mask, tf.ones(tf.shape(artifact_mask)), tf.zeros(tf.shape(artifact_mask))) # boolean artifact mask to binary

                        cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                        cce = tf.multiply(cce, artifact_mask)

                        n_elements_in_batch = tf.reduce_sum(artifact_mask)
                elif self.config.nclasses_model == 1:
                    input_y_categorical = tf.reshape(input_y_categorical, [-1, config.nclasses_model]) # in binary ce both labels and scores need to be 2-dimensional
                    scores = tf.nn.sigmoid(scores)
                    cce = tf.keras.metrics.binary_crossentropy(y_true=tf.cast(input_y_categorical, tf.float32), y_pred=scores, from_logits=False)
                    n_elements_in_batch = tf.cast(tf.size(cce), dtype=tf.float32)

                cce = tf.reduce_sum(cce)
                self.output_loss = cce / n_elements_in_batch # average over sequence length and (not-artifacts) elements in batch

            elif self.config.loss_type == 'weighted_ce':
                input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers
                scores = tf.reshape(self.score, [-1, self.config.nclasses_model])

                if self.config.nclasses_model > 1:
                    input_y_categorical = tf.reshape(input_y_categorical, [-1]) # in categorical ce labels need to be 1-dimensional and scores 2-dimensional
                    scores = tf.nn.softmax(scores)

                    if self.config.mask_artifacts == False:
                        cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                        n_elements_in_batch = tf.cast(tf.size(cce), dtype=tf.float32)
                    elif self.config.mask_artifacts == True:
                        artifacts_column = tf.zeros([tf.shape(scores)[0],1])
                        scores = tf.concat([scores, artifacts_column], 1)

                        artifact_mask = tf.not_equal(input_y_categorical, self.config.artifacts_label) # artifact mask (boolean)
                        artifact_mask = tf.where(artifact_mask, tf.ones(tf.shape(artifact_mask)), tf.zeros(tf.shape(artifact_mask))) # boolean artifact mask to binary

                        cce = tf.keras.metrics.sparse_categorical_crossentropy(y_true=input_y_categorical, y_pred=scores, from_logits=False)
                        cce = tf.multiply(cce, artifact_mask)

                        n_elements_in_batch = tf.reduce_sum(artifact_mask)

                elif self.config.nclasses_model == 1:
                    input_y_categorical = tf.reshape(input_y_categorical, [-1, config.nclasses_model]) # in binary ce both labels and scores need to be 2-dimensional
                    scores = tf.nn.sigmoid(scores)
                    cce = tf.keras.metrics.binary_crossentropy(y_true=tf.cast(input_y_categorical, tf.float32), y_pred=scores, from_logits=False)
                    n_elements_in_batch = tf.cast(tf.size(cce), dtype=tf.float32)

                class_counts = []
                def cond_function_true_wce(cce, n_elements_in_batch, n_classes_in_batch, labels_class_i_binary, labels_class_i_bool):

                    w = n_elements_in_batch / (n_classes_in_batch * tf.reduce_sum(labels_class_i_binary))
                    weights_mask = tf.where(labels_class_i_bool, w*tf.ones(tf.shape(labels_class_i_bool)), tf.ones(tf.shape(labels_class_i_bool)))

                    weighted_cce = tf.multiply(cce, weights_mask)

                    return weighted_cce 

                def cond_function_false_wce(cce):
                    
                    identical_cce = tf.multiply(cce, tf.ones(tf.shape(cce)))

                    return identical_cce
                
                if config.nclasses_model == 1:
                    loop_range = 2
                else:
                    loop_range = config.nclasses_model

                for i in range(loop_range):
                    labels_class_i = tf.equal(input_y_categorical, i)
                    labels_class_i = tf.where(labels_class_i, tf.ones(tf.shape(labels_class_i)), tf.zeros(tf.shape(labels_class_i))) # boolean artifact mask to binary
                    
                    class_counts.append(tf.reduce_sum(labels_class_i))
                
                n_classes_in_batch = tf.math.count_nonzero(class_counts, dtype=tf.dtypes.float32)

                for i in range(loop_range):
                    labels_class_i_bool = tf.equal(input_y_categorical, i)
                    labels_class_i_binary = tf.where(labels_class_i_bool, tf.ones(tf.shape(labels_class_i_bool)), tf.zeros(tf.shape(labels_class_i_bool))) # boolean artifact mask to binary

                    cce  = tf.cond(tf.reduce_sum(labels_class_i_binary) > 0, lambda: cond_function_true_wce(cce, n_elements_in_batch, n_classes_in_batch, labels_class_i_binary, labels_class_i_bool), lambda: cond_function_false_wce(cce))

                cce = tf.reduce_sum(cce)
                self.output_loss = cce / n_elements_in_batch

        # add on L2-norm regularization (excluding the filter bank layers)
        with tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        with tf.name_scope("accuracy"):
            input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers
            input_y_categorical = tf.reshape(input_y_categorical, [-1]) 
            yhat = tf.reshape(self.prediction, [-1,])

            correct_prediction = tf.equal(yhat, input_y_categorical)
            correct_prediction = tf.where(correct_prediction, tf.ones(tf.shape(correct_prediction)), tf.zeros(tf.shape(correct_prediction)))

            if self.config.mask_artifacts == False:
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32)) / tf.size(correct_prediction)
            elif self.config.mask_artifacts == True:
                artifact_mask = tf.not_equal(input_y_categorical, self.config.artifacts_label) # artifact mask (boolean)
                artifact_mask = tf.where(artifact_mask, tf.ones(tf.shape(artifact_mask)), tf.zeros(tf.shape(artifact_mask))) # boolean artifact mask to binary
                correct_prediction_masked = tf.multiply(correct_prediction, artifact_mask)
                self.accuracy = tf.reduce_sum(correct_prediction_masked) / tf.reduce_sum(artifact_mask)
        
        with tf.name_scope("balanced_accuracy"):
            def cond_function_true(prediction_class_i, labels_class_i, recalls_sum, n_classes_in_balanced_accuracy):
                correct_prediction_class_i = tf.multiply(prediction_class_i, labels_class_i)

                recalls_sum += (tf.reduce_sum(correct_prediction_class_i) / tf.reduce_sum(labels_class_i))
                n_classes_in_balanced_accuracy += 1

                return [recalls_sum, n_classes_in_balanced_accuracy] 

            def cond_function_false(recalls_sum, n_classes_in_balanced_accuracy):
                
                recalls = tf.multiply(recalls_sum, tf.ones(tf.shape(recalls_sum)))
                n_classes = tf.multiply(n_classes_in_balanced_accuracy, tf.ones(tf.shape(n_classes_in_balanced_accuracy)))

                return [recalls, n_classes] 
        
            recalls_sum = 0.0
            n_classes_in_balanced_accuracy = 0.0
            
            input_y_categorical = tf.math.argmax(self.input_y, -1) # dummy labels to numbers

            if self.config.nclasses_model == 1:
                loop_range = 2
            else:
                loop_range = self.config.nclasses_model

            for i in range(loop_range):
                prediction_class_i = tf.equal(self.prediction, i)
                labels_class_i = tf.equal(input_y_categorical, i)
                prediction_class_i = tf.where(prediction_class_i, tf.ones(tf.shape(prediction_class_i)), tf.zeros(tf.shape(prediction_class_i))) # boolean artifact mask to binary
                labels_class_i = tf.where(labels_class_i, tf.ones(tf.shape(labels_class_i)), tf.zeros(tf.shape(labels_class_i))) # boolean artifact mask to binary
                
                [recalls_sum, n_classes_in_balanced_accuracy]  = tf.cond(tf.reduce_sum(labels_class_i) > 0, lambda: cond_function_true(prediction_class_i, labels_class_i, recalls_sum, n_classes_in_balanced_accuracy), lambda: cond_function_false(recalls_sum, n_classes_in_balanced_accuracy))

            self.balanced_accuracy = recalls_sum / n_classes_in_balanced_accuracy

    def preprocessing(self, input):
        # input of shape [-1, ndim, nchannel]
        # triangular filterbank shape
        Wbl = tf.constant(self.filtershape.lin_tri_filter_shape(nfilt=self.config.nfilter,
                                                                nfft=self.config.nfft,
                                                                samplerate=self.config.samplerate,
                                                                lowfreq=self.config.lowfreq,
                                                                highfreq=self.config.highfreq),
                          dtype=tf.float32,
                          name="W-filter-shape-eeg")

        # filter bank layer for eeg
        with tf.variable_scope("seq_filterbank-layer-eeg", reuse=tf.AUTO_REUSE):
            # Temporarily crush the feature_mat's dimensions
            Xeeg = tf.reshape(tf.squeeze(input[:, :, 0]), [-1, self.config.ndim])
            # first filter bank layer
            Weeg = tf.get_variable('Weeg', shape=[self.config.ndim, self.config.nfilter], initializer=tf.random_normal_initializer())
            # non-negative constraints
            Weeg = tf.sigmoid(Weeg)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            Wfb_eeg = tf.multiply(Weeg, Wbl)
            HWeeg = tf.matmul(Xeeg, Wfb_eeg)  # filtering

        # filter bank layer for eog
        if (self.config.nchannel > 1):
            with tf.variable_scope("seq_filterbank-layer-eog", reuse=tf.AUTO_REUSE):
                # Temporarily crush the feature_mat's dimensions
                Xeog = tf.reshape(tf.squeeze(input[:, :, 1]), [-1, self.config.ndim])
                # first filter bank layer
                Weog = tf.get_variable('Weog', shape=[self.config.ndim, self.config.nfilter],initializer=tf.random_normal_initializer())
                # non-negative constraints
                Weog = tf.sigmoid(Weog)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                Wfb_eog = tf.multiply(Weog, Wbl)
                HWeog = tf.matmul(Xeog, Wfb_eog)  # filtering

        # filter bank layer for emg
        if (self.config.nchannel > 2):
            with tf.variable_scope("seq_filterbank-layer-emg", reuse=tf.AUTO_REUSE):
                # Temporarily crush the feature_mat's dimensions
                Xemg = tf.reshape(tf.squeeze(input[:, :, 2]), [-1, self.config.ndim])
                # first filter bank layer
                Wemg = tf.get_variable('Wemg', shape=[self.config.ndim, self.config.nfilter], initializer=tf.random_normal_initializer())
                # non-negative constraints
                Wemg = tf.sigmoid(Wemg)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                Wfb_emg = tf.multiply(Wemg, Wbl)
                HWemg = tf.matmul(Xemg, Wfb_emg)  # filtering

        if (self.config.nchannel > 2):
            X2 = tf.concat([HWeeg, HWeog, HWemg], axis=1)
        elif (self.config.nchannel > 1):
            X2 = tf.concat([HWeeg, HWeog], axis=1)
        else:
            X2 = HWeeg

        return X2

    # epoch-wise sequential modelling with bidirectional LSTM
    def epoch_encoder(self, input):
        # input of shape [-1, frame_seq_len, dim]
        with tf.variable_scope("seq_frame_rnn_layer", reuse=tf.AUTO_REUSE) as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer_bn_new(self.config.nhidden1,
                                                                    self.config.lstm_nlayer1,
                                                                    seq_len=self.config.frame_seq_len,
                                                                    is_training=self.istraining,
                                                                    input_keep_prob=self.dropout_rnn,
                                                                    output_keep_prob=self.dropout_rnn)
            rnn_out1, _ = bidirectional_recurrent_layer_output_new(fw_cell, bw_cell, input, self.frame_seq_len, scope=scope)
            # print(rnn_out1.get_shape())

        with tf.variable_scope("frame_attention_layer", reuse=tf.AUTO_REUSE):
            frame_attention_out1, _ = attention(rnn_out1, self.config.attention_size)
            # print(frame_attention_out1.get_shape())
        return frame_attention_out1

    def residual_rnn(self, input, seq_len, in_dropout=1.0, out_dropout=1.0, name='rnn_res'):
        # input of shape [-1, nseq, dim]
        _, nseq, dim = input.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer_bn_new(self.config.nhidden2,
                                                                    self.config.lstm_nlayer2,
                                                                    seq_len = nseq,
                                                                    is_training=self.istraining,
                                                                    input_keep_prob=in_dropout,
                                                                    output_keep_prob=out_dropout)
            rnn_out, _ = bidirectional_recurrent_layer_output_new(fw_cell, bw_cell, input, seq_len, scope=scope)
            # linear projection
            rnn_out = fc(tf.reshape(rnn_out, [-1, self.config.nhidden2*2]),
                         self.config.nhidden2*2, dim, name="fc", relu=False)
            rnn_out = tf.contrib.layers.layer_norm(rnn_out)
            rnn_out = tf.reshape(rnn_out, [-1, nseq, dim]) + input

            return rnn_out


    # dual encoder for long sequential modelling
    def dual_sequence_encoder(self, input, N):
        # input of shape (-1, num_subsequence, subsequence_len, dim)
        _, nsubseq, subseq_len, dim = input.get_shape().as_list()
        for n in range(N):
            input = tf.reshape(input, [-1, subseq_len, dim])
            rnn1_out = self.residual_rnn(input, self.sub_seq_len,
                                         in_dropout = 1.0 if n == 0 else self.dropout_rnn,
                                         out_dropout = self.dropout_rnn,
                                         name = 'intra_chunk_rnn_' + str(n+1))
            rnn1_out = tf.reshape(rnn1_out, [-1, nsubseq, subseq_len, dim]) # [-1, num_subseq, subseq_len, nhidden2*2]
            rnn1_out = tf.transpose(rnn1_out, perm=[0, 2, 1, 3]) # [-1, subseq_len, num_subseq, nhidden2*2]
            rnn1_out = tf.reshape(rnn1_out, [-1, nsubseq, dim])

            rnn2_out = self.residual_rnn(rnn1_out, self.inter_subseq_len,
                                         in_dropout = self.dropout_rnn,
                                         out_dropout = self.dropout_rnn,
                                         name = 'inter_chunk_rnn_' + str(n+1))
            rnn2_out = tf.reshape(rnn2_out, [-1, subseq_len, nsubseq, dim])  # [-1, subseq_len, num_subseq, nhidden2*2]
            rnn2_out = tf.transpose(rnn2_out, perm=[0, 2, 1, 3])  # [-1, num_subseq, subseq_len, nhidden2*2]
            rnn2_out = tf.reshape(rnn2_out, [-1, subseq_len, dim])
            input = rnn2_out

        input = tf.reshape(input, [-1, nsubseq, subseq_len, dim])
        input = dropout(input, self.dropout_rnn)
        return input


