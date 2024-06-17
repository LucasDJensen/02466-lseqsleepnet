import os
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import h5py
import hdf5storage

from lseqsleepnet import LSeqSleepNet
from config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_wrapper import DataGeneratorWrapper
import time
from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_string("allow_soft_placement", 'True', "Allow device soft device placement")
tf.app.flags.DEFINE_string("log_device_placement", 'False', "Log placement of ops on devices")
# My Parameters
tf.app.flags.DEFINE_string("eeg_test_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt"), "file containing the list of test EEG data")
tf.app.flags.DEFINE_string("eog_test_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt"), "file containing the list of test EOG data")
tf.app.flags.DEFINE_string("emg_test_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt"), "file containing the list of test EMG data")

tf.app.flags.DEFINE_string("out_dir", os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet_latent_space/outputs/train_test/"), "Output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet_latent_space/checkpoint/"), "Checkpoint directory")

tf.app.flags.DEFINE_float("dropout_rnn", 0.9, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("batch_size", 8, "Number of instances per mini-batch (default: 32)")
tf.app.flags.DEFINE_integer("nclasses_data", 4, "Number of classes in the data (whether artifacts are discarded or not is controlled in nclasses_model)")
tf.app.flags.DEFINE_string("mask_artifacts", 'True', "whether masking artifacts in loss")
tf.app.flags.DEFINE_string("artifact_detection", 'False', "whether just predicting if an epoch is an artifact")
tf.app.flags.DEFINE_integer("ndim", 129, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("frame_seq_len", 17, "Sequence length (default: 20)")

# subsuqence length
tf.app.flags.DEFINE_integer("sub_seq_len", 10, "Sequence length (default: 32)")
# number of subsequence
tf.app.flags.DEFINE_integer("nsubseq", 8, "number of overall segments (default: 9)")

tf.app.flags.DEFINE_string("best_model_criteria", 'balanced_accuracy', "whether to save the model with best 'balanced_accuracy' or 'accuracy' (default: accuracy)")
tf.app.flags.DEFINE_string("loss_type", 'weighted_ce', "whether to use 'weighted_ce' or 'normal_ce' (default: accuracy)")

tf.app.flags.DEFINE_integer("dualrnn_blocks", 1, "Number of dual rnn blocks (default: 1)")

tf.app.flags.DEFINE_float("gpu_usage", 0.5, "Dropout keep probability (default: 0.5)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
print(sys.argv[0])
flags_dict = {}
for idx, a in enumerate(sys.argv):
    if a[:2] == "--":
        flags_dict[a[2:]] = sys.argv[idx + 1]

for attr in sorted(flags_dict):  # python3
    print("{}={}".format(attr.upper(), flags_dict[attr]))
print("")

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

with open(os.path.join(out_path, 'test_settings.txt'), 'w') as f:
    for attr in sorted(flags_dict):  # python3
        f.write("{}={}".format(attr.upper(), flags_dict[attr]))
        f.write('\n')

config = Config()
config.dropout_rnn = FLAGS.dropout_rnn
config.sub_seq_len = FLAGS.sub_seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size = FLAGS.attention_size

# Pass boolean flags from string to bool
# The reason is that tensorflow boolean flags don't look at the value specified in the command line. Whenever a boolean flag is present in the command line, it will evaluate to True.
boolean_flags = [
    'allow_soft_placement', 
    'log_device_placement', 
    'mask_artifacts',
    'artifact_detection',
]
for bf in boolean_flags:
    assert getattr(FLAGS, bf) == 'True' or getattr(FLAGS, bf) == 'False', "%s must be a string and either 'True' or 'False'" % bf
    setattr(config, bf, getattr(FLAGS, bf) == 'True')

if config.artifact_detection == True:
    config.artifacts_label = FLAGS.nclasses_data - 1 # right now the code probably just works when the artifact label is the last one
    config.nclasses_model = 1
    config.nclasses_data = 2
    assert config.mask_artifacts == False, "mask_artifacts must be False if artifact_detection=True"
    print('Artifact detection is active. nclasses_data set to 2, nclasses_model set to 1.')
elif config.artifact_detection == False:
    if config.mask_artifacts == True:
        config.nclasses_data = FLAGS.nclasses_data
        config.artifacts_label = FLAGS.nclasses_data - 1 # right now the code probably just works when the artifact label is the last one
        config.nclasses_model = config.nclasses_data - 1 
    else:
        config.nclasses_data = FLAGS.nclasses_data
        config.nclasses_model = config.nclasses_data
        config.artifacts_label = FLAGS.nclasses_data - 1 # right now the code probably just works when the artifact label is the last one
config.ndim = FLAGS.ndim
config.frame_seq_len = FLAGS.frame_seq_len
config.best_model_criteria = FLAGS.best_model_criteria
config.loss_type = FLAGS.loss_type
config.batch_size = FLAGS.batch_size
config.l2_reg_lambda = config.l2_reg_lambda / FLAGS.batch_size # scaling by btach size because now I'm normalizing the loss by the number of elements in batch

config.nsubseq = FLAGS.nsubseq
config.dualrnn_blocks = FLAGS.dualrnn_blocks

eeg_active = (FLAGS.eeg_test_data != "")
eog_active = (FLAGS.eog_test_data != "")
emg_active = (FLAGS.emg_test_data != "")

if not eog_active and not emg_active:
    print("eeg active")
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.sub_seq_len * config.nsubseq,
                                             nclasses = config.nclasses_data, 
                                             artifact_detection = config.artifact_detection,
                                             artifacts_label = config.artifacts_label,
                                             shuffle=False)
    test_gen_wrapper.compute_eeg_normalization_params_by_signal()
    nchannel = 1

elif eog_active and not emg_active:
    print("eeg and eog active")
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                            eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                            num_fold=config.num_fold_testing_data,
                                            data_shape_2=[config.frame_seq_len, config.ndim],
                                            seq_len = config.sub_seq_len * config.nsubseq,
                                            nclasses = config.nclasses_data, 
                                            artifact_detection = config.artifact_detection,
                                            artifacts_label = config.artifacts_label,
                                            shuffle=False)
    test_gen_wrapper.compute_eeg_normalization_params_by_signal()
    test_gen_wrapper.compute_eog_normalization_params_by_signal()
    nchannel = 2
elif eog_active and emg_active:
    print("eeg, eog, and emg active")
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                            eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                            emg_filelist=os.path.abspath(FLAGS.emg_test_data),
                                            num_fold=config.num_fold_testing_data,
                                            data_shape_2=[config.frame_seq_len, config.ndim],
                                            seq_len = config.sub_seq_len * config.nsubseq,
                                            nclasses = config.nclasses_data, 
                                            artifact_detection = config.artifact_detection,
                                            artifacts_label = config.artifacts_label,
                                            shuffle=False)
    test_gen_wrapper.compute_eeg_normalization_params_by_signal()
    test_gen_wrapper.compute_eog_normalization_params_by_signal()
    test_gen_wrapper.compute_emg_normalization_params_by_signal()
    nchannel = 3

config.nchannel = nchannel

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_usage, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=config.allow_soft_placement,
      log_device_placement=config.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = LSeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables())
        # Load the saved model
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model all loaded")

        def dev_step(x_batch, y_batch):
            x_shape = x_batch.shape
            y_shape = y_batch.shape
            x = np.zeros(x_shape[:1] + (config.nsubseq, config.sub_seq_len,) + x_shape[2:])
            y = np.zeros(y_shape[:1] + (config.nsubseq, config.sub_seq_len,) + y_shape[2:])
            for s in range(config.nsubseq):
                x[:, s] = x_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]
                y[:, s] = y_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]

            frame_seq_len = np.ones(len(x_batch) * config.sub_seq_len * config.nsubseq,
                                        dtype=int) * config.frame_seq_len
            sub_seq_len = np.ones(len(x_batch) * config.nsubseq, dtype=int) * config.sub_seq_len
            inter_subseq_len = np.ones(len(x_batch) * config.sub_seq_len, dtype=int) * config.nsubseq
            feed_dict = {
                net.input_x: x,
                net.input_y: y,
                net.dropout_rnn: 1.0,
                net.inter_subseq_len: inter_subseq_len,
                net.sub_seq_len: sub_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 0
            }
            output_loss, total_loss, yhat, score = sess.run([net.output_loss, net.loss, net.prediction, net.score], feed_dict)
            return output_loss, total_loss, yhat, score

        def _evaluate(gen):
            # Validate the model on the entire data in gen

            score = np.zeros([len(gen.data_index), config.sub_seq_len*config.nsubseq, config.nclasses_model])

            factor = 10

            # use 10x of minibatch size to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                for s in range(config.nsubseq):
                    score[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size,
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]
                test_step += 1
            if gen.pointer < len(gen.data_index):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                for s in range(config.nsubseq):
                    score[(test_step - 1) * factor * config.batch_size: len(gen.data_index),
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]

            return score


        def evaluate(gen_wrapper):

            N = int(np.sum(gen_wrapper.file_sizes) - (config.sub_seq_len*config.nsubseq - 1)*len(gen_wrapper.file_sizes))
            yhat = np.zeros([N, config.sub_seq_len*config.nsubseq])
            y = np.zeros([N, config.sub_seq_len*config.nsubseq])

            score = np.zeros([N, config.sub_seq_len*config.nsubseq, config.nclasses_model])

            count = 0
            output_loss = 0
            total_loss = 0
            gen_wrapper.new_subject_partition()
            for data_fold in range(config.num_fold_testing_data):
                gen_wrapper.next_fold()
                score_ = _evaluate(gen_wrapper.gen)

                score[count : count + len(gen_wrapper.gen.data_index)] = score_

                count += len(gen_wrapper.gen.data_index)

            return score

        # evaluation on test data
        start_time = time.time()
        test_score = evaluate(gen_wrapper=test_gen_wrapper)
        end_time = time.time()
        with open(os.path.join(out_dir, "test_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))

        hdf5storage.savemat(os.path.join(out_path, "test_ret.mat"),
                            {'score': test_score},
                            format='7.3')
        test_gen_wrapper.gen.reset_pointer()

print("End of script.")
