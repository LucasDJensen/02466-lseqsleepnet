import os
import numpy as np
import sys
import h5py
import hdf5storage
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, auc, silhouette_score, silhouette_samples, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from matplotlib import cm
from datagenerator_wrapper import DataGeneratorWrapper
from lseqsleepnet import LSeqSleepNet
from config import Config
from _globals import *

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_string("allow_soft_placement", 'True', "Allow device soft device placement")
tf.app.flags.DEFINE_string("log_device_placement", 'False', "Log placement of ops on devices")
# My Parameters
tf.app.flags.DEFINE_string("eeg_test_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt"), "file containing the list of test EEG data")
tf.app.flags.DEFINE_string("eog_test_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt"), "file containing the list of test EOG data")
tf.app.flags.DEFINE_string("emg_test_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt"), "file containing the list of test EMG data")

tf.app.flags.DEFINE_string("eeg_eval_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/eval_list.txt"), "file containing the list of evaluation EEG data")
tf.app.flags.DEFINE_string("eog_eval_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/eval_list.txt"), "file containing the list of evaluation EOG data")
tf.app.flags.DEFINE_string("emg_eval_data", os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/eval_list.txt"), "file containing the list of evaluation EMG data")

tf.app.flags.DEFINE_string("out_dir", os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/outputs/train_test/"), "Output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", os.path.join(HPC_STORAGE_PATH, "results_lseqsleepnet/checkpoint/"), "Checkpoint directory")

tf.app.flags.DEFINE_float("dropout_rnn", 0.9, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("batch_size", 8, "Number of instances per mini-batch (default: 32)")
tf.app.flags.DEFINE_integer("nclasses_data", 4, "Number of classes in the data (whether artifacts are discarded or not is controlled in nclasses_model)")
tf.app.flags.DEFINE_string("mask_artifacts", 'False', "whether masking artifacts in loss")
tf.app.flags.DEFINE_string("artifact_detection", 'True', "whether just predicting if an epoch is an artifact")
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
    config.nclasses_model = 1
    config.nclasses_data = 2
    assert config.mask_artifacts == False, "mask_artifacts must be False if artifact_detection=True"
    print('Artifact detection is active. nclasses_data set to 2, nclasses_model set to 1.')
elif config.artifact_detection == False:
    if config.mask_artifacts == True:
        config.nclasses_data = FLAGS.nclasses_data
        config.nclasses_model = config.nclasses_data - 1
    else:
        config.nclasses_data = FLAGS.nclasses_data
        config.nclasses_model = config.nclasses_data


config.artifacts_label = FLAGS.nclasses_data - 1
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



# Fit Gaussian Mixture Model to the provided embeddings
def fit_gmm(embeddings, n_components):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(embeddings)
    return gmm


# Fit K-Means Clustering
def fit_kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


# Perform t-SNE
def perform_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


# Plot density estimation
def plot_density(gmm, embeddings, out_dir):
    plt.figure(figsize=(8, 6))

    # Create a mesh grid for plotting
    x = np.linspace(np.min(embeddings[:, 0]), np.max(embeddings[:, 0]), 100)
    y = np.linspace(np.min(embeddings[:, 1]), np.max(embeddings[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    # Compute the log likelihood of the data under the GMM
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)

    # Plot the contour map
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Log likelihood')
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=2)

    plt.title('Density Estimation using Gaussian Mixture Model')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.grid(True)

    # Save the plot
    plot_file = os.path.join(out_dir, 'gmm_density_estimation.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    plt.show()


# Plot t-SNE results
def plot_tsne(embeddings_2d, labels, out_dir):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plot_file = os.path.join(out_dir, 'tsne_plot.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    plt.show()


# Plot ROC curve
def plot_roc(labels, scores, out_dir, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels == i, scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Save the plot
    plot_file = os.path.join(out_dir, 'roc_curve.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    plt.show()


# Plot Silhouette Analysis
def plot_silhouette(embeddings, cluster_labels, out_dir):
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    print(f"Average Silhouette Score: {silhouette_avg:.2f}")

    # Create a silhouette plot
    plt.figure(figsize=(10, 6))
    y_lower = 10
    for i in range(np.max(cluster_labels) + 1):
        ith_cluster_silhouette_values = silhouette_samples(embeddings, cluster_labels == i)
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / (np.max(cluster_labels) + 1))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.xticks(np.arange(-0.1, 1.1, 0.2))

    plot_file = os.path.join(out_dir, 'silhouette_plot.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file)
    plt.show()


# Standardize the embeddings
def standardize_embeddings(embeddings, mean, std):
    return (embeddings - mean) / std


# Initialize the validation data generator
validation_gen_wrapper = DataGeneratorWrapper(
    eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
    eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
    emg_filelist=os.path.abspath(FLAGS.emg_eval_data),
    num_fold=config.num_fold_testing_data,
    data_shape_2=[config.frame_seq_len, config.ndim],
    seq_len=config.sub_seq_len * config.nsubseq,
    nclasses=4,
    artifact_detection=False,
    artifacts_label=3,
    shuffle=False
)

# Initialize the test data generator
test_gen_wrapper = DataGeneratorWrapper(
    eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    eog_filelist=os.path.abspath(FLAGS.eog_test_data),
    emg_filelist=os.path.abspath(FLAGS.emg_test_data),
    num_fold=config.num_fold_testing_data,
    data_shape_2=[config.frame_seq_len, config.ndim],
    seq_len=config.sub_seq_len * config.nsubseq,
    nclasses=4,
    artifact_detection=False,
    artifacts_label=3,
    shuffle=False
)

config.nchannel = 3


# Main function
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
            output_loss, total_loss, yhat, score, embeddings = sess.run([net.output_loss, net.loss, net.prediction, net.score, net.embeddings], feed_dict)
            return output_loss, total_loss, yhat, score, embeddings, y_batch

        def _evaluate(gen):
            # Validate the model on the entire data in gen

            score = np.zeros([len(gen.data_index), config.sub_seq_len*config.nsubseq, config.nclasses_model])

            embeddings_list = []
            labels_list = []

            factor = 10

            # use 10x of minibatch size to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_, embeddings_, labels_ = dev_step(x_batch, y_batch)
                for s in range(config.nsubseq):
                    score[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size,
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]
                embeddings_list.append(embeddings_)
                labels_list.append(labels_)
                test_step += 1
            if gen.pointer < len(gen.data_index):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_, score_, embeddings_, labels_ = dev_step(x_batch, y_batch)
                for s in range(config.nsubseq):
                    score[(test_step - 1) * factor * config.batch_size: len(gen.data_index),
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]
                embeddings_list.append(embeddings_)
                labels_list.append(labels_)

            embeddings = np.concatenate(embeddings_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            return score, embeddings, labels


        def evaluate(gen_wrapper):

            N = int(np.sum(gen_wrapper.file_sizes) - (config.sub_seq_len*config.nsubseq - 1)*len(gen_wrapper.file_sizes))
            yhat = np.zeros([N, config.sub_seq_len*config.nsubseq])
            y = np.zeros([N, config.sub_seq_len*config.nsubseq])

            score = np.zeros([N, config.sub_seq_len*config.nsubseq, config.nclasses_model])
            embeddings_all = []
            labels_all = []

            count = 0
            output_loss = 0
            total_loss = 0
            gen_wrapper.new_subject_partition()
            for data_fold in range(config.num_fold_testing_data):
                gen_wrapper.next_fold()
                score_, embeddings_, labels_ = _evaluate(gen_wrapper.gen)

                score[count: count + len(gen_wrapper.gen.data_index)] = score_

                count += len(gen_wrapper.gen.data_index)

                embeddings_all.append(embeddings_)
                labels_all.append(labels_)

            embeddings = np.concatenate(embeddings_all, axis=0)
            labels = np.concatenate(labels_all, axis=0)
            return score, embeddings, labels

        # evaluation on test data
        test_score, test_embeddings, test_labels = evaluate(gen_wrapper=test_gen_wrapper)
        validation_score, validation_embeddings, validation_labels = evaluate(gen_wrapper=validation_gen_wrapper)

        print(f'Validation Embedding shape: {validation_embeddings.shape}')
        print(f'Validation Label shape: {validation_labels.shape}')

        print(f'Test Embedding shape: {test_embeddings.shape}')
        print(f'Test Label shape: {test_labels.shape}')
        # Reshape embeddings
        validation_embeddings = validation_embeddings.reshape(-1, validation_embeddings.shape[-1])
        test_embeddings = test_embeddings.reshape(-1, test_embeddings.shape[-1])

        # Standardize the embeddings using validation set statistics
        validation_mean = np.mean(validation_embeddings, axis=0)
        validation_std = np.std(validation_embeddings, axis=0)

        validation_embeddings = standardize_embeddings(validation_embeddings, validation_mean, validation_std)
        test_embeddings = standardize_embeddings(test_embeddings, validation_mean, validation_std)
        print('After reshape and standardization')
        print(f'Validation Embedding shape: {validation_embeddings.shape}')
        print(f'Validation Label shape: {validation_labels.shape}')

        print(f'Test Embedding shape: {test_embeddings.shape}')
        print(f'Test Label shape: {test_labels.shape}')

        # Apply t-SNE
        embeddings_2d = perform_tsne(validation_embeddings, n_components=2)

        # # Plot t-SNE results
        # plot_tsne(embeddings_2d, validation_labels, output_dir)

        # Fit GMM on clean validation data
        # clean_indices = labels != artifact_label
        gmm = fit_gmm(validation_embeddings, n_components=3)
        plot_density(gmm, validation_embeddings, out_path)

        # Compute the probability scores for ROC-AUC
        scores = gmm.score_samples(test_embeddings)

        # Plot ROC curve and compute AUC
        n_classes = len(np.unique(test_labels))
        plot_roc(test_labels, scores, out_path, n_classes)

        # Fit K-Means
        n_clusters = 3  # Number of clusters for K-Means
        kmeans, cluster_labels = fit_kmeans(validation_embeddings, n_clusters)
        plot_silhouette(validation_embeddings, cluster_labels, out_path)

        # Plot Confusion Matrix
        conf_matrix = confusion_matrix(validation_labels, cluster_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        plot_file = os.path.join(out_path, 'confusion_matrix.png')
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file)
        plt.show()
