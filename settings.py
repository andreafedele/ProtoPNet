base_architecture = 'vgg19'
img_size = 225
img_channels = 1
prototype_shape = (600, 128, 1, 1)
num_classes = 60
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path = './datasets/audiomnist_split/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train_augmented/'
train_annotation_dir = data_path + 'annotations_train.csv'
test_annotation_dir = data_path + 'annotations_test.csv'
train_push_annotation_dir = data_path + 'annotations_train_augmented.csv'
train_batch_size = 10
test_batch_size = 30
train_push_batch_size = 25

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
last_layer_convex_optimizations = 3 # 20 si hard-coded by default in ProtoPNet

# --- train early stopping ---
es_last_n_epochs = 5 # last n epochs to watch in order to verify if convergence was reached on train accuracy
es_conv_threshold = 0.01 # convergency threshold, if std(acc(last_n_epochs)) < thr then convergence has been reached

# --- audio input data-type integration ---
# audio sample rate
sample_rate = 41000
num_samples = 41000

# spectrogram conversion
n_fft = 4096
hop_length = 183
n_mels = 225

# power spectrogram or dB units spect
power_or_db = 'p' # power spectrogram 'p', decibel dB units 'd'
