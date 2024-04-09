import os
import torch
import argparse
import torchaudio

from audio_dataset import AudioDataset

# parsing model directory and model name to load
parser = argparse.ArgumentParser()
parser.add_argument('-model_dir', nargs=1, type=str, default='0')
parser.add_argument('-model_name', nargs=1, type=str, default='0')
args = parser.parse_args()

model_dir = args.model_dir[0]
model_name =  args.model_name[0] 
# model_path = os.path.join(model_dir, model_name)
model_path = './saved_models/vgg19/007/10_0push0.9571.pth'

# loading the model
ppnet = torch.load(model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# loading the test set
from settings import test_dir, test_annotation_dir, sample_rate, num_samples, n_fft, hop_length, n_mels, power_or_db, test_batch_size

mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

test_dataset = AudioDataset(test_annotation_dir, test_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db)
print(f"There are {len(test_dataset)} samples in the test dataset.")

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False
)

for x, y in test_loader:
    print("y", y)
    # print(data)


# img_variable = test_dataset._get_signal_from_audio_path(test_image_path)
# img_tensor = torch.tensor(img_variable)
# img_variable = Variable(img_tensor.unsqueeze(0))
# x = img_variable.cuda()


# labels_test = torch.tensor([test_image_label])

# logits, min_distances = ppnet_multi(x)
# conv_output, distances = ppnet.push_forward(x)
# prototype_activations = ppnet.distance_2_similarity(min_distances)
# prototype_activation_patterns = ppnet.distance_2_similarity(distances)
# if ppnet.prototype_activation_function == 'linear':
#     prototype_activations = prototype_activations + max_dist
#     prototype_activation_patterns = prototype_activation_patterns + max_dist

# tables = []
# for i in range(logits.size(0)):
#     tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
#     log(str(i) + ' ' + str(tables[-1]))

# idx = 0
# predicted_cls = tables[idx][0]
# correct_cls = test_image_label#tables[idx][1]
# log('Predicted: ' + str(predicted_cls))
# log('Actual: ' + str(correct_cls))