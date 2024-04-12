##### MODEL AND DATA LOADING
import torch
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from PIL import Image
import re

import os
import copy
# from skimage.transform import resize
from helpers import makedir, find_high_activation_crop
# import model
# import push
import train_and_test as tnt
# import save
from log import create_logger
# from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse
import pandas as pd
import ast
# import png

import time
import torchaudio
from audio_dataset import AudioDataset

k=3

# specify the test image to be analyzed
parser = argparse.ArgumentParser()
parser.add_argument('-test_img_name', nargs=1, type=str, default='0')
parser.add_argument('-test_img_dir', nargs=1, type=str, default='0')
parser.add_argument('-test_img_label', nargs=1, type=int, default='-1')
parser.add_argument('-test_model_dir', nargs=1, type=str, default='0')
parser.add_argument('-test_model_name', nargs=1, type=str, default='0')
args = parser.parse_args()

test_image_dir = args.test_img_dir[0]
test_image_name =  args.test_img_name[0] #'DP_AJOD_196544.npy' # 'DP_AAPR_R_MLO_3#0.npy' # 
test_image_label = args.test_img_label[0]

test_image_path = os.path.join(test_image_dir, test_image_name)

# load the model
check_test_accu = False

load_model_dir = args.test_model_dir[0] #'/usr/xtmp/mammo/alina_saved_models/vgg16/finer_1118_top2percent_randseed=1234/'
load_model_name = args.test_model_name[0] # '100_9push0.9258.pth'

#if load_model_dir[-1] == '/':
#    model_base_architecture = load_model_dir.split('/')[-3]
#    experiment_run = load_model_dir.split('/')[-2]
#else:
#    model_base_architecture = load_model_dir.split('/')[-2]
#    experiment_run = load_model_dir.split('/')[-1]

model_base_architecture = load_model_dir.split('/')[-3]
experiment_run = '/'.join(load_model_dir.split('/')[-2:])

save_analysis_path = os.path.join(load_model_dir, test_image_name)
makedir(save_analysis_path)
print(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log')) #logger fails on Colab
# def log(string_here):
#     print(string_here)
# def logclose():
#     pass

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)
log('test image to analyse' + test_image_path)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

# normalize = transforms.Normalize(mean=mean, std=std)
# load the test data and check test accuracy
from settings import test_dir, test_annotation_dir, sample_rate, num_samples, n_fft, hop_length, n_mels, power_or_db

mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

# test dataset
test_dataset = AudioDataset(test_annotation_dir, test_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db)
print(f"There are {len(test_dataset)} samples in the test dataset.")

# get the start time
st = time.time()

if check_test_accu:
    test_batch_size = 100

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False
    )
    log('test set size: {0}'.format(len(test_loader.dataset)))

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=print)

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]
prototype_img_identity = np.array([identity -1 for identity in prototype_img_identity]) 
# rimuovo -1 da ciò che è stato salvato dal train perchè le label le img identity non le salva a partire da 0
##  (usa i nomi delle cartelle che partono da 1)

num_classes = len(set(prototype_img_identity))

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()

if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    print('image index {0} in batch'.format(index))
    img_copy = img_copy[0]
    img_copy = img_copy.detach().cpu().numpy()
    img_copy = np.transpose(img_copy, [1,2,0])

    plt.imshow(img_copy, origin='lower')
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    
    # plt.imsave(fname, img_copy)
    return img_copy

def save_prototype(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    #plt.axis('off')
    # plt.imsave(fname, p_img)
    plt.imshow(p_img, origin='lower')
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    
def save_prototype_self_activation(fname, epoch, index):
    # p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original_bw_with_self_act'+str(index)+'.png'))
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    # plt.imsave(fname, p_img)
    plt.imshow(p_img, origin='lower')
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')

def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 225)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    # cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
    #               color, thickness=2)
    # p_img_rgb = p_img_bgr[...,::-1]
    # p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    # plt.imsave(fname, p_img_rgb)
    # plt.imshow(p_img_rgb)
    # plt.axis('off')
    # plt.savefig(fname, bbox_inches='tight')


    np.save(fname.split('.png')[0] + 'p_img_bgr.png', p_img_bgr)
    np.save(fname.split('.png')[0] + 'bbox_height_start.png', bbox_height_start)
    np.save(fname.split('.png')[0] + 'bbox_height_end.png', bbox_height_end)
    np.save(fname.split('.png')[0] + 'bbox_width_start.png', bbox_width_start)
    np.save(fname.split('.png')[0] + 'bbox_width_end.png', bbox_width_end)


    p_img_rgb_copy = p_img_bgr.copy()
    img_rect = np.ones((p_img_rgb_copy.shape[0], p_img_rgb_copy.shape[1], 3), np.uint8) * 125
    cv2.rectangle(img_rect, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, 2)

    plt.imshow(p_img_rgb_copy)
    plt.imshow(img_rect, alpha=0.5)
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')

def save_prototype_full_size(fname, epoch, index,
                            color=(0, 255, 255)):
    
    p_full_size = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.npy'))
    np.save(fname[:-4] + '.npy', p_full_size)

    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    # plt.imsave(fname, p_img_rgb)
    plt.imshow(p_img_rgb)
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 225)):
    
    # np.save(fname.split('.png')[0] + 'img_rgb.png', img_rgb)
    # np.save(fname.split('.png')[0] + 'bbox_height_start.png', bbox_height_start)
    # np.save(fname.split('.png')[0] + 'bbox_height_end.png', bbox_height_end)
    # np.save(fname.split('.png')[0] + 'bbox_width_start.png', bbox_width_start)
    # np.save(fname.split('.png')[0] + 'bbox_width_end.png', bbox_width_end)

    # img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    # cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    # img_rgb_uint8 = img_bgr_uint8[...,::-1]
    # img_rgb_float = np.float32(img_rgb_uint8) / 255

    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    # plt.imsave(fname, img_rgb_float)
    # plt.imshow(img_rgb_float)

    img_rgb_copy = img_rgb.copy()
    img_rect = np.ones((img_rgb.shape[0], img_rgb.shape[1], 3), np.uint8) * 125
    cv2.rectangle(img_rect, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, 2)

    plt.imshow(img_rgb_copy, origin='lower')
    plt.imshow(img_rect, alpha=0.5, origin='lower')

    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')


img_variable = test_dataset._get_signal_from_audio_path(test_image_path)
img_tensor = torch.tensor(img_variable)
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()
labels_test = torch.tensor([test_image_label])

logits, min_distances = ppnet_multi(images_test)
conv_output, distances = ppnet.push_forward(images_test)
prototype_activations = ppnet.distance_2_similarity(min_distances)
prototype_activation_patterns = ppnet.distance_2_similarity(distances)
if ppnet.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0] 
correct_cls = test_image_label #tables[idx][1]
log('Predicted: ' + str(predicted_cls + 1)) # +1 predicted due to -1 assignment to labels in train_and_test.py
log('Actual: ' + str(correct_cls))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
log('***************************************************************')
log('***************************************************************')
print("prototype info", prototype_info)
print("prototype activations", prototype_activations)
print("prototype_activation_patterns", prototype_activation_patterns)
print("ppnet.last_layer.weight", ppnet.last_layer.weight)

makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))
max_act = 0
log('Most activated 5 prototypes of this image:')
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
print("array act", array_act)
print("sorted_indices_act", sorted_indices_act)

# for i in range(1,6):
#     log('top {0} activated prototype for this image:'.format(i))
#     save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
#                                 'top-%d_activated_prototype.png' % i),
#                                 start_epoch_number, sorted_indices_act[-i].item())
#     save_prototype_full_size(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
#                             'top-%d_activated_prototype_full_size.png' % i),
#                             epoch=start_epoch_number,
#                             index=sorted_indices_act[-i].item(),
#                             color=(0, 255, 255))
#     save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
#                                         'top-%d_activated_prototype_in_original_pimg.png' % i),
#                                         epoch=start_epoch_number,
#                                         index=sorted_indices_act[-i].item(),
#                                         bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
#                                         bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
#                                         bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
#                                         bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
#                                         color=(0, 255, 255))
#     save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
#                                                 'top-%d_activated_prototype_self_act.png' % i),
#                                    start_epoch_number, sorted_indices_act[-i].item())
#     log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    
#     print("First term", prototype_max_connection[sorted_indices_act[-i].item()])
#     print("Second term", prototype_img_identity[sorted_indices_act[-i].item()])

#     log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()] + 1)) # qui è +1 solo per logging reasons (1 class label, lui nella pnet ce l'ha da 0)
#     if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
#         log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()] + 1)) # qui è +1 solo per logging reasons (1 class label, lui nella pnet ce l'ha da 0)
#     log('activation value (similarity score): {0}'.format(array_act[-i]))

#     f = open(save_analysis_path + '/most_activated_prototypes/' + 'top-' + str(i) + '_activated_prototype.txt', "w")
#     f.write('similarity: {0:.3f}\n'.format(array_act[-i].item()))
#     f.write('last layer connection with predicted class: {0} \n'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
#     f.write('proto index:')
#     f.write(str(sorted_indices_act[-i].item()) + '\n')
#     for class_id_ in range(num_classes):
#         f.write(f'proto connection to class {class_id_ + 1}:') # +1 for loggning reasons, to match the class folder target label
#         f.write(str(ppnet.last_layer.weight[class_id_][sorted_indices_act[-i].item()]) + '\n')
#     f.close()
#     log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
#     activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
#     upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
#                                               interpolation=cv2.INTER_CUBIC)
    
#     # show the most highly activated patch of the image by this prototype
#     high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
#     high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
#                                   high_act_patch_indices[2]:high_act_patch_indices[3], :]
#     log('most highly activated patch of the chosen image by this prototype:'
#         + str(os.path.join(save_analysis_path, 'most_activated_prototypes',
#             'most_highly_activated_patch_by_top-%d_prototype.png' % i)))
#     #plt.axis('off')
#     # plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',  'most_highly_activated_patch_by_top-%d_prototype.png' % i), high_act_patch)
    
#     plt.imshow(high_act_patch, origin='lower')
#     plt.axis('off')
#     plt.savefig(os.path.join(save_analysis_path, 'most_activated_prototypes',  'most_highly_activated_patch_by_top-%d_prototype.png' % i), bbox_inches='tight')

#     log('most highly activated patch by this prototype shown in the original image:'
#         + str(os.path.join(save_analysis_path, 'most_activated_prototypes',
#             'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i)))
#     imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
#                             'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
#                      img_rgb=original_img,
#                      bbox_height_start=high_act_patch_indices[0],
#                      bbox_height_end=high_act_patch_indices[1],
#                      bbox_width_start=high_act_patch_indices[2],
#                      bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    
#     # show the image overlayed with prototype activation map
#     rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
#     rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
#     heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[...,::-1]
#     overlayed_img = 0.5 * original_img + 0.3 * heatmap
#     log('prototype activation map of the chosen image:' + str(os.path.join(save_analysis_path, 'most_activated_prototypes',
#         'prototype_activation_map_by_top-%d_prototype.png' % i)))
#     #plt.axis('off')
#     # plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype.png' % i), overlayed_img)

#     plt.imshow(original_img, cmap='gray', origin='lower')
#     plt.imshow(heatmap, alpha=0.5, origin='lower')
#     plt.axis('off')
#     plt.savefig(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype.png' % i), bbox_inches='tight')

#     # show the image overlayed with different normalized prototype activation map
#     rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    
#     # get the max activation of any proto on this image (works because we start with highest act, must be on rescale)
#     if np.amax(rescaled_activation_pattern) > max_act:
#         max_act = np.amax(rescaled_activation_pattern)

#     rescaled_activation_pattern = rescaled_activation_pattern / max_act
#     heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[...,::-1]
#     overlayed_img = 0.5 * original_img + 0.3 * heatmap
#     #plt.axis('off')
#     log('normalized prototype activation map of the chosen image:' 
#          + str(os.path.join(save_analysis_path, 'most_activated_prototypes',
#                 'prototype_activation_map_by_top-%d_prototype_normed.png' % i)))
#     # plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype_normed.png' % i), overlayed_img)

#     plt.imshow(original_img, cmap='gray', origin='lower')
#     plt.imshow(heatmap, alpha=0.5, origin='lower')
#     plt.axis('off')
#     plt.savefig(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype_normed.png' % i), bbox_inches='tight')

#     log('--------------------------------------------------------------')
# log('***************************************************************')
# log('***************************************************************')
# log('***************************************************************')


#### PROTOTYPES FROM TOP-k CLASSES
log('Prototypes from top-%d classes:' % k)
topk_logits, topk_classes = torch.topk(logits[idx], k=k)

print("logits", logits)
print("topk_logits", topk_logits)
print("topk_classes", topk_classes)

for i,c in enumerate(topk_classes.detach().cpu().numpy()):
    print("i", i)
    print("c", c)

    makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

    log('top %d predicted class: %d' % (i+1, c + 1)) # c+1 predicted class for the same train_and_test reason 
    log('logit of the class: %f' % topk_logits[i])

    print("ppnet prototype class identity", ppnet.prototype_class_identity.detach().cpu().numpy())

    class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]

    print("class_prototype_indices", class_prototype_indices)
    
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    
    print("class_prototype_activations", class_prototype_activations)

    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    # cut to only top 5 prototypes for each of the top k classes
    # sorted_indices_cls_act = sorted_indices_cls_act[0:5]

    print("sorted_indices_cls_act", sorted_indices_cls_act)
    
    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_prototype.png' % prototype_cnt),
                                    start_epoch_number, 
                                    prototype_index)
        save_prototype_full_size(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                            'top-%d_activated_prototype_full_size.png' % prototype_cnt),
                                            epoch=start_epoch_number,
                                            index=prototype_index,
                                            color=(0, 255, 255))
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                            'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                            epoch=start_epoch_number,
                                            index=prototype_index,
                                            bbox_height_start=prototype_info[prototype_index][1],
                                            bbox_height_end=prototype_info[prototype_index][2],
                                            bbox_width_start=prototype_info[prototype_index][3],
                                            bbox_width_end=prototype_info[prototype_index][4],
                                            color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 
                                                    'top-%d_class_prototypes' % (i+1),
                                                    'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                                    start_epoch_number, 
                                                    prototype_index)
        log('prototype index: {0}'.format(prototype_index))
        log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index] + 1)) # qui è +1 solo per logging reasons (1 class label, lui nella pnet ce l'ha da 0)
        if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index] + 1)) # qui è +1 solo per logging reasons (1 class label, lui nella pnet ce l'ha da 0)
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
        log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))

        print("ppnet.last_layer.weight", ppnet.last_layer.weight)
        print("ppnet.last_layer.weight[c]", ppnet.last_layer.weight[c])
        
        activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                  interpolation=cv2.INTER_CUBIC)
       
        #################
        #### logging ####
        #################

        f = open(save_analysis_path + '/top-' + str(i+1) + '_class_prototypes/' + 'top-' + str(prototype_cnt) + '_activated_prototype.txt', "w")
        f.write('similarity: {0:.3f}\n'.format(prototype_activations[idx][prototype_index]))
        f.write('last layer connection: {0:.3f}\n'.format(ppnet.last_layer.weight[c][prototype_index]))
        f.write('proto index: ' + str(prototype_index) + '\n')
        for class_id_ in range(num_classes):
            f.write(f'proto connection to class {class_id_ + 1}:') # +1 for loggning reasons, to match the class folder target label
            f.write(str(ppnet.last_layer.weight[class_id_][prototype_index]) + '\n')
        f.close()
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:' + 
            str(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
              'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt)))
        #plt.axis('off')
        # plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),  'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt), high_act_patch)
        
        plt.imshow(high_act_patch, origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),  'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt), bbox_inches='tight')
        
        log('most highly activated patch by this prototype shown in the original image:' 
            + str(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
               'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt)))
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                           'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:'
            + str(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                        'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt)))
        #plt.axis('off')
        # plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt), overlayed_img)

        plt.imshow(original_img, cmap='gray', origin='lower')
        plt.imshow(heatmap, alpha=0.5, origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt), bbox_inches='tight')

        # show the image overlayed with differently normed prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / max_act
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('normalized prototype activation map of the chosen image:'
           + str(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                'prototype_activation_map_by_top-%d_prototype_normed.png' % prototype_cnt)))
        #plt.axis('off')
        # plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype_normed.png' % prototype_cnt), overlayed_img)
        
        plt.imshow(original_img, cmap='gray', origin='lower')
        plt.imshow(heatmap, alpha=0.5, origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype_normed.png' % prototype_cnt), bbox_inches='tight')

        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')
    log('***************************************************************')


if predicted_cls + 1 == correct_cls:
    log('Prediction is correct.')
else:
    log('Prediction is wrong.')
print("saved in ", save_analysis_path)


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
log('Execution time (seconds): ' + str(elapsed_time / 60))

logclose()