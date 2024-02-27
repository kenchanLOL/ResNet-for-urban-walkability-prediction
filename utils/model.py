import torch
import torchvision
from collections import OrderedDict

def get_pretrained_resnet(device, model_path="models/resnet50_best.pth"):
    feature = torchvision.models.resnet50(pretrained = False, num_classes = 365)
    feature =  feature.to(device)
    pre_state_dict = OrderedDict()
    origin_state_dict = torch.load(model_path)["state_dict"]
    finding_masks = False

    for k, v in origin_state_dict.items():
        if k[:7] != 'module.':
            continue
        name = k[7:]
        pre_state_dict[name] = v

    pre_list = sorted(list(pre_state_dict.keys()))
    now_state_dict = feature.state_dict()
    now_list = sorted(list(now_state_dict.keys()))
    ind = 0
    for i in range(len(now_list)):
        key = now_list[i]
        if key.split('.')[-1] == 'mask':
            if finding_masks == False:
                now_state_dict[key] = torch.ones(now_state_dict[key].shape, dtype=torch.float)
        else:
            now_state_dict[key] = pre_state_dict[pre_list[ind]]
            ind = ind + 1
    feature.load_state_dict(now_state_dict)
    # Freeze the pretrain model
    for name, layer in feature.named_parameters():
        if "bn" not in name:
            layer.requires_grad = False
    feature = torch.nn.Sequential(*(list(feature.children())[:-1]))
    return feature

class WRM(torch.nn.Module):
    def __init__(self, global_feature, patch_feature, device):
        super().__init__()
        self.device = device
        # Global
        self.global_feature = global_feature
        self.global_flatten = torch.nn.Flatten()
        self.global_dropout1 = torch.nn.Dropout(0.5)
        self.global_fc1 = torch.nn.Linear(512*2, 256)
        self.global_dropout2 = torch.nn.Dropout(0.5)
        self.global_fc2 = torch.nn.Linear(256, 2)
        self.global_softmax = torch.nn.Softmax(dim = -1)
        # Patch
        self.patch_feature = patch_feature
        self.patch_flatten = torch.nn.Flatten()
        self.patch_dropout1 = torch.nn.Dropout(0.5)
        self.patch_fc1 = torch.nn.Linear(512*5*2, 512)
        self.patch_dropout2 = torch.nn.Dropout(0.5)
        self.patch_fc2 = torch.nn.Linear(512, 2)
        self.patch_softmax = torch.nn.Softmax(dim = -1)
        # Score
        self.score_flatten = torch.nn.Flatten()
        self.score_dropout1 = torch.nn.Dropout(0.5)
        self.score_fc1 = torch.nn.Linear(512*6, 256)
        self.score_dropout2 = torch.nn.Dropout(0.5)
        self.score_fc2 = torch.nn.Linear(256, 1)
        self.score_sigmoid = torch.nn.Sigmoid()

        self.projection_global_flatten = torch.nn.Flatten()
        self.projection_global_dropout1 = torch.nn.Dropout(0.5)
        self.projection_global_fc = torch.nn.Linear(2048,512) #[V3]
        self.projection_patch_dropout1 = torch.nn.Dropout(0.5)
        self.projection_patch_fc = torch.nn.Linear(2048,512) #[V3]
    def forward(self, x_1 = None, x_2 = None):
        if self.training:
            # LEFT
            left_global = x_1[0]
            left_patch = x_1[1]
            batch_size = left_patch.shape[0]
            left_patch = left_patch.reshape(-1, 3, left_patch.shape[-2], left_patch.shape[-1]) 
            left_global = self.global_feature(left_global)
            left_global = self.projection_global_flatten(left_global)
            left_global = self.projection_global_dropout1(left_global)
            left_global = self.projection_global_fc(left_global)
            left_patch = self.patch_feature(left_patch)
            left_patch = left_patch.reshape(batch_size, -1, left_patch.shape[-3]) 
            # left_patch = self.projection_flatten(left_patch)
            left_patch = self.projection_patch_dropout1(left_patch)
            left_patch = self.projection_patch_fc(left_patch)
            #RIGHT
            right_global = x_2[0]
            right_patch = x_2[1]
            right_patch = right_patch.reshape(-1, 3, right_patch.shape[-2], right_patch.shape[-1]) 
            right_global = self.global_feature(right_global)
            right_global = self.projection_global_flatten(right_global)
            right_global = self.projection_global_dropout1(right_global)
            right_global = self.projection_global_fc(right_global)
            right_patch = self.patch_feature(right_patch)
            right_patch = right_patch.reshape(batch_size, -1, right_patch.shape[-3]) 
            # right_patch = self.projection_flatten(right_patch)
            right_patch = self.projection_patch_dropout1(right_patch)
            right_patch = self.projection_patch_fc(right_patch)
            #CONCAT
            # print(left_global.shape, left_patch.shape)
            # concat_global = torch.cat((left_global, right_global), 1).to(self.device)
            # concat_patch = torch.cat((left_patch, right_patch), 1).to(self.device)
            # # print(concat_global.shape, concat_patch.shape)
            # #GLOBAL
            # concat_global = self.global_flatten(concat_global)
            # concat_global = self.global_dropout1(concat_global)
            # concat_global = self.global_fc1(concat_global)
            # concat_global = self.global_dropout2(concat_global)
            # concat_global = self.global_fc2(concat_global)
            # y_global = self.global_softmax(concat_global)
            # #PATCH
            # concat_patch = self.patch_flatten(concat_patch)
            # concat_patch = self.patch_dropout1(concat_patch)
            # concat_patch = self.patch_fc1(concat_patch)
            # concat_patch = self.patch_dropout2(concat_patch)
            # concat_patch = self.patch_fc2(concat_patch)
            # y_patch = self.patch_softmax(concat_patch)
            #SCORE
            concat_left = torch.cat((left_global.reshape(batch_size, 1, -1), left_patch), 1).to(self.device)
            concat_right = torch.cat((right_global.reshape(batch_size, 1, -1), right_patch), 1).to(self.device)

            concat_left = self.score_flatten(concat_left)
            concat_left = self.score_dropout1(concat_left)
            concat_left = self.score_fc1(concat_left)
            concat_left = self.score_dropout2(concat_left)
            score_left = self.score_fc2(concat_left)
            # score_left = self.score_sigmoid(concat_left)

            concat_right = self.score_flatten(concat_right)
            concat_right = self.score_dropout1(concat_right)
            concat_right = self.score_fc1(concat_right)
            concat_right = self.score_dropout2(concat_right)
            score_right = self.score_fc2(concat_right)
            # score_right = self.score_sigmoid(concat_right)

            y_score = score_right - score_left
            y_score = self.score_sigmoid(y_score)
            return score_left, score_right, y_score
        else: # inference (only one image is inputted)
            assert x_2 == None, "x_2 should be None during inference"
            x_global = x_1[0]
            x_patch = x_1[1]
            batch_size = x_patch.shape[0]
            x_patch = x_patch.reshape(-1, 3, x_patch.shape[-2], x_patch.shape[-1]) 
            x_global = self.global_feature(x_global)
            x_global = self.projection_global_flatten(x_global)
            x_global = self.projection_global_dropout1(x_global)
            x_global = self.projection_global_fc(x_global)
            x_patch = self.patch_feature(x_patch)
            x_patch = x_patch.reshape(batch_size, -1, x_patch.shape[-3]) 
            x_patch = self.projection_patch_dropout1(x_patch)
            x_patch = self.projection_patch_fc(x_patch)
            #SCORE
            concat_x = torch.cat((x_global.reshape(batch_size, 1, -1), x_patch), 1).to(self.device)
            concat_x = self.score_flatten(concat_x)
            concat_x = self.score_dropout1(concat_x)
            concat_x = self.score_fc1(concat_x)
            concat_x = self.score_dropout2(concat_x)
            score_x = self.score_fc2(concat_x)
            return score_x
####################################################################
# class WRM_v1(torch.nn.Module):
#     def __init__(self, global_feature, patch_feature):
#         super().__init__()
#         # Global
#         self.global_feature = global_feature
#         self.global_flatten = torch.nn.Flatten()
#         self.global_dropout1 = torch.nn.Dropout(0.5)
#         self.global_fc1 = torch.nn.Linear(2048*2, 256)
#         self.global_dropout2 = torch.nn.Dropout(0.5)
#         self.global_fc2 = torch.nn.Linear(256, 2)
#         self.global_softmax = torch.nn.Softmax(dim = -1)
#         # Patch
#         self.patch_feature = patch_feature
#         self.patch_flatten = torch.nn.Flatten()
#         self.patch_dropout1 = torch.nn.Dropout(0.5)
#         self.patch_fc1 = torch.nn.Linear(2048*5*2, 512)
#         self.patch_dropout2 = torch.nn.Dropout(0.5)
#         self.patch_fc2 = torch.nn.Linear(512, 2)
#         self.patch_softmax = torch.nn.Softmax(dim = -1)
#         # Score
#         self.score_flatten = torch.nn.Flatten()
#         self.score_dropout1 = torch.nn.Dropout(0.5)
#         self.score_fc1 = torch.nn.Linear(2048*6, 256)
#         self.score_dropout2 = torch.nn.Dropout(0.5)
#         self.score_fc2 = torch.nn.Linear(256, 1)
#         self.score_sigmoid = torch.nn.Sigmoid()

#     def forward(self, x_1 = None, x_2 = None):
#         if self.training:
#             # LEFT
#             left_global = x_1[0]
#             left_patch = x_1[1]
#             batch_size = left_patch.shape[0]
#             left_patch = left_patch.reshape(-1, 3, left_patch.shape[-2], left_patch.shape[-1]) 
#             left_global = self.global_feature(left_global)
#             left_patch = self.patch_feature(left_patch)
#             left_patch = left_patch.reshape(batch_size, -1, left_patch.shape[-3]) 
#             #RIGHT
#             right_global = x_2[0]
#             right_patch = x_2[1]
#             right_patch = right_patch.reshape(-1, 3, right_patch.shape[-2], right_patch.shape[-1]) 
#             right_global = self.global_feature(right_global)
#             right_patch = self.patch_feature(right_patch)
#             right_patch = right_patch.reshape(batch_size, -1, right_patch.shape[-3]) 
#             #CONCAT
#             # print(left_global.shape, left_patch.shape)
#             concat_global = torch.cat((left_global, right_global), 1).to(device)
#             concat_patch = torch.cat((left_patch, right_patch), 1).to(device)
#             # print(concat_global.shape, concat_patch.shape)
#             #GLOBAL
#             concat_global = self.global_flatten(concat_global)
#             concat_global = self.global_dropout1(concat_global)
#             concat_global = self.global_fc1(concat_global)
#             concat_global = self.global_dropout2(concat_global)
#             concat_global = self.global_fc2(concat_global)
#             y_global = self.global_softmax(concat_global)
#             #PATCH
#             concat_patch = self.patch_flatten(concat_patch)
#             concat_patch = self.patch_dropout1(concat_patch)
#             concat_patch = self.patch_fc1(concat_patch)
#             concat_patch = self.patch_dropout2(concat_patch)
#             concat_patch = self.patch_fc2(concat_patch)
#             y_patch = self.patch_softmax(concat_patch)
#             #SCORE
#             concat_left = torch.cat((left_global.reshape(batch_size, 1, -1), left_patch), 1).to(device)
#             concat_right = torch.cat((right_global.reshape(batch_size, 1, -1), right_patch), 1).to(device)

#             concat_left = self.score_flatten(concat_left)
#             concat_left = self.score_dropout1(concat_left)
#             concat_left = self.score_fc1(concat_left)
#             concat_left = self.score_dropout2(concat_left)
#             score_left = self.score_fc2(concat_left)
#             score_left = self.score_sigmoid(score_left)

#             concat_right = self.score_flatten(concat_right)
#             concat_right = self.score_dropout1(concat_right)
#             concat_right = self.score_fc1(concat_right)
#             concat_right = self.score_dropout2(concat_right)
#             score_right = self.score_fc2(concat_right)
#             score_right = self.score_sigmoid(score_right)

#             # y_score = score_right - score_left
#             # y_score = self.score_sigmoid(y_score)
#             return y_global, y_patch, score_left, score_right
#         else: # inference (only one image is inputted)
#             assert x_2 == None, "x_2 should be None during inference"
#             x_global = x_1[0]
#             x_patch = x_1[1]
#             batch_size = x_patch.shape[0]
#             x_patch = x_patch.reshape(-1, 3, x_patch.shape[-2], x_patch.shape[-1]) 
#             x_global = self.global_feature(x_global)
#             x_global = self.projection_global_flatten(x_global)
#             x_global = self.projection_global_dropout1(x_global)
#             x_global = self.projection_global_fc(x_global)
#             x_patch = self.patch_feature(x_patch)
#             x_patch = x_patch.reshape(batch_size, -1, x_patch.shape[-3]) 
#             x_patch = self.projection_patch_dropout1(x_patch)
#             x_patch = self.projection_patch_fc(x_patch)
#             #SCORE
#             concat_x = torch.cat((x_global.reshape(batch_size, 1, -1), x_patch), 1).to(self.device)
#             concat_x = self.score_flatten(concat_x)
#             concat_x = self.score_dropout1(concat_x)
#             concat_x = self.score_fc1(concat_x)
#             concat_x = self.score_dropout2(concat_x)
#             score_x = self.score_fc2(concat_x)
#             return score_x
