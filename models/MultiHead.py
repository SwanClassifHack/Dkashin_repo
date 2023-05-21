import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import lightning.pytorch as pl
import torch
from torchvision import models
import pandas as pd

mapping = {'разметка_шипун' : 2, 'klikun' : 1, 'разметка_малый':0}
inv_mapping = { 2 : 'разметка_шипун', 1 : 'klikun', 0:'разметка_малый'}


def get_point(pred_id, gt_id):
    matr = [[3,-1,-3],
            [-1,3,-3],
            [-3,-3,2]]
    return matr[pred_id][gt_id]

def calc_metric(gt, pred):
    df = pd.DataFrame({'gt': gt, 'pred':pred})
    df['gt'] = df['gt'].map(mapping)
    df['pred'] = df['pred'].map(mapping)
    gt_points = []
    pred_points = []
    for i in df['gt']:
        gt_points.append(get_point(i, i))
    for i,j in zip(df['pred'],df['gt']):
        pred_points.append(get_point(i, j))
    df['gt_points'] = gt_points
    df['pred_points'] = pred_points
    return df


class SegmentModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(SegmentModel, self).__init__()
        
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        self.learning_rate = learning_rate
        self.freeze_layers()
        self.save_hyperparameters()
    
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.backbone.body.parameters():
            param.requires_grad = False

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed to the forward method.")
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        with torch.no_grad():
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.log('validation_loss', losses)
            return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        def func(epoch: int):
            return  2 ** (-epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'interval': 'epoch'
            }
        }
    

class ThreeHeadModel(torch.nn.Module):

    def __init__(self, num_classes, emb_size = 256):
        super(ThreeHeadModel, self).__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        
        hid_dim = emb_size * num_classes
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hid_dim, int(hid_dim / 2)),
                    torch.nn.BatchNorm1d(int(hid_dim / 2)),
                    torch.nn.Linear(int(hid_dim / 2), int(hid_dim / 4)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.2, inplace=False),
                    torch.nn.Linear(int(hid_dim / 4), num_classes)
                    )
        self.classifier = torch.nn.Linear(emb_size * 3, num_classes)
        
        self.head_model = models.mobilenet_v2()
        self.head_model.classifier[1] = torch.nn.Linear(self.head_model.last_channel, emb_size)
        
        self.crop_model = torchvision.models.resnet50(pretrained = True)
        self.crop_model.fc = torch.nn.Linear(in_features=2048, out_features=emb_size)
        
        self.img_model = torchvision.models.resnet50(pretrained = True)
        self.img_model.fc = torch.nn.Linear(in_features=2048, out_features=emb_size)
        

    def forward(self, x):
        img_tensor, crop_tensor, head_tensor = x
        
        img_tensor = self.img_model(img_tensor)
        crop_tensor = self.crop_model(crop_tensor)
        head_tensor = self.head_model(head_tensor)
        
        temp = torch.zeros_like(img_tensor)
        
        output = torch.cat((img_tensor, crop_tensor, head_tensor), dim = 1)
        output = self.classifier(output)
    
        return output
    
class SwanClassifierBIG(pl.LightningModule):
    def __init__(self, learning_rate):
        super(SwanClassifierBIG, self).__init__()
        self.learning_rate = learning_rate
        self.model = ThreeHeadModel(3)
        weights = [1,1,4]
        class_weights = torch.FloatTensor(weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight = class_weights)
        self.inv_mapping = { 2 : 'разметка_шипун', 1 : 'klikun', 0:'разметка_малый'}
        self.save_hyperparameters()
    
    def calc_score(self, preds, labels):
        gt_list = []
        pred_list = []
        max_preds = torch.argmax(preds, dim = 1)
        for pred, label in zip(max_preds, labels):
            pred_list.append(self.inv_mapping[pred.item()])
            gt_list.append(self.inv_mapping[label.item()])
        stat_df = calc_metric(gt_list, pred_list)
        return stat_df['pred_points'].sum()/stat_df['gt_points'].sum()
        
    def training_step(self, batch, batch_idx):
        images = batch['image']
        crops = batch['crop']
        head = batch['head']
        labels = batch['img_class'].to(torch.long)
        preds = self.model(( images, crops, head))
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        self.log('step', self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        crops = batch['crop']
        head = batch['head']
        labels = batch['img_class'].to(torch.long)
        preds = self.model(( images, crops, head))
        loss = self.criterion(preds, labels)
        self.log("validation_loss", loss, sync_dist=True)
        self.log("validation_score", self.calc_score(preds, labels), sync_dist=True)
    
    def forward(self, images, crops, head):
        preds = self.model(( images, crops, head))
        preds = torch.argmax(preds, dim = 1)
        out = []
        for pred in preds:
            out.append(self.inv_mapping[pred.item()])
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        def func(epoch: int):
            return  2 ** (-epoch//4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'interval': 'epoch'
            }
        }