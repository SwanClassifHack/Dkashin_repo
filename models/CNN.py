import torch
import torchvision
import lightning.pytorch as pl
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

class SwanClassifier(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = torchvision.models.resnet50(pretrained = True)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=3)
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
        crops = batch['image']
        labels = batch['img_class'].to(torch.long)
        preds = self.model(crops)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        self.log('step', self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        crops = batch['image']
        labels = batch['img_class'].to(torch.long)
        preds = self.model(crops)
        loss = self.criterion(preds, labels)
        self.log("validation_loss", loss, sync_dist=True)
        self.log("validation_score", self.calc_score(preds, labels), sync_dist=True)
    
    def forward(self, images):
        preds = self.model(images)
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