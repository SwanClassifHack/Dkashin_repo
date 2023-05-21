import argparse
from torch.utils.data import DataLoader
import torch
import tqdm
import pandas as pd

from models.CNN import SwanClassifier
from utils.transforms import resnet_test_transforms
import utils.datasets as ds


def make_parser():
    parser = argparse.ArgumentParser("AngleRegressor train parser")
    parser.add_argument(
        "-dir",
        "--image_dir",
        default = None,
        type = str,
        help = "path to derictory with crops",
    )
    parser.add_argument(
        "-out",
        "--output_file",
        default = './results.csv',
        type = str,
        help = "csv file to store predicts",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        default = './weights/CNN.ckpt',
        type = str,
        help = "path to checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        default = 32,
        type = int,
        help = "num imgs in batch",
    )
    parser.add_argument(
        "--device",
        default = 'cuda',
        type = str,
        help = "device name",
    )
    return parser



def predict(args):
    pl_model = SwanClassifier.load_from_checkpoint(args.checkpoint)
    pl_model.to(args.device)
    pl_model.eval()

    test_dataset = ds.SwanTestDataset(args.image_dir, resnet_test_transforms, ds.mapping)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=None, pin_memory=False)
    predicts = []
    names = []
    with torch.no_grad():
        for i in tqdm.tqdm(test_loader):
            pred = pl_model(i['image'].to('cuda'))
            name = i['image_name']
            predicts += pred
            names += name
    
    result = pd.DataFrame({'name':names, 'class':predicts})
    result['class'] = result['class'].map({'разметка_шипун' : 3, 'klikun' : 2, 'разметка_малый' : 1})
    result.to_csv(args.output_file, sep = ';', encoding='utf-8', index = False)


if __name__ == "__main__":
    args = make_parser().parse_args()
    print(args)
    predict(args)
