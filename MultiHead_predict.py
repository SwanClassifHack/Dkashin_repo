import argparse
from torch.utils.data import DataLoader
import torch
import tqdm
import pandas as pd
import numpy as np
import cv2
import torchvision
import os
from PIL import Image
import models.MultiHead as mh
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
        "-ckptseg",
        "--checkpoint_segment",
        default = './weights/Segmentator.ckpt',
        type = str,
        help = "path to checkpoint",
    )
    parser.add_argument(
        "-ckptmul",
        "--checkpoint_multihead",
        default = './weights/MultiHead.ckpt',
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

segment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])

transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(), 
    torchvision.transforms.Resize(
        size=(224, 224)
    ),
    torchvision.transforms.ToTensor(),  
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mapping = {'разметка_шипун' : 2, 'klikun' : 1, 'разметка_малый':0}
inv_mapping = { 2 : 'разметка_шипун', 1 : 'klikun', 0:'разметка_малый'}

def masking(img, mask):
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    bg = np.ones_like(img, np.uint8) * 255
    bg_mask = cv2.bitwise_not(mask)
    bg_masked = cv2.bitwise_and(bg, bg, mask=bg_mask)
    result = cv2.add(img_masked, bg_masked)
    return result

def find_max_contour(contours):
    max_contour_area = 0
    max_contour = None
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > max_contour_area:
            max_contour_area = contour_area
            max_contour = contour
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, w, h

def find_min_contour(contours):
    min_contour_area = 1e6
    min_contour = None
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_contour_area:
            min_contour_area = contour_area
            min_contour = contour
    x, y, w, h = cv2.boundingRect(min_contour)
    return x, y, w, h

def crop_image(img, mask, x, y, w, h, exp_rate = 1.1):
    if exp_rate  <= 1:
        exp_rate = 1
    dw = int((exp_rate**0.5-1) * w /2)
    dh = int((exp_rate**0.5-1) * h /2)
    x = x - dw if x-dw>=0 else 0
    y = y - dh if y-dh>=0 else 0
    w = w + dw * 2
    h = h + dh * 2    
    return img[y:y+h, x:x+w, ::], mask[y:y+h, x:x+w]

def get_bb(im, mask):
    iim = np.array(im).astype(np.uint8)
    mmask = np.array(mask).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = find_max_contour(contours)

    return x, y, w, h
    
def change_brightness(img, alpha=0.5, beta=85):
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def get_beak_mask(img, mask):
    im_changed = change_brightness(img)
    masked_im = masking(im_changed, mask)
    
    img_hsv = cv2.cvtColor(masked_im, cv2.COLOR_RGB2HSV)
    mask_warm_col = cv2.inRange(img_hsv, (0,50,20), (35,255,255)) # yellow&red
    
    blured = cv2.blur(mask_warm_col, (6,6))
    thresh = cv2.threshold(blured, 100, 255, cv2.THRESH_BINARY)[1]
        
    return thresh
    
def find_head(img, mask):
    img = np.array(img).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    
    x, y, w, h = get_bb(img, mask)
    im_bb, mask_bb = crop_image(img, mask, x, y, w, h)
    
    im_bb = np.array(im_bb).astype(np.uint8)
    mask_bb = np.array(mask_bb).astype(np.uint8)
    
    mask_for_final = masking(im_bb, mask_bb)
    
    beak_mask = get_beak_mask(im_bb, mask_bb)
    #plt.imshow(mask_for_final, cmap='gray', vmin=0, vmax=255)
    x, y, w, h = get_bb(mask_for_final, beak_mask)
    
    ratio = w * h / (im_bb.shape[0] * im_bb.shape[1])
    if ratio == 0:
        return np.zeros((50, 50, 3)), np.zeros((50, 50)), im_bb, mask_bb

    head, head_mask = crop_image(mask_for_final, mask_bb, x, y, w, h, exp_rate=0.2/ratio)

    return head, head_mask, im_bb, mask_bb # или mask_for_final

def process_mask(mask):
    masknp = mask.cpu().detach().numpy()
    result = np.where(masknp > 0.6, 255, 0).astype(np.uint8)
    return result

def process_input(img, dct):
    """ Одна картинка (c, h, w) как после Image.open"""
    imgnp = img.astype(np.uint8)
    masks = process_mask(dct['masks'][dct['scores']>0.9]).squeeze(1)    
    heads = []
    cropped = []
    for i in range(masks.shape[0]):
        hed, hed_mask, imc, imcm = find_head(imgnp, masks[i])
        heads.append(hed)
        cropped.append(imc)
    
    return cropped, heads

def predict(args):
    orig_imgs = os.listdir(args.image_dir)
    imgs = [os.path.join(args.image_dir, i) for i in orig_imgs]
    Segmenter = mh.SegmentModel.load_from_checkpoint(args.checkpoint_segment)
    Segmenter.eval()
    Segmenter.to('cuda')

    Classifier = mh.SwanClassifierBIG.load_from_checkpoint(args.checkpoint_multihead)
    Classifier.to('cuda')
    Classifier.model.to('cuda')
    Classifier.model.eval()
    Classifier.eval()

    crop_arr = []
    head_arr = []
    with torch.no_grad():
        for img_path in tqdm.tqdm(imgs):
            img = Image.open(img_path).convert('RGB')
            tensor_image = segment_transforms(np.array(img))
            res = Segmenter(tensor_image.unsqueeze(0).to('cuda'))
            crops, heads = process_input(np.array(img), res[0])
            crop_arr.append(crops)
            head_arr.append(heads)

    test_dataset = ds.MultiHeadTestSwanDataset(imgs, crop_arr, head_arr, transform=transform_pipe)
    

    predicted_classes = []
    with torch.no_grad():
        for num in tqdm.tqdm(range(len(test_dataset))):
            sample = test_dataset[num]
            image = sample['image']
            crops = sample['crops']
            heads = sample['heads']
            preds = []
            for i in range(len(heads)):
                pred = Classifier.model((image.unsqueeze(0).to('cuda'), 
                                    crops[i].unsqueeze(0).to('cuda'), 
                                    heads[i].unsqueeze(0).to('cuda'))).detach().cpu().numpy()
                preds.append(pred)
            idx = np.argmax(np.array(preds).max(axis = 0))
            predicted_classes.append(inv_mapping[idx])

    names = orig_imgs    
    result = pd.DataFrame({'name': names, 'class' : predicted_classes})
    result['class'] = result['class'].map({'разметка_шипун' : 3, 'klikun' : 2, 'разметка_малый' : 1})
    result.to_csv(args.output_file, sep = ';', encoding='utf-8', index = False)


if __name__ == "__main__":
    args = make_parser().parse_args()
    print(args)
    predict(args)
