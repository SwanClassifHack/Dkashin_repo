# Kekichi team solution for 4th case

## setup 
```
pip install -r requirements.txt
```
## using examples
### Multihead model
```
python MultiHead_predict.py -dir <image dir> -ckptseg ./weights/Segmentator.ckpt -ckptmul ./weights/MultiHead.ckpt -out out.csv
```
### Baseline
```
python predict.py -dir <image dir> -ckpt ./weights/CNN.ckpt
```
##
weights for model: https://drive.google.com/drive/folders/1krUPyjCZRn67Bob8zx1Vh8sx3MDUIiI8?usp=sharing