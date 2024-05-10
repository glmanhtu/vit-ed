# ViT-ED network  
  
The training code is adapted from https://github.com/microsoft/Swin-Transformer  

## Compatibility  
The dependencies can be installed by running the following command:  
```bash  
python3 -m pip install -r requirements.txt
```
  
## Problem 1: Puzzle solver 
  
### Dataset  
The DIV2K dataset:  
From the official page: https://data.vision.ee.ethz.ch/cvl/DIV2K/  
Link to download the training set [here](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) and validation set [here](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip).
Generate training & validating patches:
```bash
mkdir /path/to/datasets/DIV2K_Patches
PYTHONPATH=$PYTHONPATH:. python3 scripts/generate_patches.py --data-path /path/to/DIV2K_train_HR --output-path /path/to/DIV2K_Patches/DIV2K_train_HR
PYTHONPATH=$PYTHONPATH:. python3 scripts/generate_patches.py --data-path /path/to/DIV2K_valid_HR --output-path /path/to/DIV2K_Patches/DIV2K_valid_HR
```
  
### Training  

```bash
python3 -u -m torch.distributed.run --nproc_per_node 1 --master_port 13556 main.py --cfg configs/puzzle/div2k_erosion7_4bin_patch8_64.yaml --data-path /path/to/DIV2K_Patches --batch-size 1024 --opts TRAIN.WARMUP_EPOCHS 5 TRAIN.WEIGHT_DECAY 0. TRAIN.EPOCHS 1000
```
  
### Evaluation 
Download the benchmark datasets from papers [1], [2], [3]. The folder structure of the datasets should be:
```
puzzle-dataset/
├─ BGU/
├─ Cho/
├─ McGill/
```
For erosion of 7%
```bash
python3 -m torch.distributed.run --standalone --nproc_per_node 1 evaluation.py --cfg configs/puzzle/puzzle_eval_4bin_patch8_64.yaml --data-path /path/to/datasets/puzzle-dataset --batch-size 128 --pretrained output/div2k_erosion7_4bin_patch8_64/default/best_model.pth --opts DATA.EROSION_RATIO 0.07
``` 
For erosion of 14%
```bash
python3 -m torch.distributed.run --standalone --nproc_per_node 1 evaluation.py --cfg configs/puzzle/puzzle_eval_4bin_patch8_64.yaml --data-path /path/to/datasets/puzzle-dataset --batch-size 128 --pretrained output/div2k_erosion7_4bin_patch8_64/default/best_model.pth --opts DATA.EROSION_RATIO 0.14
``` 

Pretrained model for this task can be downloaded here: https://drive.google.com/file/d/1K4eAqEOALsy_FF218cHrtG-i__wyJKz7/view?usp=sharing
  
## Problem 2: Image retrieval

### Dataset
The Hisfrag20 dataset:
Link to the dataset: https://zenodo.org/records/3893807
Download the `hisfrag20_train.zip` and `hisfrag20_test.zip` from the link above and extract them 

### Training
 ```bash
 python3 -u -m torch.distributed.run --nproc_per_node 2 --standalone hisfrag.py --cfg configs/hisfrag/hisfrag20_patch16_512.yaml --data-path /path/to/datasets/HisFrag20 --tag no-pretrained --batch-size 24 --eval-n-items-per-category 2 --opts TRAIN.WARMUP_EPOCHS 5 TRAIN.WEIGHT_DECAY 0.01 TRAIN.EPOCHS 300 TRAIN.BASE_LR 3e-4 DATA.NUM_WORKERS 3 PRINT_FREQ 50 DATA.TEST_BATCH_SIZE 384
 ```
 
### Evaluation
```bash
python3 -u -m torch.distributed.run --nproc_per_node 2 --standalone hisfrag.py --cfg configs/hisfrag/hisfrag20_patch16_512.yaml --data-path /path/to/datasets/HisFrag20 --mode test --tag no-pretrained --batch-size 64 --opts MODEL.PRETRAINED output/hisfrag20_patch16_512/no-pretrained/best_model.pth DATA.NUM_WORKERS 3 PRINT_FREQ 50 TRAIN.AUTO_RESUME False DATA.TEST_BATCH_SIZE 512
```

Pretrained model for this task can be downloaded here: https://drive.google.com/file/d/1wrecy0nMAAFUIM0Bvi7nI8tEAYQzEawC/view?usp=sharing

## References

[1] T. S. Cho, S. Avidan, and W. T. Freeman. A probabilistic image jigsaw puzzle solver. In Proc. CVPR, pages 183–190, 2010. 9, 10, 11, 12, 13, 14, 15

[2] A. Olmos and F. A. A. Kingdom. McGill calibrated colour image database. http://tabby.vision.mcgill.ca., 2005

[3] Pomeranz, Dolev, Michal Shemesh, and Ohad Ben-Shahar. "A fully automated greedy square jigsaw puzzle solver." CVPR 2011. IEEE, 2011.