## 1D Regression

---
### Training
```
python gp.py --mode train --expid lbanp-num_latents-8 --model lbanp --num_latents 8
```
The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

### Evaluation
```
python gp.py --mode eval --expid lbanp-num_latents-8 --model lbanp --num_latents 8
```
Note that `{expid}` must match between training and evaluation since the model will load weights from `results/gp/{model}/{expid}` to evaluate.

## CelebA Image Completion
---

### Data Preparation
Download the files from the official CelebA [google drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8). Specifically, download [list_eval_partitions.txt](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk), [identity_CelebA.txt](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS), and  [img_align_celeba.zip](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM) and unzip, placing the downloaded files in `datasets/celeba` folder. 

Alternatively, follow the instructions from the official [website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to download the aforementioned necessary files. 

After downloading the data, preprocess it by running `python data/celeba.py --resolution 32`, `python data/celeba.py --resolution 64`, `python data/celeba.py --resolution 128`.

### Training

**CelebA (32 x 32):**
```
python celeba.py --mode train --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --resolution 32 --max_num_points 200
```

**CelebA (64 x 64):**
```
python celeba.py --mode train --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --resolution 64 --max_num_points 800
```

**CelebA (128 x 128):**
```
python celeba.py --mode train --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --resolution 128 --max_num_points 1600
```

### Evaluation

When evaluating for the first time, the evaluation data will be generated and saved in `evalsets/celeba`.

**CelebA (32 x 32):**
```
python celeba.py --mode eval --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --resolution 32 --max_num_points 200
```

**CelebA (64 x 64):**
```
python celeba.py --mode eval --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --resolution 64 --max_num_points 800
```

**CelebA (128 x 128):**
```
python celeba.py --mode eval --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --resolution 128 --max_num_points 1600
```

## EMNIST Image Completion
---

### Training

If training for the first time, EMNIST training data will automatically downloaded and saved in `datasets/emnist`.

```
python emnist.py --mode train --expid lbanp-num_latents-8 --model lbanp --num_latents 8
```

### Evaluation

When evaluating for the first time, the evaluation data will be generated and saved in `evalsets/emnist`.
**EMNIST (0-9):**
```
python emnist.py --mode eval --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --class_range 0 10
```

**EMNIST (10-46):**

```
python emnist.py --mode eval --expid lbanp-num_latents-8 --model lbanp --num_latents 8 --class_range 10 47
```
