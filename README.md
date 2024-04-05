# AdaBM
This repository includes the official implementation of the paper [**AdaBM: On-the-Fly Adaptive Bit Mapping for Image Super-Resolution**](https://arxiv.org/abs/2404.03296) (CVPR2024).


<!-- [arXiv](TBD) | [BibTeX](#bibtex) -->


<!-- <p align="center">
<img src=assets/results.gif />
</p> -->

<p align="center">
<img src=assets/cover_adabm.png />
</p>



## Requirements
A suitable [conda](https://conda.io/) environment named `adabm` can be created and activated with:
```
conda env create -f environment.yaml
conda activate adabm
```

## Preparation
### Dataset
* For training, we use LR images sampled from [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
* For testing, we use [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) and large input datasets [Test2K,4K,8K](https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH?usp=sharing).
Test8K contains the images (index 1401-1500) from [DIV8K](https://competitions.codalab.org/competitions/22217#participate). Test2K/4K contain the images (index 1201-1300/1301-1400) from DIV8K which are downsampled to 2K and 4K resolution.
After downloading the datasets, the dataset directory should be organized as follows:

```
datasets
  -DIV2K
    - DIV2K_train_LR_bicubic # for training
    - DIV2K_train_HR
    - test2k # for testing
    - test4k
    - test8k
  -benchmark # for testing
```

### Pretrained Models
Please download the pretrained models from [here](https://drive.google.com/drive/folders/1GLuvwy3WWFG2H6iEA6-7tqRj_Jnzcn86?usp=drive_link) and place them in `pretrained_model`.

## Usage

### How to train

```
sh run.sh edsr 0 6 8 # gpu_id a_bit w_bit 
sh run.sh edsr 0 4 4 # gpu_id a_bit w_bit 
```

### How to test

```
sh run.sh edsr_eval 0 6 8 # gpu_id a_bit w_bit 
sh run.sh edsr_eval 0 4 4 # gpu_id a_bit w_bit
```

> * set `--dir_data` to the directory path for datasets. 
> * set `--pre_train` to the saved model path for testing model. 
> * the trained model is saved in `experiment` directory.
> * set `--test_own` to the own image path for testing.

More running scripts can be found in `run.sh`. 

## Comments
Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

#### Coming Soon...
 - [ ] parallel patch inference

## BibTeX
If you found our implementation useful, please consider citing our paper:
```
@misc{hong2024adabm,
      title={AdaBM: On-the-Fly Adaptive Bit Mapping for Image Super-Resolution}, 
      author={Cheeun Hong and Kyoung Mu Lee},
      year={2024},
      eprint={2404.03296},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
> Email: [cheeun914@snu.ac.kr](cheeun914@snu.ac.kr)