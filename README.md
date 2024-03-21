# AdaBM
[arXiv](TBD) | [BibTeX](#bibtex)


<p align="center">
<img src=assets/results.gif />
</p>

<p align="center">
<img src=assets/modelfigure.png />
</p>



## Environment
A suitable [conda](https://conda.io/) environment named `adabm` can be created and activated with:
```
conda env create -f environment.yaml
conda activate adabm
```

## Dataset Preparation
Please download DIV2K datasets from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) for training and [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) for testing.
Then, organize the dataset directory as follows:

```
datasets
  -benchmark
  -DIV2K
```


## Usage

* How to train

```
sh run.sh edsr 0 4 4 # gpu_id a_bit w_bit 
```

* How to test

```
sh run.sh edsr_eval 0 4 4 # gpu_id a_bit w_bit 
```

> set `--pre_train` to the saved model path for testing model.

More running scripts can be found in `run.sh`. 



## Comments
Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

#### Coming Soon
 - [ ] bac for multi gpu (2)
 - [ ] check functionality of crop, combine (refer to ape)
 - [ ] run other models with lr_a 0.5 -> 0.01
 - [ ] check with paper (loss, ...)
 - [ ] paper update TBD: lr_a, results using lr_a 0.01
 - [ ] test checkpoints release
 - [ ] environment file

## BibTeX