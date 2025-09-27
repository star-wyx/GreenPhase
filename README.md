# GreenPhase

This repository contains the implementation of **GreenPhase**, a green learning approach for seismic detection and earthquake phase picking. The method aims to improve computational efficiency while maintaining competitive performance in phase identification.



## Data Preparation

Preprocess the data:

```bash
python preprocess.py
```

* The [STEAD dataset](https://github.com/smousavi05/STEAD) is required for model training and testing.
* The list of testing traces used in the paper can be downloaded from the [EQTransformer repository](https://github.com/smousavi05/EQTransformer).



## Model Training

Train P-wave and S-wave models:

```
python train.py --wave p --exp_name demo
python train.py --wave s --exp_name demo
```

Train the classifier (with parallel workers):

```
python train_clf.py --exp_name demo --n_jobs 16
```



## Model Evaluation

Pretrained models are available in the `exp` directory.

To evaluate them, run:

```bash
python test.py --exp_name pretrained --n_jobs 16
python evaluate.py --exp_name pretrained
```



