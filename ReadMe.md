## oculus

### Introduction

Oculus is a timeseries forecast framework first designed for monthly T2m and Precitation anomaly forecast in mainland China meteorological stations.


### Installation

Install pytorch after installing anaconda3 if you hope to use ANN.

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Structure

* `./feature_lib/`: Potential features (X matrix) for build model
* `./label/`: Predicted labels (Y matrix) for build model
* `./db/`: Archived model and evaluation dictionary
* `./inferX/`: X matrix for inference 
* `./output/`: Archived predictions 

### Usage


#### Single Instance

Build the model after properly setting `./conf/config.ini`

```
python build_model.py
```

When you have inference X, just write a X file follow the format of trainning data. Run:
```
python infer_model.py
```
Check `./output/` for results.

#### Multiple Instances

`batch_build.py` will get a list of all files in `./feature_lib` and `./label`, and loop them together.
In each loop, the files from `./feature_lib` and `./label` will be treated as an X-Y pair. 
The config file will be modified accordingly and then call `build_model.py`.

**Plese first modify `./conf/config.sample.ini`**, then execute:

```
python batch_build.py
```

Similarly, use `batch_infer.py` for batch inference.

```
python batch_infer.py
```

