### hoof

#### Installation
```
conda create -n hoof_env python=3.6 pip -y
source activate hoof_env

pip install -r requirements.txt
conda install pytorch cudatoolkit=9.0 -c pytorch

pip install -e .
```

#### Usage
```
cd experiments
```