## Statistical features for detection of voice spoofing
Code for the paper "Bag of features for voice anti-spoofing"

We introduce a ”Bag of features”: a large number of different features for synthesized voice detection. "Bag of features" consist of a bunch of statistical parameters calculated on the raw audio signal and various spectrograms generated from it.
We developed anti-spoofing system based on the introduced set of features that demonstrates outstanding results on ASVspoof 2019 challenge LA section as a single system giving a 3.93% equal error rate (EER) on the evaluation set.

### Setup

```
git clone https://github.com/IDRnD/antispoofing-features.git
cd antispoofing-features
pip install -r requirements.txt
```

 - Setup path for downloading and extracting of the ASVspoof 2019 dataset (**dataset_path** variable in the config.py)
 - Run the next script for downloading and extraction of the dataset:
 ```
 python download_dataset.py
 ```
  - Setup number of processes used for parallel computations (8 by default)

### Extraction of features

For extraction of statistical features run the next script:
```
python extract_features.py
```
The script outputs extracted features to the **data** directory:
```
data
|__dev
   |__repeats.npy
   |__stats.npy
    ...
    
|__train
   |__repeats.npy
   |__stats.npy
    ...
   
|__val
   |__repeats.npy
   |__stats.npy
    ...
```

### Training the model

For training of decision tree-based models on the top of generated features run the next script:
```
python train_pipeline.py
```
Trained models will be saved to the **models** directory if **save_models** parameter of config is set to True.

**Note**

If you have a GPU with CUDA support you can use it for acceleration of training process. 
**use_gpu** parameter of config should be set to True (False by default, also check **gpu_device_id** parameter).

### Evaluation

For evaluation of the EER score on the validation set of ASVspoof 19 LA dataset use **model_testing.ipynb** notebook.

### Citation

If you find this code useful please cite us in your work:
```
@article{Torgashov2020BagOfFeatures,
  title={Bag of features for voice anti-spoofing},
  author={Nikita Torgashov, Ivan Iakovlev and Konstantin Simonchik},
  booktitle = {submitted to Interspeech},
  year={2020}
}
```