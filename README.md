# Single-Image Localised Reflection Removal with *k*-Order Differences Term
Official PyTorch implementation of the SCIA 2025 paper **“Single-Image Localised Reflection Removal with *k*-Order Differences Term”**.  

---  

## Lamps Dataset

Is stored [here](https://ptak.felk.cvut.cz/personal/spetlrad/lamps.tar.gz).


## Highlights
* **Localised Glass-Reflection Removal (LGRR)** – first method that specifically targets small, bright highlights caused by lamps, flash and other point sources behind glass.  
* **LGRR-Net** – a lightweight U-Net variant with a MobileNetV2 encoder and dual residual-attention decoders.  
* ***k*-order differences loss** – adds high-order colour-difference constraints that improve both reflection removal **(+5 % MAE)** and inpainting **(+3.5 % MAE)**.  
* **`Lamps` dataset** – 191 real lamp photographs for training / evaluation (released here).  
* Single model generalises to glint removal, display-case artefacts, dust & snow specks.  


## Python Environment Installation
The following python environment setup (in Ubuntu or alike) is recommended:
1. `sudo apt-get update`
1. `sudo apt-get install python3-pip python3-dev`
1. `sudo pip3 install virtualenv`
1. `cd project_directory`
1. `virtualenv lgrr`
1. `source lgrr/bin/activate`
1. `pip install -r requirements.txt`


## Training
> All this needs to be run from `nn/trn` directory.

To start the training, you simply run
`python3 trn_pytorch.py --json-path conf.json --skip-tst-dataset`

You should pay attention to several options in the `conf.json` configuration file.

> `"continue_model_partly"` a path to a model file
> 
> `"coco_train2017_directory"` a path to the train2017 subset of the COCO dataset
> 
> `"lamp_dataset_directory"` a path to the LAMPS dataset
> 
> `"experiments_directory"` the directory in which the results of the experiments will be stored, note that in this directory, a `tensorboard` directory will be created and in this directory, the tensorboard logs will be stored 
> 
> `"batch_size"` quite apparently - the batch size
> 
> `"epochs"` number of epochs to train
> 
> `"optimizer"` originally an AdamW algorithm was used, also might be Adam and SGD
> 
> `"learning_rate"`
> 
> `"loss_fg_weight"` and `"loss_bg_weight"` - scalars which weight the amount of contribution of foreground (the reflected scene) and background (the scene "behind a glass")
> 
> `"criterions"` a dictionary with loss terms and their weights (you may find a list of all implemented criterions in the `"all_criterions"` list)
> 
> `"feed_mask"` (bool) is the user mask fed as an input to the network?
> 
> `"blend_on_fly_mode"` a method of blending the images - `satone` allows for saturation, `nosat` scales down brightness values to prevent saturation; to find more about the blending methods, go to the line `444` in the file `dataset.py` and find the method `blend`
> 
> `"augment"` should augmentation be used?
> 
> `"augment_reflection"` parameters of reflection augmentation

## Visual Inpection
> All this needs to be run from `nn/trn` directory.

To see results on available datasets, run 
```
python3 visual_inspection.py --json-path conf_visual_inspection.json
```

The configuration file `conf_visual_inspection.json` contains the following :

>`"continue_model_partly"` a path to a model file
> 
>`"dataset_directory"` a path to a directory which will be visually inspected
> > note that a directory structure in form of `"dataset_directory"/dirname/image.ext` is expected
> 
>`"visual_inspection_directory"` a directory in which the results will be stored (if it doe not exists, it will be created)

The rest of the configuration json should be left intact.

---

This work was supported by Huawei company.