<div align="center">
  <img src="doc/crtnet.png" alt="BigPictureNet">

  <h4>The Context-aware Recognition Transformer Network</h4>

  <a href="#about">About</a> •
  <a href="#crtnet-model">CRTNet Model</a> •
  <a href="#code-architecture">Code Architecture</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#mturk-experiments">Mturk Experiments</a> •
  <a href="#citation">Citation</a>
</div>




---

## About

Conventional object recognition models are designed for images that are focused on a single object. While it is of course always possible to crop a large image to an object of interest, a lot of potentially valuable contextual information is sacrificed in that process. As our experiments show, humans are able to make use of additional context to reason about the object of interest and achieve considerably higher recognition performance.

Our Context-aware Recognition Transfomer (CRTNet) is designed to shrink this gap between human and computer vision capabilities by looking at the big picture and leveraging the contextual information.



## CRTNet Model
<div align="center">
  <img src="doc/model_architecture.png" alt="model_architecture">
</div>



CRTNet is presented with an image containing multiple objects
and a bounding box to indicate the target object location.
Inspired by the eccentricity dependence of human vision,
CRTNet has one stream that processes only the target
object (I<sub>t</sub> , 224 × 224), and a second stream devoted to the
periphery (I<sub>c</sub> , 224 × 224). I<sub>t</sub> is obtained by cropping the
input image to the bounding box whereas I<sub>c</sub> covers the
entire contextual area of the image. I<sub>c</sub> and I<sub>t</sub> are then
resized to the same dimensions. Thus, the target object’s
resolution is higher in I<sub>t</sub> . The two streams are encoded
through two separate 2D-CNNs. After the encoding stage,
CRTNet tokenizes the feature maps of I<sub>t</sub> and I<sub>c</sub> , integrates
object and context information via hierarchical reasoning
through a stack of transformer decoder layers, and predicts
class label probabilities y<sub>t,c</sub> within C classes.

A model that always relies on context can make mistakes
under unusual context. To increase robustness, CRTNet
makes a second prediction y<sub>t</sub> , based on target object
information alone, estimates the confidence p of this
prediction, and computes a confidence-weighted average of
y<sub>t</sub> and y<sub>t,c</sub> to get the final prediction y<sub>p</sub> . If the model makes
a confident prediction with the object only, it can overrule the
context reasoning stage.

## Code Architecture

- All relevant components are implemented in `core/`.
- We use COCO-style annotations for train and test sets. An example can be found in the `debug/` folder.
- Training and testing can be performed with `train.py` and `test.py` respectively. Annotations, image directory and relevant parameters should be set via command line arguments. Available command line arguments can be displayed by running `python train.py --help` and `python test.py --help` respectively.



#### Examples

Train model with default settings using the specified annotations and images. Outputs including a config and model checkpoints are saved to the directory specified via `--outdir`.
```
python train.py --annotations debug/annotations.json --imagedir debug/images --outdir output
```

Test a trained model on a dataset with the given annotations and images.
```
python test.py --checkpoint output/checkpoint_1.tar --config output/config.yaml
--annotations testset/annotations.json --imagedir testset/images --weighted_prediction
```

#### Our pre-trained models 
One can download our pre-trained models:
 - model on OCD dataset [HERE](https://drive.google.com/drive/folders/1jO8M51F2zhBcPLwoguuzeMspFWqRvj5s?usp=sharing)
 - model on UnRel dataset [HERE](https://drive.google.com/drive/folders/1nOVKcHzCi9xtYY4BvVHjo9fJTBxsXwrg?usp=sharing)
 - model on Cut-paste dataset [HERE](https://drive.google.com/drive/folders/18ggHf49jdCh-7qOCLe7mPnaSwhbNV69w?usp=sharing)

## Datasets

Download all the folders in ```human``` from [HERE](https://drive.google.com/drive/folders/1lm5Zt96-n6uzGbXyW6n0q2QzFghdGyjf?usp=sharing) and place them in ```human``` in the current repository.

###  Existing datasets
Existing datasets can be downloaded from [UnRel](https://www.di.ens.fr/willow/research/unrel/), [Cocostuff](https://github.com/nightrome/cocostuff) and [Cut-and-Paste](https://github.com/kreimanlab/Put-In-Context).

### our Out-of-Context Dataset (OCD)

Our OCD dataset is developed based on VirtualHome simulation environment. Download the python github repository [HERE](https://github.com/xavierpuigf/virtualhome) and the original unity repository [HERE](https://github.com/xavierpuigf/virtualhome_unity).

(Skip this step) If one wants to build Unity Virtualhome environment from scratch,  replace the old ```virtualhome_unity/Assets/Story Generator/Scripts/TestDriver.cs``` in the original unity repository with ```unity/TestDriver.cs``` in the current repository. Re-export Unity executable files. 

If one wants to directly run the Virtualhome environment with our pre-defined out-of-context conditions to generate our OCD dataset, it is NOT necessary to download and install Unity. Directly download the pre-compiled Unity executable file from [HERE](https://drive.google.com/file/d/1zMondbrWKHglwbxG7WinLnP0FK2JsHh2/view?usp=sharing). It runs on Linux 64-bit platform (such as Ubuntu18.04). Make sure to double click this executable ```linux/MMVHU.x86_64```, and it is running. 

MAC OSX version can be downloaded [HERE](https://drive.google.com/file/d/1Bp-R9OTIJRMQ9miMl6hvmG7ooGKEBoIR/view?usp=sharing) and Windows version can be downloaded [HERE](https://drive.google.com/file/d/1MV2oLrUfDSTYKcpS_rUw-PMdN8Olx-Qs/view?usp=sharing). 

Copy all the files in ```unity``` folder in the current repository to ```virtualhome/demo/``` folder in the downloaded python github repository [HERE](https://github.com/xavierpuigf/virtualhome).

And then, go to ```cd virtualhome/demo/``` folder, launch any of the following Python scripts in the command window:
```
#generate environment graphs (compulsory before running any of the following conditions)
python exp_HumanGraph.py
python exp_HumanGraph_anomaly.py
#different contextual conditions
#generate images for gravity
python exp_graivty.py
python exp_gravity_ori.py
#Size
python exp_size.py
python exp_size_2.py
python exp_size_ori.py
python exp_size_ori_2.py
#Normal Conditions
python exp_GT.py
python exp_GT_ori.py
#Co-occurence (C) and Gravity + C
python exp_anomaly.py
python exp_anomaly_wall.py
#NoContext
python exp_GT_seg.py
python exp_GT_ori_seg.py
#Training images from VH and test on COCOstuff
python exp_train_5.py
python exp_train_6.py
```
It would generate image stimulus and save the corresponding 3D object configurations in the path and directory specified in each python script, e.g.:
```
stimulusdirname = 'stimulus_gravity'
jasondirname = 'jason_gravity'
```
You can skip all the steps above, if you want to directly use the images from our dataset without any modifications.
Download links for the dataset:
 - Normal: raw images[HERE](https://www.dropbox.com/sh/2he6a884v56tml3/AACS-74urq7lAuocE_5yHDTZa?dl=0)
 - Gravity raw images[HERE](https://www.dropbox.com/sh/3spivxx6c5hhn98/AADcUMhQBNGEmCOiZhLgp4JQa?dl=0)
 - Size raw images[HERE](https://www.dropbox.com/sh/5yv31zw7631peod/AABxyiltxs8XYwvBGEtGIbBHa?dl=0)
 - Co-occurrence raw images[HERE](https://www.dropbox.com/sh/mza2njlqtc1ttqv/AABT-Ju9tSkCI0zLIPQOwffca?dl=0)
 - G+C raw images (naming convention with _wall) [HERE](https://www.dropbox.com/sh/mza2njlqtc1ttqv/AABT-Ju9tSkCI0zLIPQOwffca?dl=0)
 - NoContext raw images (naming convention with _seg) [HERE](https://www.dropbox.com/sh/2he6a884v56tml3/AACS-74urq7lAuocE_5yHDTZa?dl=0)
 - Training images from VirtualHome raw images[HERE](https://www.dropbox.com/sh/dite90lv0s0kkr1/AAB9xz2oX1s7AqddxkUvdn-Ua?dl=0) and [HERE](https://www.dropbox.com/sh/0w8zu80erw61i5i/AABVkfe4FwZqRDeoGQqCCplYa?dl=0)
 - Jason files [HERE]() For each image in the conditions above, there exists a corresponding jason file storing the target object classname, class-id, apartment-id, room-id, surface-id, the bounding box (left, right, bottom, top coordinate wrt (1024, 1280) image size). 

**NOTE** NOT all images are used for testing. Within each condition, we manually filtered and selected the good quality images for human and model testing. The ```human/Mat/VHhumanStats_*.mat``` stores the SELECTED test images. The raw filtered image lists for each dataset is in ```human/filtered/```. For example, ```filtered_gravity```, ```filtered_gravity_ori```  and ```human/Mat/VHhumanStats_gravity.mat``` are the selected image information for gravity condition. 

(not recommended) If one wants to filter images again, use scripts ```unity/filterImages_gravity.ipynb``` and ```human/ProcessFilteredTextFiles_gravity.m``` to re-generate ```human/Mat/VHhumanStats_gravity.mat```.

## Mturk Experiments
We conduct Amazon Mechanical Turk experiments using the selected images from above in OCD datasets. For all mturk experiments, one can download [HERE](https://drive.google.com/drive/folders/1C6T4waXHlAyAxYqrvjfbssFSO4HtWJ2y?usp=sharing). We are now using ```human/expGravity``` as an example. 

We designed a series of Mturk experiments using [Psiturk](https://psiturk.org/) which requires javascripts, HTML and python 2.7. The source codes have been successfully tested on MAC OSX and Ubuntu 18.04. See sections below for installation, running the experiments locally and launching the experiments online.

### Installation of Psiturk

Refer to [link](https://www.anaconda.com/distribution/) for Anaconda installation. Alternatively, execute the following command:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```
After Anaconda installation, create a conda environment:
```
conda create -n mturkenv python=2.7
```
Activate the conda environment:
```
conda activate mturkenv
```
Install psiturk using pip:
```
pip install psiturk
```
Refer to [HERE](https://drive.google.com/open?id=1FblDG7OuWXVRfWo0Djb5eDiYgKqnk9wU) for detailed instruction on setting up psiturk key and paste them in .psiturkconfig.

### Running the experiment locally

Navigate to any experiments in ```human/expGravity``` folder. In the following, we take gravity as an example, one can replace it with any other experiments. Open a command window, navigate to ```human/expGravity```, and run the experiment in debug mode:
```
cd human/expGravity
psiturk
server on
debug
```
**NOTE** You can run the source codes directly. All the stimulus set (all GIF files) have been hosted in our lab server: http://kreimanlab.com/mengmiMturkHost/VirtualHome/keyframe_VH_gravity_gif/. One can freely view any stimulus (.gif) via Internet, e.g. http://kreimanlab.com/mengmiMturkHost/VirtualHome/keyframe_VH_gravity_gif/gif_712_7_5_3.gif. In case that the links are unavailable, one can generate the whole stimulus set for each experiment by running ```human/PreprocessVH_gravity.m``` to generate GIF, running ```human/GenerateMturkSets_expGravity.m``` to generate random shuffled sequence of GIF presentation. The pre-generated random shuffled sequence has been stored in ```human/expGravity/static/ImageSet/``` folder.

We now list a detailed description of important source files:
- human/db/expGravity.db: a SQL database storing online subjects' response data. 
- human/expGravity/template/instructions/instruct-1.html: show instructions to the human subjects
- human/expGravity/static/js/task.js: main file to load stimulus and run the experiment

It is optional to re-process these .db files. Since all the pre-processed results have been stored in ```human/Mat/```. If one wants to re-convert these .db files to .mat files. For each experiment, one can run ```human/ProcessDBfile_expGravity.m``` and ```mturk/CompileAllExpGravity.m```.

To plot results in the paper, run the following scripts:
 - PlotAblationOverall_humanoverlap.m #ablation plots
 - PlotBar_unrel.m #bar plots #for unrel experiment
 - PlotCorrelation_table_cvpr.m #for cut-and-paste dataset
 - PlotModelOverall_humanoverlap.m #models on OCD dataset
 - PlotHumanOverall_modeloverlap.m #human on OCD dataset

### Launching the experiment online using Elastic Cloud Computing (EC2) in Amazon Web Services (AWS)

Copy the downloaded source codes to EC2 server and run the psiturk experiment online. Refer to [HERE](https://drive.google.com/open?id=1FblDG7OuWXVRfWo0Djb5eDiYgKqnk9wU) for detailed instruction.


## Citation

> TODO

The images for the sample dataset in the `debug` folder were generated with VirtualHome (see http://virtual-home.org/).

## Notes

The source code is for illustration purpose only. Path reconfigurations may be needed to run some MATLAB scripts. We do not provide techinical supports but we would be happy to discuss about SCIENCE!

## License

See [Kreiman lab](http://klab.tch.harvard.edu/code/license_agreement.pdf) for license agreements before downloading and using our source codes and datasets.

