# Color Modulation Diffusion and Cross-spectral Detail Refinement for Efficient Underwater Image Restoration

## Introduction
Raw underwater images often suffer from significant visual degradation, which limits their suitability for subsequent applications. While recent underwater image enhancement (UIE) methods rely on the current advances in deep neural network architecture designs, there is still considerable room for improvement in terms of cross-scenario robustness and computational efficiency. Recently, diffusion models have shown great success in image generation, prompting us to consider their application to UIE tasks. However, directly applying the diffusion models to UIE tasks will pose two challenges, i.e., high computational budget and underwater degradation perturbations. To tackle these challenges, we present a novel Underwater Frequency-guided Diffusion Model (UFDM) for efficient UIE. Unlike vanilla diffusion models that diffuse in the raw pixel space of the image, we convert the image into the wavelet domain to independently correct color and refine detail. By utilizing discrete wavelet transform to obtain such low-frequency and high-frequency spectra, it inherently reduces the image spatial dimensions by half after each transformation, thereby achieving potential inference acceleration on the low-frequency component. For the sacrificed image details caused by underwater medium scattering, we propose the cross-spectral detail refinement (CSDR) module to enhance the high-frequency details, which are then integrated with the low-frequency signal as input conditions for guiding the diffusion. This way not only ensures the high-fidelity of sampled content but also compensates for the sacrificed details. Additionally, we propose the global color correction (GCC) module to modulate the various color shifts, which arises from the imbalanced information distribution caused by underwater selective absorption. Comprehensive experiments demonstrate the superior performance of UFDM over state-of-the-art methods in both quantitative and qualitative evaluations.

![image](https://github.com/LaibinChang/UFDM/assets/88143736/c51aaf17-c600-4d39-9bc0-ed3deea7f366)

## Keywords
Underwater image restoration; Efficient diffusion model; Global color correction; High-frequency detail refinement.
## Requirement
* Python 3.8
* Pytorch 2.0.1
* CUDA 11.7
```bash
pip install -r requirements.txt
```
## Testing
You can directly test the performance of the pre-trained model as follows:
1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `configs/UFDM_config.json` 
```python
test_dataset # testing dataset path
ckpt_dir # pre-trained maode path
```
2. Test the model
```python
python evaluate.py
```

## Training
1. Prepare the underwater dataset and set it to the following structure:
```
|-- WaterDatasets
    |-- train
        |-- input # raw underwater images
        |-- target # reference images
        |-- WaterDatasets_train.txt # image information
    |-- val
        |-- input # raw underwater images
        |-- target # reference images
        |-- WaterDatasets_val.txt # image information
```
2. Run the following to generate the `.txt` files based on the datasets.
```python
# generate WaterDatasets_train.txt and WaterDatasets_val.txt
import os
img_folder = "WaterDatasets/train/input/"
paths = []
for root, _, names in os.walk(img_folder):
    for name in names:
        path = os.path.join(root, name)
        path_gt = path.replace('input', 'target')
        if os.path.exists(path_gt):
            paths.append(path)
        else:
            paths.append(path)
            print(f"No corresponding file for {path}")
save_path = "WaterDatasets_train.txt"
with open(save_path, 'w') as f:
    for p1 in paths:
        f.write(f"{p1}\n")
```
3. Modify the following path in the `configs/UIE.yml` according to your needs:
```python
data: 
    train_dataset: "WaterDatasets" # dataset name
    val_dataset: "WaterDatasets"
    test_dataset: "WaterDatasets"
    patch_size: 256
    num_workers: 4
    data_dir: "datasets/" # dataset path
    ckpt_dir: "ckpt/" # weight saving path
    conditional: True 
training:
    batch_size: 16
    n_epochs: 500
    validation_freq: 1000
optim:
    optimizer: "Adam"
    lr: 0.0001
    amsgrad: False
    eps: 0.00000001
    step_size: 50
    gamma: 0.8
```
4. train the model
```python
python train.py
```
Note: we use the DDIM sampling to speed up the inference stage. The number of steps is set as 10.
```python
--sampling_timesteps = 10 #You can revise it if necessary.
```

## Notes
After the paper is accepted, we will upload the complete code here. Thanks for your attention!
