# Color Correction Meets Cross-Spectral Refinement: A Distribution-Aware Diffusion for Underwater Image Restoration

## Introduction
Underwater imaging often suffers from significant visual degradation, which limits its suitability for subsequent applications. While recent underwater image enhancement (UIE) methods rely on the current advances in deep neural network architecture designs, there is still considerable room for improvement in terms of cross-scene robustness and computational efficiency. Diffusion models have shown great success in image generation, prompting us to consider their application to UIE tasks. However, directly applying them to UIE tasks will pose two challenges, \textit{i.e.}, high computational budget and color unbalanced perturbations. To tackle these issues, we propose DiffColor, a distribution-aware diffusion and cross-spectral refinement model for efficient UIE. Instead of diffusing in the raw pixel space, we transfer the image into the wavelet domain to obtain such low-frequency and high-frequency spectra, it inherently reduces the image spatial dimensions by half after each transformation. Unlike single-noise image restoration tasks, underwater imaging exhibits unbalanced channel distributions due to the selective absorption of light by water. To address this, we design the Global Color Correction (GCC) module to handle the diverse color shifts, thereby avoiding potential global degradation disturbances during the denoising process. For the sacrificed image details caused by underwater scattering, we further present the Cross-Spectral Detail Refinement (CSDR) to enhance the high-frequency details, which are integrated with the low-frequency signal as input conditions for guiding the diffusion. This way not only ensures the high-fidelity of sampled content but also compensates for the sacrificed details. Comprehensive experiments demonstrate the superior performance of DiffColor over state-of-the-art methods in both quantitative and qualitative evaluations.

![image](https://github.com/user-attachments/assets/8483be7a-e193-4785-a252-657a7aa117a3)


## Keywords
Underwater image restoration; Efficient diffusion model; Global color correction; High-frequency detail refinement.
## Requirement
* Python 3.8
* Pytorch 2.0.1
* CUDA 11.7
```bash
pip install -r requirements.txt
```

## Training
1. Prepare the underwater dataset and set it to the following structure.
```
|-- WaterDatasets
    |-- train
        |-- input  #raw underwater images
        |-- target  #reference images
        |-- WaterDatasets_train.txt  #image information
    |-- val
        |-- input
        |-- target
        |-- WaterDatasets_val.txt
```
For the two `WaterDatasets_train.txt` and `WaterDatasets_val.txt`, you can run the `Img2Text.py` to generate the `.txt` files.

2. Revise the following hyper parameters in the `configs/UIE.yml` according to your situation.

3. Begin the training.
```python
python train.py
```
Note: we use the DDIM sampling to speed up the inference stage. The number of steps is set as 10.
```python
--sampling_timesteps = 10 #You can revise it if necessary.
```

## Testing
1. Modify the paths to dataset and pre-trained model. You need to revise the following path in the `configs/UIE.yml` .
```python
test_dataset # testing dataset path
ckpt_dir # pre-trained maode path
```
2. Begin the testing.
```python
python evaluate.py
```
## Notes
After the paper is accepted, we will upload the complete code here. Thanks for your attention!
