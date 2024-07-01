# Color Modulation Diffusion and Cross-spectral Detail Refinement for Efficient Underwater Image Restoration

## Introduction
Raw underwater images often suffer from significant visual degradation, which limits their suitability for subsequent applications. While recent underwater image enhancement (UIE) methods rely on the current advances in deep neural network architecture designs, there is still considerable room for improvement in terms of cross-scenario robustness and computational efficiency. Recently, diffusion models have shown great success in image generation, prompting us to consider their application to UIE tasks. However, directly applying the diffusion models to UIE tasks will pose two challenges, i.e., high computational budget and underwater degradation perturbations. To tackle these challenges, we present a novel Underwater Frequency-guided Diffusion Model (UFDM) for efficient UIE. Unlike vanilla diffusion models that diffuse in the raw pixel space of the image, we convert the image into the wavelet domain to independently correct color and refine detail. By utilizing discrete wavelet transform to obtain such low-frequency and high-frequency spectra, it inherently reduces the image spatial dimensions by half after each transformation, thereby achieving potential inference acceleration on the low-frequency component. For the sacrificed image details caused by underwater medium scattering, we propose the cross-spectral detail refinement (CSDR) module to enhance the high-frequency details, which are then integrated with the low-frequency signal as input conditions for guiding the diffusion. This way not only ensures the high-fidelity of sampled content but also compensates for the sacrificed details. Additionally, we propose the global color correction (GCC) module to modulate the various color shifts, which arises from the imbalanced information distribution caused by underwater selective absorption. Comprehensive experiments demonstrate the superior performance of UFDM over state-of-the-art methods in both quantitative and qualitative evaluations.

## Requirement

## Train

## Evaluation

## References
