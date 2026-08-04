# Reveal.AI: A Scalable and Interpretable Framework for Detecting Synthetic Videos

## Project Description
Reveal.AI is a multimodal deep learning framework designed to identify videos produced by generative models, regardless of their subject matter. Unlike conventional detectors that search for localized facial artifacts, Reveal.AI utilizes a multimodal architecture to capture global spatial, temporal, and acoustic inconsistencies. 

The system processes user-uploaded content through two complementary paths:
* **Visual Path:** A Video Swin Transformer (3D variant) analyzes sampled video frames to capture subtle spatial and temporal inconsistencies.
* **Audio Path:** Mel-Frequency Cepstral Coefficients (MFCC) are extracted from the audio track and classified with a Convolutional Neural Network (CNN) to detect synthetic acoustic signatures.

The outputs from both modalities are combined using a weighted fusion strategy to produce a final classification and a confidence score. Furthermore, to bridge the gap between detection and human understanding, the system integrates Explainable AI (XAI) using Grad-CAM visualizations to highlight the specific frame regions and audio spectrograms driving the model's decision.

## Contributors
* **[Alfiya Ashraf](https://github.com/Alfieeya)**
* **[Madhavdas M](https://github.com/madhavdasm)**
* **[Joann Jibin](https://github.com/JoannJibin)**
* **[Niya Rajan](https://github.com/Niya-Rajan)**
