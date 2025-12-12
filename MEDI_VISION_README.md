# Medi-Vision: Integrating Vision and Language Models for Medical QA

This directory contains the implementation of the **Medi-Vision** model, as described in the paper *Medi-Vision: Integrating Vision and Language Models for Medical QA*.

## Overview

The Medi-Vision model is designed to enhance Medical Visual Question Answering (VQA) by integrating:
1.  **Multi-Scale Vision Transformer (ViT)**: Utilizes two branches (Large and Small) to capture both fine-grained and global visual features.
2.  **Cross-Attention Mechanism**: Fuses features from the dual ViT branches, allowing the Large branch's Class (CLS) token to attend to the Small branch's patch tokens.
3.*   **MedSAM Preprocessing**: Leverages the **Segment Anything Model (MedSAM)** to segment medical images before processing.
    *   **IMPORTANT**: The code provided includes a placeholder. For the actual implementation and inference logic, you **MUST** follow the official instructions at:
        [https://github.com/bowang-lab/MedSAM.git](https://github.com/bowang-lab/MedSAM.git)
4.  **Language Understanding**: Integrated with a robust text encoder (BERT-based in this implementation, conceptually LLAMA 2) for question understanding.

## Implementation Details

The core implementation is located in `model/medi_vision.py`.

### Classes

*   **`MediVisionModel`**: The main model class.
    *   Initializes two `MplugVisualTransformer` instances for the Large and Small branches.
    *   Implements the `CrossAttention` module to fuse the branches.
    *   Uses a `text_encoder` (defaulting to BERT via `xbert`) to process the questions.
    *   Combines the visual and textual features for the final classification.

*   **`CrossAttention`**: A module that implements the cross-attention fusion strategy described in Figure 3 of the paper. It allows the query (from the Large branch) to attend to keys/values (from the Small branch).

*   **`MedSAMPreprocessor`**: A utility class to wrap the MedSAM segmentation logic.
    *   **Note**: This requires the `segment-anything` library and the MedSAM checkpoint.
    *   Refer to the official implementation for setup: [MedSAM GitHub](https://github.com/bowang-lab/MedSAM.git)
    *   For local reference (if available): `/Users/amoghdumbre/Downloads/medsam.py`

## Usage

### 1. Model Configuration

To use `MediVisionModel`, ensure your configuration dictionary includes the necessary parameters for both branches:

```python
model_config = {
    'large_branch': {
        'input_resolution': 224,
        'patch_size': 16,
        'width': 768,
        'layers': 12,
        'heads': 12,
        'output_dim': 768
    },
    'small_branch': {
        'input_resolution': 224, 
        'patch_size': 8, 
        'width': 384, 
        'layers': 6, 
        'heads': 6, 
        'output_dim': 768
    },
    'text_tokenizer': 'bert-base-uncased',
    # ... other config options
}
```

### 2. Instantiation

```python
from model.medi_vision import MediVisionModel
# train_dataset object needed for num_answers
model = MediVisionModel(model_config, train_dataset)
```

### 3. MedSAM Preprocessing

The paper emphasizes using MedSAM for segmentation. Ensure images are processed before passing them to the model or integrate the `MedSAMPreprocessor` into your data loader `__getitem__` method.

```python
from model.medi_vision import MedSAMPreprocessor

# Initialize with path to MedSAM checkpoint (refer to downloaded file or repo)
preprocessor = MedSAMPreprocessor(medsam_ckpt_path='/path/to/medsam_vit_b.pth')

# Segment image
mask = preprocessor.segment(image, box_prompt)
# Combine mask with original image or use as attention mask
```

## References

*   **Paper**: `My_paper_Medi-Vision.txt` (Local)
*   **MedSAM**: [GitHub Repository](https://github.com/bowang-lab/MedSAM.git)
