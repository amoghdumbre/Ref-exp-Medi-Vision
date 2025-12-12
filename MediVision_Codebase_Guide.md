# Medi-Vision: Codebase & Implementation Guide

This document provides an in-depth mapping of the codebase to the architectural concepts and experimental setup described in the **Medi-Vision** research paper. It details how each file contributes to the realization of the Dual-Branch Vision Transformer, Cross-Attention Fusion, and Medical VQA pipeline.

## 1. Core Model Architecture (Paper Section 3)

The heart of the Medi-Vision system—integrating multi-scale visual features with textual understanding—is contained within the `model/` directory.

### **`model/medi_vision.py`**
*   **Role in Paper**: This is the **primary implementation file** for the Medi-Vision architecture.
*   **Key Components**:
    *   **Dual-Branch ViT (Section 3.2)**: Orchestrates the **Large Branch (L-Branch)** and **Small Branch (S-Branch)**. It initializes two separate Transformers with different patch sizes (e.g., $P_l$ and $P_s$) to capture complementary visual details.
    *   **Cross-Attention Module (Section 3.3 & 3.4)**: Implements the specific fusion logic where the Class Token (CLS) from the Large Branch acts as a query to attend to the patch tokens from the Small Branch. This directly maps to **Figure 3** of the paper.
    *   **MedSAM Preprocessing Wrapper**: A placeholder class for integrating the **Segment Anything Model (MedSAM)** (Section 4.2), facilitating the focus on specific organs or lesions.

### **`model/mplug.py`**
*   **Role in Paper**: Provides the foundational **Vision Transformer (ViT)** building blocks (Section 3.1).
*   **Key Components**:
    *   `MplugVisualTransformer`: Handles patch embeddings, positional embeddings, and the stack of transformer encoder layers. The `MediVisionModel` instantiates this class twice to create the dual-branch structure.

### **`model/xbert.py`**
*   **Role in Paper**: Serves as the **Language Encoder** (Section 4.3).
*   **Key Components**:
    *   Implements the BERT-based text encoder used to tokenize and process the medical questions.
    *   While the paper discusses advanced language modeling (conceptually LLAMA), this file provides the robust, trainable text encoding backbone typically used in VQA to generate the textual feature embeddings ($x_{txt}$) described in the mathematical notations.

## 2. Data Preparation & Loading (Paper Section 4.1)

Handling the specific medical datasets (VQA-RAD, SLAKE) requires specialized loaders found in the root directory.

### **`vqa_med_dataset.py`**
*   **Role in Paper**: Implements the data ingestion pipeline for the datasets compared in **Table 1**.
*   **Key Classes**:
    *   `slake_cls_dataset`: Handles the **SLAKE** dataset (English subset), coordinating images with their questions, answers, and metadata (modality, body part).
    *   `vqa_rad_cls_dataset`: Handles the **VQA-RAD** dataset, parsing the concise question-answer pairs relevant to radiology.
    *   **Preprocessing**: Includes text normalization functions (`pre_question`, `pre_answer`) ensuring consistent inputs for the model.

## 3. Training & Experimental Setup (Paper Section 4.5)

The files responsible for executing the experiments, managing hyperparameters, and producing the results shown in **Section 5**.

### **`main.py`**
*   **Role in Paper**: The **Execution Engine**.
*   **Key Functions**:
    *   `train_one_epoch`: Implements the training loop, including forward passes, loss calculation (Cross-Entropy), and backpropagation.
    *   `eval_func`: Runs the validation/testing phases to compute accuracy metrics.
    *   **Optimization**: Manages learning rates ($\tau$), gradient accumulation, and utilizing the GPU for parallel processing as mentioned in the paper's introduction.

### **`train_parser.py`**
*   **Role in Paper**: Hyperparameter Configuration.
*   **Key Components**:
    *   Defines command-line arguments for batch sizes, learning rates, epoch counts, and model checkpoints. This allows for the precise reproduction of the "Experiment Setup" details.

### **`train_utils.py`**
*   **Role in Paper**: Support Utilities.
*   **Key Components**:
    *   `generate_model`: A factory function that initializes the specific model architecture based on the configuration file.
    *   `create_dataset`: Selects the correct dataset class (SLAKE vs VQA-RAD) based on input arguments.
    *   `configure_optimizers`: Sets up the AdamW optimizer with cosine learning rate schedules, critical for the model's convergence.

## 4. Helper Utilities

### **`vit.py`**
*   **Role in Paper**: Positional Embedding Adjustment.
*   **Key Function**: `resize_pos_embed`. When initializing the ViT branches from pre-trained weights, the resolution might differ from the paper's specific input size. This utility interpolates positional embeddings to match the required grid size, enabling the use of powerful pre-trained checkpoints.

## Summary of File Mapping

| Paper Component | File Implementation |
| :--- | :--- |
| **Multi-Scale ViT / Fusion** | `model/medi_vision.py` (Logic), `model/mplug.py` (Blocks) |
| **Text Encoding** | `model/xbert.py` |
| **Data (SLAKE, VQA-RAD)** | `vqa_med_dataset.py` |
| **MedSAM Segmentation** | `model/medi_vision.py` (Wrapper), External Repo (Logic) |
| **Training Loop** | `main.py` |
