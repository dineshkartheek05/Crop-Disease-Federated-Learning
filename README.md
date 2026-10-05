# Image-Based Crop Disease Detection with Federated Learning

This project implements Federated Learning (FL) for image-based crop disease classification using selected subsets of the PlantVillage dataset.

The main objective is to evaluate whether multiple clients can collaboratively train a global deep learning model without transferring their raw training images to a central server.

The implementation uses FedAvg (Federated Averaging) with three CNN models:

- ResNet50
- VGG16
- MobileNetV2

Four crop datasets were used:

- Grape
- Apple
- Corn
- Tomato

A total of 12 experiments were conducted using 4 datasets × 3 models.

---

## Project Overview

In conventional centralized machine learning, training images from different sources are collected at a central location.

In this project, the training data is divided among multiple simulated clients. Each client trains the model using its own local images, and only the model parameters/updates are used for server-side aggregation.

The server applies weighted Federated Averaging (FedAvg) to obtain a new global model.

The process is repeated for 30 communication rounds, and the global model is evaluated after every round.

### Main Workflow

1. Dataset selection
2. Image preprocessing
3. Training-data partitioning among 3 clients
4. Pretrained model initialization
5. Local client training
6. Weighted FedAvg aggregation
7. Global model evaluation
8. Repetition for 30 communication rounds

---

## Models Used

| Model | Description |
|---|---|
| ResNet50 | Deep residual CNN architecture |
| VGG16 | Deep convolutional neural network |
| MobileNetV2 | Lightweight CNN architecture |

All models were initialized using pretrained weights and adapted for the corresponding crop-disease classification task.

---

## Datasets

Selected subsets of the PlantVillage dataset were used.

| Dataset | Classes | Training Images | Validation Images | Total |
|---|---:|---:|---:|---:|
| Grape | 4 | 3,251 | 811 | 4,062 |
| Apple | 4 | 2,536 | 635 | 3,171 |
| Corn | 4 | 3,080 | 772 | 3,852 |
| Tomato | 10 | 14,529 | 3,631 | 18,160 |

The images were resized to 224 × 224 pixels and normalized using statistics calculated from the training data.

---

## Federated Learning Configuration

| Parameter | Value |
|---|---|
| Number of Clients | 3 |
| Communication Rounds | 30 |
| Local Epochs | 1 |
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Optimizer | SGD |
| Loss Function | CrossEntropyLoss |
| Image Size | 224 × 224 |
| Aggregation | Weighted FedAvg |
| Random Seed | 42 |
| Framework | PyTorch |
| GPU | NVIDIA Tesla T4 |

The training data for each crop dataset was randomly divided into approximately equal client partitions.

---

## How Federated Learning Works

For each communication round:

1. The server sends the current global model to the clients.
2. Each client trains the model on its local training data.
3. Each client completes one local training epoch.
4. Client model parameters are sent to the server.
5. The server performs weighted FedAvg aggregation.
6. A new global model is produced.
7. The global model is evaluated on the validation dataset.
8. The process continues for the next communication round.

The raw training images remain within their assigned client partitions in this simulated setup.

> Note: This implementation demonstrates decentralized training through client simulation. It does not provide formal privacy guarantees such as encryption, secure aggregation, or differential privacy.

---

## Project Files

```text
Crop-Disease-Federated-Learning/
│
├── federated_crop_disease.ipynb
│
├── Internship_Report.pdf
│
├── Internship_Presentation.pptx
├── Internship_Presentation.pdf
│
├── grape_accuracy_vs_rounds.png
├── apple_accuracy_vs_rounds.png
├── corn_accuracy_vs_rounds.png
├── tomato_accuracy_vs_rounds.png
│
├── grape_f1_vs_rounds.png
├── apple_f1_vs_rounds.png
├── corn_f1_vs_rounds.png
├── tomato_f1_vs_rounds.png
│
├── overall_model_comparison.png
├── overall_model_accuracy_f1_green.png
├── figure09_final_accuracy_comparison.png
├── figure10_final_f1_comparison.png
├── figure11_average_accuracy_by_model.png
├── figure12_execution_time_comparison.png
└── figure13_dataset_size_comparison.png-
