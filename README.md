# Physics-Informed Neural Network (PINN) for Thermal Degradation API

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scientific ML](https://img.shields.io/badge/Domain-Scientific_Machine_Learning-blue?style=for-the-badge)

## 📌 Project Overview
This repository contains the architecture, training pipeline, and deployment code for a **Physics-Informed Neural Network (PINN)** designed to model thermal degradation in solid-state battery nanomaterials. 

Unlike traditional data-driven deep learning models, this PINN embeds the laws of thermodynamics directly into its loss function. By calculating the exact derivatives of the network's outputs with respect to its inputs using PyTorch's `autograd`, the model is penalized if its predictions violate the 1D Heat Equation. This results in a highly data-efficient AI capable of solving complex Partial Differential Equations (PDEs) for predictive maintenance and materials science applications.

The final trained model is packaged and served as a real-time REST API using **FastAPI**, demonstrating an end-to-end transition from theoretical physics to enterprise cloud architecture.

---

## 🔬 The Physics & Mathematics
The core physical phenomena modeled is the thermal diffusion across a 1D material space over time. 

The governing Partial Differential Equation (1D Heat Equation) is:
∂u/∂t = α(∂²u/∂x²)

Where:
* u is the temperature.
* t is time.
* x is the spatial coordinate.
* α is the thermal diffusivity of the material (α = 0.01).

### The PINN Loss Function
The model's total loss is a weighted sum of the Mean Squared Error (MSE) from a small subset of known boundary/initial conditions and the physical residual (the PDE loss) evaluated at random collocation points:
Loss_total = Loss_data + λ(Loss_PDE)

---

## 🚀 Key Features & Architecture
1. **Physics Baseline (FDM):** A highly stable Finite Difference Method solver was built to generate the exact ground-truth thermodynamic data, utilizing the Courant-Friedrichs-Lewy (CFL) stability criterion.
2. **Custom PyTorch Architecture:** A 3-layer Multi-Layer Perceptron (MLP) utilizing `Tanh` activation functions to ensure the network remains infinitely differentiable for exact spatial and temporal gradients.
3. **Data-Scarce Learning:** The model was trained using only 2,000 exact data points and 10,000 unsupervised collocation points, proving the efficiency of physics-guided regularization.
4. **Production API Endpoint:** The trained PyTorch weights are loaded into a FastAPI server, allowing external applications to pass JSON payloads (position and time) and receive live thermal degradation predictions.

---

## 📊 Results & Performance
The model achieved a phenomenal convergence, successfully learning the physics of the system rather than just overfitting to the data.

* **Relative L2 Error:** **5.53%** (Achieved in just 5000 epochs)

*(Note: The exact FDM simulation and PINN error heatmaps are available in the repository images).*

---

## 💻 Project Structure
    ├── PINN_Thermal_Simulation.ipynb  # Complete Colab notebook (Data Generation & Training)
    ├── score.py                       # Inference script for cloud deployment
    ├── conda.yml                      # Environment dependencies for Azure ML
    ├── thermal_pinn_weights.pth       # Trained PyTorch model weights
    ├── README.md                      # Project documentation

---

## ⚙️ How to Run the API Locally
To test the FastAPI deployment locally:

1. Clone the repository and install the dependencies:
    ```bash
    pip install torch fastapi uvicorn nest-asyncio requests
    ```

2. Run the server utilizing the `score.py` logic (refer to the Colab notebook for the FastAPI wrapper implementation).

3. Send a test JSON payload to the endpoint:
    ```python
    import requests
    payload = {"x": 0.2, "t": 0.4}
    response = requests.post("[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)", json=payload)
    print(response.json())
    ```
    **Expected Output:** `{"predicted_temperature": 0.02513}`

---

## 👨‍🔬 About the Author
**Ammar Yasir Hattar** *Physicist & AI Engineer* Bridging the gap between theoretical physics and enterprise artificial intelligence. With a BS in Physics from the International Islamic University Islamabad and a deep focus on quantum technologies, nanomaterials, and autonomous AI agents, my work focuses on translating complex mathematical models into scalable, production-ready AI solutions.

**Previous Projects:**  [NanoGap AI (Hugging Face)](#) 
* [Agentic Molecular Dynamics](#) 
* [Quantum Entanglement RL](#)
