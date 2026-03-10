# Inverse Pendulum MLP modeling

https://github.com/user-attachments/assets/7c768df4-f280-4753-b72d-65f352cf17ce

### File structure :
1. Inverted-Pendulum-viz.py -> Modeling awal inverted Pendulum menggunakan library gymnasium dari pygame
2. cartpole_model.pth -> Model cartpole ANN yang disimpan
3. Generate_dataset -> Code untuk menggenerasikan dataset menggunakan game engine
4. Training_MLP -> Training MLP dengan architecture 5 -> 128, 128 -> 128, 128 -> 64, 64 -> 4 dan Relu untuk setiap activation function
5. x, day Y scaler -> Scaler untuk keperluan visualisasi

### Hyperparameters:
1. Epoch                : 1000
2. Optimizer            : Adam
3. Activation function  : Relu
4. Batch                : 256 menggunakan Mini Batches

### Data Training yang dibuat:
[state sekarang] -> [State Selanjutnya]
time interval : 0.2 S
Jumlah data   : 500
