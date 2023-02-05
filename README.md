# DENO4pytorch
 Differential equation neural operator

## Operator Learning Models

### FNO (Fourier Neural Operator)

### UNet (Convolutional neural network)

### deepONet (Fourier Neural Operator)

## Training strategies

### Data driven only (Supervised learning)

### Physics-informed (Self-Supervised learning)

### Physics-Data fusion (Semi-Supervised learning)


```
📂 DENO4pytorch
|_📁 Models
  |_📄 basic_layers.py  
  |_📁 basic        # Model: Basic neural layers
    |_📄 basic_layers.py  
  |_📁 fno          # Model: Fourier Neural Based
    |_📄 FNOs.py
    |_📄 spectrual_layers.py
  |_📁 cnn          # Model: Convolutional Neural Network Based (UNet supported)
    |_📄 conv_layers.py
    |_📄 ConvNets.py
  |_📁 gnn # Model: # Model: Graph Neural Network
    |_📄 graph_layers.py
    |_📄 GraphNets.py
  |_📁 transformer  # Model: Transformer Based
    |_📄 graph_layers.py
    |_📄 GraphNets.py
  |_📁 pinn         # Model: autograd to derivate output of networks
    |_📄 differ_layers.py
|_📁 Utilizes  # Tools: Scripts to statistic and plot
  |_📁 config
  |_📄 loss_metrics.py
  |_📄 process_data.py
  |_📄 visual_data.py
|_📁 Demo      # Demos: Scripts and Data to run demo scripts
  |_📁 config
  |_📁 Advection_2d
    |_📁 data  #  Data for training and valid
    |_📁 gen   #  Scripts to generate data
    |_📄 run_deepONet+PINN.py  #  Scripts to run demo
  |_📁 Turbulence_2d
    |_📁 data  #  Data for training and valid
    |_📁 gen   #  Scripts to generate data
    |_📄 run_FNO&UNet.py  #  Scripts to run demo
```