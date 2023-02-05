# DENO4pytorch (Pytorch Tools for Differential Equation Neural Operator)

## Requirements
- The code only depends on pytorch(>=1.8.0) [PyTorch](https://pytorch.org/) and torch_geometric(>=2.0.3) [TorchGeo](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)  

## Operator Learning Models

### CNN 
- **Convolutional neural network as backbone** 
- 

#### Reference
- Ronneberger, O. et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. Preprint at https://doi.org/10.48550/arXiv.1505.04597 (2015).
- Liu, T. et al. Supervised learning method for the physical field reconstruction in a nanofluid heat transfer problem. International Journal of Heat and Mass Transfer 165, 120684 (2021).
- Du, Q. et al. Airfoil design and surrogate modeling for performance prediction based on deep learning method. Physics of Fluids 20 (2022).
- Li, Y. et al. Deep learning based real-time energy extraction system modeling for flapping foil. Energy 246, 123390 (2022).

[![10.1016/j.ijheatmasstransfer.2020.120684](https://img.shields.io/badge/DOI-doi%10.1016/j.ijheatmasstransfer.2020.120684-red)](https://linkinghub.elsevier.com/retrieve/pii/S0017931020336206)

### FNO 
- **Fourier Neural Operator as backbone**
- 

#### Reference
- Li, Z. et al. Fourier Neural Operator with Learned Deformations for PDEs on General Geometries. Preprint at https://doi.org/10.48550/arXiv.2207.05209 (2022).
- Li, Z. et al. Fourier Neural Operator for Parametric Partial Differential Equations. arXiv:2010.08895 [cs, math] (2021).
- Li, Z. et al. Physics-Informed Neural Operator for Learning Partial Differential Equations. Preprint at http://arxiv.org/abs/2111.03794 (2021).


### deepONet
- **Fully Connected Neural Network as backbone**

#### Reference
- Lu, L. et al. DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators. Nat Mach Intell 3, 218–229 (2021).
- Jin, P. et al. MIONet: Learning multiple-input operators via tensor product. Preprint at http://arxiv.org/abs/2202.06137 (2022).\
- Lu, L. et al. A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data. Computer Methods in Applied Mechanics and Engineering 393, 114778 (2022).
### GNN
- **Graph Neural network as backbone**
- 

#### Reference
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A. & Battaglia, P. W. Learning Mesh-Based Simulation with Graph Networks. arXiv:2010.03409 [cs] (2021).
- Boussif, O., Assouline, D., Benabbou, L. & Bengio, Y. MAgNet: Mesh Agnostic Neural PDE Solver. Preprint at http://arxiv.org/abs/2210.05495 (2022).
- Shukla, K. et al. Scalable algorithms for physics-informed neural and graph networks. Preprint at http://arxiv.org/abs/2205.08332 (2022).

### Trasnformer 
- **Attention mechanism as backbone**

#### Reference
- Cao, S. Choose a Transformer: Fourier or Galerkin. Preprint at http://arxiv.org/abs/2105.14995 (2021). 
- Kissas, G. et al. Learning Operators with Coupled Attention. Preprint at http://arxiv.org/abs/2201.01032 (2022).
- Li, Z., Meidani, K. & Farimani, A. B. Transformer for Partial Differential Equations’ Operator Learning. Preprint at http://arxiv.org/abs/2205.13671 (2022).


## Training strategies

### Data-driven only (Supervised learning)
- 
#### Reference
- Guo, X. et al. Convolutional Neural Networks for Steady Flow Approximation. in The 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 481–490 (2016). doi:10.1145/2939672.2939738.
- Lee, S. et al. Prediction of laminar vortex shedding over a cylinder using deep learning. arXiv:1712.07854 [physics] (2017).
- Sekar, V. et al. Fast flow field prediction over airfoils using deep learning approach. Physics of Fluids 31, 057103 (2019).

### Physics-informed (Self-Supervised learning)

#### Reference
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378, 686–707 (2019).
- Raissi, M., Yazdani, A. & Karniadakis, G. E. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. Science 367, 1026–1030 (2020).
- Wang, S. et al. When and why PINNs fail to train: A neural tangent kernel perspective. arXiv:2007.14527 [cs, math, stat] (2020).

### Physics-data fusion (Semi-Supervised learning)

#### Reference
- Raissi, M. et al. Deep learning of vortex-induced vibrations. J. Fluid Mech. 861, 119–137 (2019).
- Lu, L. et al. Physics-informed neural networks with hard constraints for inverse design. arXiv:2102.04626 [physics] (2021).
- Rohlfs, L. et al. Discovering Latent Physical Variables from Experimental Data in Supersonic Flow using Physics-Informed Neural Networks (PINNs).

## Datasets
- We are collecting some standard dataset mostly used in recent papers.
- Several data generation configuration can be found in the scripts in the Demo.
- Several data are generated in experiments/real-world in our research and projects.


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
|_📁 Utilizes       # Tools: Scripts to statistic and plot
  |_📁 config
  |_📄 loss_metrics.py
  |_📄 process_data.py
  |_📄 visual_data.py
|_📁 Demo           # Demos: Scripts and Data to run demo scripts
  |_📁 config
  |_📁 Advection_2d
    |_📁 data       #  Data for training and valid
    |_📁 gen        #  Scripts to generate data
    |_📄 run_deepONet+PINN.py  #  Scripts to run demo
  |_📁 Turbulence_2d
    |_📁 data       #  Data for training and valid
    |_📁 gen        #  Scripts to generate data
    |_📄 run_FNO&UNet.py  #  Scripts to run demo
```