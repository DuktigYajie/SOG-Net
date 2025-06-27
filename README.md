# Sum-of-Gaussians Neural Network (SOG-Net): </small> </small> A Machine-Learning Interatomic Potential for Long-Range Systems</small> </small>

## Summary
Sum-of-Gaussians Neural Network (SOG-Net) is a lightweight and versatile framework for integrating long-range interactions into machine learning force field. The SOG-Net employs a latent-variable learning network that seamlessly bridges short-range and long-range components, coupled with an efficient Fourier convolution layer that incorporates long-range effects. By learning sum-of-Gaussians multipliers across different convolution layers, the SOG-Net adaptively captures diverse long-range decay behaviors while maintaining close-to-linear computational complexity during training and simulation via non-uniform fast Fourier transforms.

Authors: Yajie Ji, Jiuyang Liang, Zhenli Xu. 

Paper Links: [ArXiv](https://arxiv.org/abs/2502.04668)

## Requirements
- Python 3.10.9 or higher
- Tensorflow-gpu
- FINUFFT (tensorflow version)
- ASE (Atomic Simulation Environment)

## Installation
Please refer to the ```setup.py``` file for installation instructions.

## Quick Start
Example scripts can be found in ```\Deep-SOG\examples``` and each numerical example folder in ```\CACE-SOG```, which are based on the [DeepMD](https://github.com/deepmodeling/deepmd-kit) short-range descriptor and the [CACE](https://github.com/BingqingCheng/cace) descriptor, respectively.

## License
This project is licensed under the MIT License.

## Citation
```
@misc{ji2025machinelearninginteratomicpotentialslongrange,
      title={Machine-Learning Interatomic Potentials for Long-Range Systems}, 
      author={Yajie Ji and Jiuyang Liang and Zhenli Xu},
      year={2025},
      eprint={2502.04668},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2502.04668}, 
}
```

## Contact
For any queries regarding SOG-Net, please contact Yajie Ji (jiyajie595@sjtu.edu.cn) or Jiuyang Liang (jliang@flatironinstitute.org).
