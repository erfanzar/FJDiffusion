# FJDiffusion

Implementation and Pretrained Diffusion Models in JAX

## Description

This project provides an implementation of diffusion models in JAX, along with pretrained models. Diffusion models are a
class of generative models that can be used for tasks such as image generation, inpainting, denoising, and
super-resolution. This project aims to make it easy for researchers and developers to use diffusion models in their own
projects.

## Features

- Implementation of diffusion models in JAX
- Pretrained models for various tasks
- Easy-to-use API for training and inference
- Support for distributed training on multiple GPUs or TPUs
- Extensive documentation and examples

## Installation

To install the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/erfanzar/FJDiffusion.git
   ```

2. Change into the project directory:

   ```bash
   cd FJDiffusion
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, follow these steps:

1. Import the necessary modules:

   ##### TODO

2. Load a pretrained model:

   ##### TODO

3. Generate samples from the model:

   ##### TODO

4. Perform inference on an input:

   ##### TODO

For more detailed usage instructions and examples, please refer to the documentation.

## Getting PartitionRules

here's how you can get partition rules of each model in order to use them for pjit and fsdp

#### Unet2D

```python
from FJDiffusion.moonwalker.configs import Unet2DConfig

partition_rules = Unet2DConfig.get_partition_rules(fully_fsdp=True)
```

#### VAE

```python
from FJDiffusion.moonwalker.configs import AutoencoderKlConfig

partition_rules = AutoencoderKlConfig.get_partition_rules(fully_fsdp=True)
```

#### CLIPTextModel

```python
from FJDiffusion.moonwalker.configs import get_clip_partition_rules

partition_rules = get_clip_partition_rules(fully_fsdp=True)
```

## Contributing

Contributions to this project are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

Please make sure to follow the code style and conventions used in the project.

## License

This project is licensed under the Apache v2.0 License. See
the [LICENSE](https://github.com/erfanzar/FJDiffusion/blob/main/LICENSE) file for more information.

## Acknowledgements

This project is built upon the work of only one researcher / developer. I would like to say if there's any problem
in open-source implementations and pretrained models after final releases please let me know <3.

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact me at
erfanzare810@gmail.com