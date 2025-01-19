# ArtisticVision

This project focuses on applying artistic styles to images using deep learning techniques. The goal is to transform the visual appearance of an image to match the style of a reference image.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Style Transfer is a technique that applies the visual style of one image to another image. This project leverages neural networks to achieve high-quality style transfer.

## Features
- Apply artistic styles to images
- Fast image processing
- Support for various neural network architectures
- Easy-to-use command-line interface

## Installation
To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Yuval728/ArtisticVision.git
cd ArtisticVision
pip install -r requirements.txt
```

## Usage
To apply style transfer to an image, use the following command:

```bash
python stylize.py --content_image path/to/input/image.jpg --model model/model.pt --output_image path/to/output/image.jpg --preserve_color   
```

## Examples
Here are some examples of images processed with different styles:

- Original Image: [![Original Image](assets/dancing.jpg)](assets/dancing.jpg)
- Styled Image 1: [![Styled Image 1](assets/output.jpg)](assets/output.jpg)
- Styled Image 2 (Preserve Color): [![Styled Image 2](assets/output_color.jpg)](assets/output_color.jpg)

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
