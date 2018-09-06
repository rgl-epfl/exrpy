# exrpy
Simple standalone library to read OpenEXR files in Python

## Installation
1. Clone the repository recursively
```bash
git clone --recursive https://github.com/rgl-epfl/exrpy.git
```
2. Install using pip
```bash
pip install ./exrpy
```
This step might take a moment, as it has to build the OpenEXR library.

## Usage
Currently the usage is limited to loading RGB files as
```python
import exrpy
img = exrpy.read('my-exr-file.exr')
```
