# Nonlinear Algebraic Approximation of Control Systems

A Python implementation of an algorithm for construction homogeneous 
approximations of nonlinear control systems. For description of the algorithm refer to ...

## Installation

You can install the package using pip

```
pip install napalm-control
```

## How to use

The main part of the package is implemented in a single class called ControlSystem. This class describes a control system and its methods implement the algorithm of approximation of a given system.

```python
from napalm_control.approximation_tools import ControlSystem
```

Given a system ![equation](https://latex.codecogs.com/gif.latex?%5Cdot%7Bx%7D%3Da%28t%2Cx%29%20&plus;%20b%28t%2Cx%29u)
you can initialize it the following way:

```python
system = ControlSystem(a, b)
```

where _a_ and _b_ are corresponding vectors.

__Note__: the vectors __must__ be composed out of sympy symbols or numbers.
For example, consider a system:


![equation](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%20%5Cbegin%7Baligned%7D%20%5Cdot%20x_1%20%26%3D%20-u%20%5C%5C%20%5Cdot%20x_2%20%26%3D%20-%5Cfrac12x_1%5E2-4tx_1-3t%5E2x_1%20%5C%5C%20%5Cdot%20x_3%20%26%3D%20-x_1%5E2-2tx_1-3t%5E2x_1%20%5Cend%7Baligned%7D%20%5Cright.)

Then the implementation using the package would be as follows:
```python
import sympy as sym
from napalm_control.approximation_tools import ControlSystem

x1, t = sym.symbols('x_{1} t')
a = sym.Matrix([0, -sym.Rational(1, 2)*x1**2 - 4*t*x1 - 3*t**2*x1, -x1**2 - 2*t*x1 - 3*t**2*x1])
b = sym.Matrix([-1, 0, 0])
system = ControlSystem(a, b)
```

Then you have 2 options: you can either approximate this system using nonlinear power moments series or using Fliess series

```python

# to approximate using nonlinear power moments algebra
system.calc_approx_system()

# to approximate using Fliess algebra
system.calc_approx_system(fliess=True)

```

To generate the pdf file you need to do as follows:

```python

# to generate pdf for systems which were appoximated using Fliess series
# add additional argument fliess=True
system.generate_pdf()

```

This will produce a pdf file named 'output.pdf' in your current working directory with all the necessary information of the system and its approximation.

To customize filename and directory where the file will be saved use additional parameters

```python

# generate pdf in custom directory and custom filename (filename does
# not need to include .pdf extension, it will be added automatically, so just provide the name)
system.generate_pdf(path="path_to_directory", filename="custom_filename")

```

