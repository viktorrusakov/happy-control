import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="napalm_control",
    version="0.0.1",
    author="Viktor Rusakov",
    author_email="vrusakov66@gmail.com",
    description="Nonlinear Algebraic Approximation in Control systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requres=[
        'antlr4-python3-runtime>=4.8',
        'Cython>=0.29.14',
        'numpy>=1.18.1',
        'sympy>=1.5.1'
    ]
)