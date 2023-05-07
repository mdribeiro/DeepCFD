from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='deepcfd',
    version='2.0.0',
    description='DeepCFD: Efficient Steady-State Laminar Flow'
        'Approximation with Deep Convolutional Neural Networks',
    author='Mateus Dias Ribeiro',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mdribeiro/DeepCFD',
    packages=['deepcfd', 'deepcfd.models'],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch==2.0.0",
        "torchvision==0.15.1",
        "matplotlib>=3.0.0,<=3.7.1"
    ],
)
