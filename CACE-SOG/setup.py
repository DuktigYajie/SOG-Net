from setuptools import setup, find_packages

setup(
    name='CACE-SOG',
    version='0.1.0',
    description='Sum-of-Gaussians Neural Network Based on Cartesian Atomic Cluster Expansion Descriptor',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'ase<=3.22.1',
        'torch',
        'matscipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

