from setuptools import setup, find_packages

setup(
    name='SOG_Net',
    version='0.1.0',
    description='Sum-of-Gaussian Neural Network',
    # long_description=readme,
    author='Yajie Ji, Jiuyang Liang',
    author_email='jiyajie595@sjtu.edu.cn',
    url='',
    # license=license,
    install_requires=['numpy', 'scipy', 'numba', 'tensorflow-gpu==2.8.1'],
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                "License :: MIT License",
                "Operating System :: OS Independent",],
)