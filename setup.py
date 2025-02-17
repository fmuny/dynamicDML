from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='dynamicDML',
    packages=['dynamicDML'],
    version='0.1.0',
    license='MIT',
    description='A Python implementation of dynamic Double Machine Learning (DML).',
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Fabian Muny',
    author_email='fabian.muny@unisg.ch',
    # url='https://orf-lab.github.io',
    # download_url='https://github.com/fmuny/ORFpy/archive/refs/tags/v0.2.0.tar.gz',
    keywords=['double machine learning', 'dynamic policy'],
    install_requires=[
        'flaml[automl]>=2.3.3',
        'matplotlib>=3.10.0',
        'mgzip>=0.2.1',
        'numpy>=2.2.3',
        'pandas>=2.2.3',
        'scikit-learn>=1.6.1',
        'scipy>=1.15.2',
        'seaborn>=0.13.2'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
