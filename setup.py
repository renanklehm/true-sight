from distutils.core import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'truesight',
    packages = ['truesight'],
    version = 'v0.0.1-alpha',
    description = 'Truesight is a python package for time series prediction using deep learning and statistical models.',
    author = 'Renan Otvin Klehm',
    author_email = 'renanotivin@hotmail.com',
    url = 'https://github.com/renanklehm/true-sight',
    download_url = 'https://github.com/renanklehm/true-sight/archive/refs/tags/v0.0.1-alpha.tar.gz',
    keywords = ['time series', 'prediction'],
    classifiers = [],
    long_description=long_description,
    long_description_content_type='text/markdown'
)