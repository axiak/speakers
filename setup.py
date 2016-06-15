from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='speakers',
    version='0.0.1',
    description='A library for DSP speakers',
    long_description=long_description,
    url='https://github.com/axiak/speakers',
    author='Mike Axiak',
    author_email='mike@axiak.net',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Nerds',
        'License :: OSI Approved :: MIT License',
    ],

    keywords='dsp fft filters fir speakers audio',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        "numpy",
        "scipy",
        "pyaudio",
        "matplotlib",
        "cffi",
        "picos",
    ],
    entry_points={
        'console_scripts': [
            'runspeakers=filter.runspeakers:main',
        ],
    },
)
