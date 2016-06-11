# Speakers

**A library for home-grown DSP speaker crossovers and EQ**

![Speaker of the House Paul Ryan](http://www.speaker.gov/sites/speaker.house.gov/files/files/2015/10-29-15%20at%2010-56-42-2.jpg)

## Installation

### Install deps


#### Apt

We currently have a make command to install dependencies from apt:

```bash
cd speakers
sudo make install-deps-apt
```


#### Manually

The following dependencies are needed to build the C library:

- [PortAudio](http://www.portaudio.com/)
- [FFTW3](http://www.fftw.org/)
- [libsndfile](http://www.mega-nerd.com/libsndfile/)

The following command should get your python environment set up:

```bash
$ cd speakers
$ pip install -e .
```


### Build *libfilter*

```bash
cd speakers
make all
```


## Authors

- Mike Axiak
- Michael Price

