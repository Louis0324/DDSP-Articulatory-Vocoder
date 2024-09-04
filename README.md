# DDSP Articulatory Vocoder
Official implementation of the paper:

**Fast, High-Quality and Parameter-Efficient Articulatory Synthesis Using Differentiable DSP (SLT 2024)** 

[[paper]](https://arxiv.org/)[[demo]](https://ddsp-vocoder.notion.site/Fast-High-Quality-and-Parameter-Efficient-Articulatory-Synthesis-Using-Differentiable-DSP-b2297b5c67834f5fb7c302b55f8a6df2)

DDSP code is based on https://github.com/sweetcocoa/ddsp-pytorch and https://intro2ddsp.github.io/intro.html

## Environment Setup
- Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Run `conda env create -f environment.yml` to create conda env

## Dataset prep
1. Download paired EMA and speech data, such as [HPRC](https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h).
2. Resample wav to be 16 kHz, and EMA data to be 200 Hz.
3. Use `data_prep/batch_invert.ipynb` or other methods to extract pitch and loudness from waveform at 200 Hz.
4. Use `data_split.ipynb` to generate jsons that define the test/val/train splits

## Training
1. Edit the config yaml, which defines hyperparameters, training parameters, and file directories.
2. Run `python vocoder/main.py --config yamls/config.yaml` from the source directory to train. 

