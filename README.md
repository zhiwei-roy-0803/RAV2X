# RAV2X
* Resource allocation in V2X-D2D scenario. This project implements various resource allocation, i.e., RB sharing and power control, algorithms. 

* The algorithms implemented here are from doctoral dissertation of Dr. Le Liang from Geogia Institute of Technology, USA.

* The reference paper, which is the doctoral thesis of Dr. Le Liang, is the PDF file in this repo.

# How to use
* Run resource allocation in a vehicular communication environment which only considers large scale fading.
```python
python simulation_entry_large_scale # run algorithm in Chapter2
```
* Run resource allocation in a vehicular communication environment which considers both large scale fading and fast fading, as well as delayed feedback.
```python
python simulation_entry_delayCSI # run algorithm in Chapter3
```
* Run resource allocation based on graph matching theory in a vehicular communication environment which considers both large scale fading and fast fading.
```python
python simulation_entry_Graph # run algorithm in Chapter4
```
* Run resource allocation in a vehicular communication environment with Multi-Agent Reinforcement Learning
```python
python marl_train # run algorithm in Chapter5
```

# Some Notes
* MARL based algorithm sometimes may not produce similar results as shown in the thesis and I do not know the reason as well. If someone find 
out the  reason, please commit an issue to let me know even if I am not working on this problems now. I am curious about why RL algorithms fail in such environment

* As for other three algorithms, just running the simulation code, and you will obtain the similar results presented in the thesis provided that your configuration is the same as those in the thesis.

# Requirements
* Pytorch
* torchtracer
* numpy
* matplotlib
* munkres
* tqdm