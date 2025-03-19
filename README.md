# Age-related Gait Patterns Classification Using Deep Learning Based on Time-series Data from One Accelerometer
![ ](https://github.com/xzheng93/Age-related_gait_classification/blob/main/result/model_plots/study%20pipeline.png)  

This repository contains the python code for comparing performance of deep learning and conventional machine learning models in classifying age related gait patterns as presented in [Age-related Gait Patterns Classification Using Deep Learning Based on Time-series Data from One Accelerometer](https://doi.org/10.1016/j.bspc.2024.107406).

```
@article{zheng2025age,
  title={Age-related gait patterns classification using deep learning based on time-series data from one accelerometer},
  author={Zheng, Xiaoping and Wilhelm, Elisabeth and Otten, Egbert and Reneman, Michiel F and Lamoth, Claudine JC},
  journal={Biomedical Signal Processing and Control},
  volume={104},
  pages={107406},
  year={2025},
  publisher={Elsevier}
}
```
## Data, Reproducibility, and Code
#### Requirement
python = 3.x   
TensorFlow-gpu >= 2.3  
keras-tuner == 1.1.0  
sklearn >= 0.24.1
### Data and Reproducibility
- datasets and models can be found here (will be released soon)
- log files can be found in [/result/logs](https://github.com/xzheng93/Age-related_gait_classification/tree/main/result/logs)
### Usage
- to tune the hyperparameters for deep learning models 
```
python /tuning/run_dl_tuning.py ${ARGS}
```
- to tune the machine learning models  
```
python /tuning/run_ml_tuning.py ${ARGS}
```

- to compare the performance of deep learning or machine learning, use dl_main.py or ml_main.py
```
python dl_main.py 
python ml_main.py
```
### Results
<table>
  <tr>
    <td><img src="https://github.com/xzheng93/Age-related_gait_classification/blob/main/result/model_plots/ROC%20comparison%20of%20deep%20learning_1.jpg" alt="Figure 1 dl results"></td>
    <td><img src="https://github.com/xzheng93/Age-related_gait_classification/blob/main/result/model_plots/ROC%20comparison%20of%20machine%20learning.jpg" alt="Figure 2 ml results"></td>
  </tr>
</table>
