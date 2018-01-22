
description of project:
1. This purpose of this project is to reduce the dimensions of data by using principle 
components analysis(PCA), and also perform naive hyperparameter search in multiprocessing
way.

2. Reuse the modeling code and dispatch in multiprocessing way, so that it can take advantages
of multiple-core CPU. I did many trails to have a good guess on initial hyperparameter, which is
the number of components by setting a accumulated variance threshold 70%. I used a naive while
loop to increment the number until finding the wanted variance level.

3. Original features in data are in the range from 4000 and 5000, it reduces to around 200 features
with total variance 74%.

4.200-feature model still kinda of big to perrform linear model type of training.

5.Try factor analysis, and also could add Bayesian optimization for hyperparameter

6.Use Pipeline and other tools to make the process easier.
