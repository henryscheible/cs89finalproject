# CS89 Final Project: Henry, Andrew, Zach, Eli, Jeff

The goal of our project was to investigate the relationship between the performance of Deep Networks and how well we can compress their representation (e.g. weight matrix)

To train our model with default parameters run:

```python3 train.py```

You can also experiment various arg options which are displayed in `parameters.py`. For instance, to run our model for 10 epochs with a dropout probability of 0.3 you can run:

```python3 train.py --epochs=10 --dropout=0.3```