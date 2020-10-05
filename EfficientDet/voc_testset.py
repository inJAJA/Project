import os

PATH

def create_testset(PATH):
    f = open(PATH, 'r')
    trainset = f.readlines()
    dataset =