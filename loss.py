import torch as th
from utils import *

class ContentLoss(th.nn.Module):
    """ Measures the "semantic" difference between the generated image passed through the VGG net and the original content image
    passed through the VGG net"""

    def __init__(self):
        super(ContentLoss,self).__init__()

    def forward(self,generated,content):
        return th.nn.MSELoss(generated,content)

class StyleLoss(th.nn.Module):
    """ Measures how close the style of the generated image is compared to the original style image"""

    def __init__(self):
        super(StyleLoss,self).__init__()

    def forward(self,generated,content):
        g_gen = gram(generated)
        g_con = gram(content)
        return th.nn.MSELoss(g_gen,g_con)
