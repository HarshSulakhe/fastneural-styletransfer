import torch as th
from utils import *

class ContentLoss(th.nn.Module):
    """ Measures the "semantic" difference between the generated image passed through the VGG net and the original content image
    passed through the VGG net"""

    def __init__(self):
        super(ContentLoss,self).__init__()

    def forward(self,generated,content):
        mse_loss = th.nn.MSELoss()
        return mse_loss(generated,content)

class StyleLoss(th.nn.Module):
    """ Measures how close the style of the generated image is compared to the original style image"""

    def __init__(self):
        super(StyleLoss,self).__init__()

    def forward(self,generated,style):
        g_gen = gram_matrix(generated)
        g_style = gram_matrix(style)
        mse_loss = th.nn.MSELoss()
        # print(g_gen.size(),g_style.size())
        return mse_loss(g_gen,g_style)
