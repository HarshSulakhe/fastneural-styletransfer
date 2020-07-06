import torch as th

def gram(tensor):
    b,c,h,w = tensor.size()
    features = tensor.view(b*c,h*w)
    return torch.mm(features,features.t()).div(b*c*h*w)

def show(img,ax,title):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    ax.set_title(title,fontweight = "bold", size = 24)
    ax.set_xticks([])
    ax.set_yticks([])

class SaveOutput:
    """ Class to call hooks on necessary layers of VGG"""
    def __init__(self,indices):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
