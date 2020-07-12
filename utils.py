import torch as th

def gram(tensor):
    b,c,h,w = tensor.size()
    features = tensor.view(b*c,h*w)
    return th.mm(features,features.t()).div(b*c*h*w)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def show(img,ax,title):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    ax.set_title(title,fontweight = "bold", size = 24)
    ax.set_xticks([])
    ax.set_yticks([])

class SaveOutput:
    """ Class to call hooks on necessary layers of VGG"""
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
