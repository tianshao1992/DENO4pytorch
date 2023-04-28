import torch
import os
from post_data import Post_2d, get_grid
class MLP(nn.Module):
    def __init__(self, layers, is_BatchNorm=True):
        super(MLP, self).__init__()
        self.depth = len(layers)
        self.activation = nn.GELU
        #先写完整的layerslist
        layer_list = []
        for i in range(self.depth-2):
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            if is_BatchNorm is True:
                layer_list.append(('batchnorm_%d' % i, nn.BatchNorm1d(layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        #最后一层，输出层
        layer_list.append(('layer_%d' % (self.depth-2), nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        #再直接使用sequential生成网络
        self.layers = nn.Sequential(layerDict)

    def forward(self,x):
        y = self.layers(x)
        return y

if __name__ == "__main__":
    #建立模型并读入参数
    name = 'MLP'
    work_path = os.path.join('work', name)

    in_dim = 28
    out_dim = 5

    layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim * 64 * 64]
    Net_model = MLP(layer_mat, is_BatchNorm=False)

    checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'))
    Net_model.load_state_dict(checkpoint['net_model'])

    #输出预测结果