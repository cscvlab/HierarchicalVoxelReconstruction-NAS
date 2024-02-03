import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import distributions as dist

import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# net utils
# -------------------------------------------------------
@torch.no_grad()
def gradient(x: torch.Tensor, f: nn.Module):
    
    N = x.shape[0]
    h = 1.0 / (256.0 * 3.0)
    k0 = torch.Tensor([ 1.0, -1.0, -1.0]).to(x.device)
    k1 = torch.Tensor([-1.0, -1.0,  1.0]).to(x.device)
    k2 = torch.Tensor([-1.0,  1.0, -1.0]).to(x.device)
    k3 = torch.Tensor([ 1.0,  1.0,  1.0]).to(x.device)
    h0 = torch.Tensor([ h, -h, -h]).to(x.device)
    h1 = torch.Tensor([-h, -h,  h]).to(x.device)
    h2 = torch.Tensor([-h,  h, -h]).to(x.device)
    h3 = torch.Tensor([ h,  h,  h]).to(x.device)
    h0 = x + h0
    h1 = x + h1
    h2 = x + h2
    h3 = x + h3
    h0 = h0
    h1 = h1
    h2 = h2
    h3 = h3
    h0 = k0 * f(h0).reshape(N,1)
    h1 = k1 * f(h1).reshape(N,1)
    h2 = k2 * f(h2).reshape(N,1)
    h3 = k3 * f(h3).reshape(N,1)
    grad = (h0+h1+h2+h3) / (h*4.0)

    return grad

def network_parameters(model: nn.Module):
    if model is None:
        return 0
    return sum([param.numel() for param in model.parameters() if param.requires_grad])

def batchify_run_network(net: nn.Module, input: torch.Tensor, batch_size=262144):
    ret = []
    i = 0
    while i < input.shape[0]:
        input_batch = input[i:i+batch_size, ...] if i + batch_size < input.shape[0] else input[i:, ...]
        output = net(input_batch)
        ret.append(output)
        i+=batch_size
    ret = torch.cat(ret, dim=0)
    return ret

def batchify_run_network_cuda(net: nn.Module, input: torch.Tensor, batch_size=262144):
    net = net.cuda()
    input = input.cuda()
    ret = batchify_run_network(net, input, batch_size)
    net= net.cpu()
    input = input.cpu()
    ret = ret.cpu()
    return ret 
    
def batchify_run_network_with_grad(net: nn.Module, input: torch.Tensor, batch_size=262144, mode='auto'): # finite
    ret, grad = [], []
    i = 0
    while i < input.shape[0]:
        input_batch = input[i:i+batch_size, ...] if i + batch_size < input.shape[0] else input[i:, ...]
        
        if mode == 'auto':
            input_batch.requires_grad = True
            output = net(input_batch)
            
            grad.append(torch.autograd.grad(output, [input_batch], grad_outputs=torch.ones_like(output))[0].detach())
            ret.append(output.detach())
            
        elif mode == 'finite':
            with torch.no_grad():
                output = net(input_batch)
            
            grad.append(gradient(input_batch, net))
            ret.append(output)
            
        i+=batch_size
        
    ret = torch.cat(ret, dim=0)
    grad = torch.cat(grad, dim=0)
    return ret, grad
  
def quantization_int8(net: nn.Module):
    net_int8 = torch.ao.quantization.quantize_dynamic(
        net, {torch.nn.Linear}, dtype=torch.qint8
    )
    return net_int8
     
# --------------------------------------------------------
# Embedding
# --------------------------------------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, 
                 level:int=6,
                 include_input:bool=True,  
                 in_features:int=3):
        super(PositionalEmbedding, self).__init__()
        self.in_features = in_features
        self.level = level
        self.include_input = include_input

        c = [2**i for i in range(self.level)]  # [level]
        zero = [0 for i in range(self.level)]
        coff = []
        for i in range(self.in_features):
            l = []
            for j in range(self.in_features):
                if i == j:
                    l += c
                else:
                    l += zero
            coff.append(l)

        self.coff = torch.Tensor(coff) # [in_features, level*in_features]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor. Shape: [N, in_features]
        """
        tmp = torch.matmul(x, self.coff)  # [N, in_features] * [in_features, level] = [N, level]
        if self.include_input:
            res = [torch.sin(tmp), torch.cos(tmp), x]   # [N, level] [N, level] [N, in_features] 
        else:
            res = [torch.sin(tmp), torch.cos(tmp)]
        res = torch.cat(res, dim=1)
        return res

    def cuda(self, device = None):
        self.coff = self.coff.cuda(device)
        return super(PositionalEmbedding, self).cuda(device)
    
    def cpu(self):
        self.coff = self.coff.cpu()
        return self
    
    def extra_repr(self):
        return 'level={}, include_input={}'.format(
            self.level, self.include_input
        )

class TriPlaneEmbedding(nn.Module):
    def __init__(self, resolution: int, channel: int):
        super(TriPlaneEmbedding, self).__init__()
        self.resolution = resolution
        self.channel = channel
        
        self.xy = nn.Parameter(torch.randn(1, self.channel, self.resolution+1, self.resolution+1) * 0.01)
        self.yz = nn.Parameter(torch.randn(1, self.channel, self.resolution+1, self.resolution+1) * 0.01)
        self.xz = nn.Parameter(torch.randn(1, self.channel, self.resolution+1, self.resolution+1) * 0.01)
        
    def forward(self, xyz):
        xLen = xyz.shape[0]
        
        fxy = F.grid_sample(self.xy, xyz[:, [0, 1]].reshape(1, xLen, 1, 2), align_corners=True, padding_mode='border', mode='bilinear').squeeze(0).squeeze(-1).transpose(0,1)#[0, :, : 0]#.transpose(0, 1)
        fyz = F.grid_sample(self.yz, xyz[:, [1, 2]].reshape(1, xLen, 1, 2), align_corners=True, padding_mode='border', mode='bilinear').squeeze(0).squeeze(-1).transpose(0,1)#[0, :, : 0].transpose(0, 1)
        fxz = F.grid_sample(self.xz, xyz[:, [0, 2]].reshape(1, xLen, 1, 2), align_corners=True, padding_mode='border', mode='bilinear').squeeze(0).squeeze(-1).transpose(0,1)#[0, :, : 0].transpose(0, 1)
        
        return torch.cat([fxy, fyz, fxz], dim=-1)
        
    def extra_repr(self):
        return 'resolution={}, channel={}'.format(self.resolution, self.channel)

class MLPEmbedding(nn.Module):
    def __init__(self, in_features, channel, out_features=32, num_layers=2):
        super(MLPEmbedding, self).__init__()
        self.in_features = in_features
        self.channel = channel
        self.out_features = out_features
        self.num_layers = num_layers
        self.net = self._build_net(in_features, channel, out_features, num_layers)

    def _build_net(self):
        net = []
        
        input_shape = self.in_features
        for i in range(self.num_layers-1):
            net += [nn.Linear(input_shape, self.channel), nn.ReLU()]
            input_shape = self.channel
            
        net.append(nn.Linear(input_shape, self.out_features))
        return nn.Sequential(nn.ModuleList(net))

    def forward(self, x):
        return self.net(x)
    
class TriMLPEmebedding(nn.Module):
    def __init__(self, num_layer: int, channel: int, out: int):
        super(TriMLPEmebedding, self).__init__()
        self.xy = MLPEmbedding(2, channel, out, num_layer)
        self.yz = MLPEmbedding(2, channel, out, num_layer)
        self.xz = MLPEmbedding(2, channel, out, num_layer)
    
    def forward(self, xyz):
        fxy = self.xy( xyz[:, [0, 1]] ) 
        fyz = self.yz( xyz[:, [1, 2]] ) 
        fxz = self.xz( xyz[:, [0, 2]] ) 
        
        return torch.cat([fxy, fyz, fxz], dim=-1)

# ----------------------------------------------------------
# activations modules
# ----------------------------------------------------------

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return F.sigmoid(x) * x

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)

class Expo(nn.Module):
    def __init__(self):
        super(Expo, self).__init__()
        
    def forward(self, x):
        return torch.exp(x)

ActivationDict = {
    'relu': nn.ReLU,
    'swish': Swish,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'sin': Sine,
    'tanh': nn.Tanh,
    'expo': Expo
}

class StandardNet(nn.Module):
    def __init__(self, arch: dict, **kwargs):
        super(StandardNet, self).__init__()
        self.arch = arch
        self.pe, self.triplane, self.trimlp, self.net = self._build_net(arch)        
        
    def _build_net(self, arch: dict):
        # arch: {
        #   'input': num
        #   'main' : [(num, act), ...]
        #   'output': (out_feature, act)
        #   ('pe': {'level': num, 'include_input': bool})
        #   ('triplane': {'reso': num, 'channel': num})
        #   ('trimlp': {'layer': num, 'channel': num, 'out': num})
        #}
        mlp_input_shape = 0
        if 'pe' in arch.keys():
            pe_embedder = self._build_pe(arch['pe'])
            
            pe_out_features = 2 * arch['pe']['level'] * 3 + 3
            mlp_input_shape = pe_out_features
        else:
            pe_embedder = None
            
        if 'triplane' in arch.keys():
            triplane_embedder = self._build_triplane(arch['triplane'])
            
            triplane_out_features = 3 * arch['triplane']['channel']
            mlp_input_shape += triplane_out_features
        else:
            triplane_embedder = None
                
        if 'trimlp' in arch.keys():
            trimlp_embedder = self._build_trimlp(arch['trimlp'])
            
            trimlp_out_features = 3 * arch['trimlp']['out']
            mlp_input_shape += trimlp_out_features
        else:
            trimlp_embedder = None
        
        mlp_input_shape = mlp_input_shape if mlp_input_shape != 0 else arch['input']
        
        # main net
        modules = []
        main = arch['main']
        in_features = mlp_input_shape
        for i in range(len(main)):
            out_features, activation = main[i]
            # weights
            modules += [nn.Linear(in_features, out_features), ActivationDict[activation]()]
            # next layer
            in_features = out_features
            
        # out
        out = arch['output']
        if out[1] == '':
            modules += [nn.Linear(in_features, out[0]), ]
        else:
            modules += [nn.Linear(in_features, out[0]), ActivationDict[out[1]]()]
            
        return pe_embedder, triplane_embedder, trimlp_embedder, nn.Sequential(*modules)
    
    def _build_pe(self, config: dict):
        level = config.get('level', 6)
        include_input = config.get('include_input', True)
        return PositionalEmbedding(level, include_input)
    
    def _build_triplane(self, config: dict):
        resolution = config.get('reso', 32)
        channel = config.get('channel', 2)
        return TriPlaneEmbedding(resolution, channel)
    
    def _build_trimlp(self, config: dict):
        layer = config.get('layer', 2)
        channel = config.get('channel', 32)
        out = config.get('channel', 2)
        return TriMLPEmebedding(layer, channel, out)
        
    def forward(self, x):        
        encoding = []
        
        if self.pe is not None:
            encoding.append(self.pe(x))
        if self.triplane is not None:
            encoding.append(self.triplane(x))
        if self.trimlp is not None:
            encoding.append(self.trimlp(x))
            
        if len(encoding) == 0:
            input = x
        elif len(encoding) == 1:
            input = encoding[0]
        else:
            input = torch.cat(encoding, dim=-1)
            
        return self.net(input)
        
    def size(self):
        return network_parameters(self.net) + network_parameters(self.pe) + network_parameters(self.triplane) + network_parameters(self.trimlp)

    def cuda(self, device = None):
        if self.pe is not None:
            self.pe = self.pe.cuda()
        if self.triplane is not None:
            self.triplane = self.triplane.cuda()
        if self.trimlp is not None:
            self.trimlp = self.trimlp.cuda()
        return super(StandardNet, self).cuda(device)
    
    def cpu(self):
        if self.pe is not None:
            self.pe = self.pe.cpu()
        if self.triplane is not None:
            self.triplane = self.triplane.cpu()
        if self.trimlp is not None:
            self.trimlp = self.trimlp.cpu()
        return super(StandardNet, self).cpu()

# RNN structure
class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_num_layers=2):
        super(StackLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.net = self._build_net()
        
    def _build_net(self):
        net = []
        for _ in range(self.lstm_num_layers):
            layer = nn.LSTMCell(self.input_size, self.hidden_size)
            net.append(layer)
        net = nn.ModuleList(net)
        
        return net
    
    def __call__(self, inputs, prev_h, prev_c):
        net = self.net
        next_h, next_c = [], []
        x = inputs
        for i in range(self.lstm_num_layers):
            cur_h, cur_c = net[i](x, (prev_h[i], prev_c[i]))
            next_h.append(cur_h)
            next_c.append(cur_c)
            x = next_h[-1]
        
        return next_h, next_c

class Controller(nn.Module):
    def __init__(self, 
                 num_ops,
                 child_max_num_layers=6,
                 lstm_size=32,
                 lstm_num_layers=2,
                 temperature=5,
                 tanh_constant=2.5,
                 device='gpu'
                 ):
        super(Controller, self).__init__()
        self.child_max_num_layers = child_max_num_layers
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        
        self.num_ops = num_ops  # include an end flag
        self.search_end_flag = num_ops - 1
        
        self.temperature = temperature
        self.tanh_constant = tanh_constant

        self.device = device
        
        self.net = self._build_net()
        
        self.op_init = torch.LongTensor([0])
        
    def _build_net(self):
        net = {}
        net['lstm'] = StackLSTM(self.lstm_size, self.lstm_size, self.lstm_num_layers)
        net['op_fc'] = nn.Linear(self.lstm_size, self.num_ops)
        # inputs will be encoded as a implicit vector but not a single value
        # we need its a learnable vector
        net['op_emb_lookup'] = nn.Embedding(self.num_ops, self.lstm_size)
        net = nn.ModuleDict(net)
        return net
    
    def _op_sample(self, args):
        net = self.net
        inputs, prev_h, prev_c, arc_seq, log_probs, entropys = args 
        # predict next layer class
        next_h, next_c = net['lstm'](inputs, prev_h, prev_c)    # [lstm_dim], [lstm_dim]
        prev_h, prev_c = next_h, next_c
        logit = net['op_fc'](next_h[-1])    # [num_ops]
        
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        
        # select a layer id
        prob = F.softmax(logit, dim=1)
        op_id = torch.multinomial(prob, 1)
        op_id = op_id[0]
        
        # the next inputs
        inputs = net['op_emb_lookup'](op_id.long())
        
        # calculate each step entropy 
        # each step needs an entropy
        log_prob = F.cross_entropy(prob, op_id)    
        entropy = log_prob * torch.exp(-log_prob)
        
        if self.device == 'gpu':
            op = op_id.cpu()
        op = int(op.data.numpy())
        arc_seq.append(op)
        log_probs.append(log_prob)
        entropys.append(entropy)
        
        return inputs, prev_h, prev_c, arc_seq, log_probs, entropys
    
    def net_sample(self):
        sequence = []
        entropys = []
        log_probs = []
        
        if self.device == 'gpu': # check whether gpu is available or not
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: device = 'cpu'
        # move model to gpu
        if self.device == 'gpu': # check whether gpu is available or not
            self.net.to(device) # move net to gpu
            
        # init inputs and states
        # init prev cell states to zeros for each layer of the lstm
        prev_c = [torch.zeros((1, self.lstm_size),device=device) for _ in range(self.lstm_num_layers)]
        # init prev hidden states to zeros for each layer of the lstm
        prev_h = [torch.zeros((1, self.lstm_size),device=device) for _ in range(self.lstm_num_layers)]
        # inputs
        if self.device == 'gpu':
            self.op_init = self.op_init.cuda()
        inputs = self.net['op_emb_lookup'](self.op_init)
        if self.device == 'gpu': # check whether gpu is available or not
            inputs = inputs.cuda()
        
        # sample an arch
        for _ in range(self.child_max_num_layers):
            arg_op_sample = [inputs, prev_h, prev_c, sequence, log_probs, entropys]
            returns_op_sample = self._op_sample(arg_op_sample)
            inputs, prev_h, prev_c, sequence, log_probs, entropys = returns_op_sample
            if sequence[-1] == self.search_end_flag: # the last class represents End signal
                break
        
        # sequence must end with end flag
        if sequence[-1] != self.search_end_flag:
            sequence.append(self.search_end_flag)

        # generate sample arch
        # cal sample entropy
        entropys = torch.stack(entropys)
        sample_entropy = torch.sum(entropys)

        # cal sample log_probs
        log_probs = torch.stack(log_probs)
        sample_log_prob = torch.sum(log_probs)
        
        return sequence, sample_entropy, sample_log_prob
 
if __name__ == '__main__':
    arch= {
       'input': 3,
       'main' : [(32, 'relu'), (32, 'relu'), (64, 'swish')],
       'output': (1, 'sigmoid'),
       'pe': {'level': 6, 'include_input': True},
       'triplane': {'reso': 32, 'channel': 2}
    }
    net = StandardNet(arch)
    print(net)
    # print(net.size())
    
    x = torch.rand(240, 3)  # 10, 24
    # print(x)
    y = batchify_run_network(net, x)
    print(y[:10])
    
    mask = torch.rand(10, 24) > 0.5
    print(mask[:, 0])

    def masked_scatter(mask, x):
        B, K = mask.size()
        if x.dim() == 1:
            return x.new_zeros(B, K).masked_scatter(mask, x)
        return x.new_zeros(B, K, x.size(-1)).masked_scatter(
            mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)
    
    masked_result = masked_scatter(mask, y)
    print(masked_result[:, 0, :])
        