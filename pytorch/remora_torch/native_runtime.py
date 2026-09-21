"""Native tensor operations embedded in generated modules; parameters stay live."""
import torch.nn.functional as _F


class _NativeOps:
    def __init__(self, device):
        self.device = device

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def conv(self, x, p, stride, pad, activation, shape):
        x = self.tensor(x)
        p = {k: self.tensor(v) for k, v in p.items()}
        w, bias = p['w'], p['bias']
        if 'variance' in p:
            denom = torch.sqrt(p['variance'] + 1e-5)
            w = w * (p['scale'] / denom).reshape(-1, 1, 1, 1)
            bias = bias - (p['scale'] * p['mean']) / denom
        # Do not cache folded weights: callers can mutate parameters or train.
        out = _F.conv2d(x.unsqueeze(0), w, stride=stride, padding=pad)[0]
        out = out + bias[:, None, None]
        if activation == 'mish':
            out = out * torch.tanh(torch.log(torch.exp(out) + 1))
        elif activation == 'leaky':
            out = torch.where(out <= out * .1, out * .1, out)
        return out

    def raw_conv(self, x, w, stride, pad, shape):
        return _F.conv2d(self.tensor(x).unsqueeze(0), self.tensor(w), stride=stride, padding=pad)[0].flatten(1)

    def spp(self, x):
        x = self.tensor(x)
        def pool(v):
            return _F.max_pool2d(_F.pad(v, (2, 2, 2, 2), value=-1e30), 5, stride=1)
        p5 = pool(x)
        p9 = pool(p5)
        p13 = pool(p9)
        return torch.cat((p13, p9, p5, x), dim=0)

    def upsample(self, x, scale):
        # reshape/expand preserves nearest-neighbor replication exactly.
        x = self.tensor(x)
        c, h, w = x.shape
        return x.reshape(c, h, 1, w, 1).expand(c, h, scale, w, scale).reshape(c, h*scale, w*scale)

    def add(self, x, y):
        return self.tensor(x) + self.tensor(y)

    def cat(self, xs, axis):
        return torch.cat([self.tensor(x) for x in xs], dim=axis)

    def decode(self, x, aw, ah, sxy, net, classes):
        x = self.tensor(x)
        s = x.shape[-1]
        z = x.reshape(3, classes+5, s, s)
        row = torch.arange(s, dtype=torch.float32, device=self.device).reshape(1, s, 1)
        col = row.transpose(1, 2)
        aw = self.tensor(aw).reshape(3, 1, 1)
        ah = self.tensor(ah).reshape(3, 1, 1)
        sigmoid = lambda v: 1 / (1 + 2.718281828459045 ** (-v))
        shift = -.5 * (sxy - 1)
        planes = [((col + (sigmoid(z[:, 0])*sxy + shift))/s)[:, None],
                  ((row + (sigmoid(z[:, 1])*sxy + shift))/s)[:, None],
                  (2.718281828459045 ** z[:, 2] * aw / net)[:, None],
                  (2.718281828459045 ** z[:, 3] * ah / net)[:, None],
                  sigmoid(z[:, 4:])]
        return torch.cat(planes, dim=1).flatten(2).transpose(1, 2).reshape(3*s*s, classes+5)
