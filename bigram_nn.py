import torch

names = open('names.txt').read().splitlines()

chars = list('abcdefghijklmnopqrstuvwxyz')
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {}
for k, v in stoi.items():
    itos[v] = k

g = torch.Generator().manual_seed(2147483647)

# GOAL: maximize likelihood of the data with respect to model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)

# log_likelihood = 0.0
# n = 0

# # for name in names:
# for name in names[:3]:
#     chs = ['.'] + list(name) + ['.'] # '.' represents a start/end character of a name
#     for ch1, ch2 in zip(chs, chs[1:]):
#         idx1 = stoi[ch1]
#         idx2 = stoi[ch2]
#         prob = P[idx1, idx2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob # log(a*b*c) = log(a) + log(b) + log(c)
#         n += 1

# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')

# create a training set of the bigrams (x, y)
xs, ys = [], []
# for name in names:
for name in names:
    chs = ['.'] + list(name) + ['.'] # '.' represents a start/end character of a name
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)
xs = torch.tensor(xs) 
ys = torch.tensor(ys)
num = xs.nelement()

W = torch.randn((27, 27), generator=g, requires_grad=True)

import torch.nn.functional as F
# gradient descent
for i in range(50):
    # forward pass
    # Convert tensor of nums to one_hot representation
    # Example: ana --> [0, 1, 14, 1] --> [[1, 0, ..., 0], [0, 1, 0, 0, 0, ..., 0], [0, ..., 0, 1, 0, ..., 0], [0, 1, 0, 0, ..., 0]]
    # shape is [4,27]
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W # @ is pytorch notation for matrix multiplication ### logits = log-counts
    # print(logits) 
    counts = logits.exp() # equivalent to arr
    probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
    # lines 93 and 94 are a softmax
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean() # This is the averaged negative log likelihood in a single line
    # The + above is a regularization
    # print(loss.item())

    # backward pass
    W.grad = None # set gradients back to zero
    loss.backward() # backward propagation

    # update
    W.data += -50 * W.grad

for i in range(5):
    out = []
    idx = 0
    while True:
        xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))