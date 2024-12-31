import torch

names = open('names.txt').read().splitlines()

arr = torch.zeros((27, 27), dtype=torch.int32)

chars = list('abcdefghijklmnopqrstuvwxyz')
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {}
for k, v in stoi.items():
    itos[v] = k

for name in names:
    chs = ['.'] + list(name) + ['.'] # '.' represents a start/end character of a name
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        arr[idx1][idx2] += 1

P = arr.float()
P = P / P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(30):
    out = []
    idx = 0
    while True:

        p = P[idx]

        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if idx == 0:
            break
        out.append(itos[idx])
    print(''.join(out))
