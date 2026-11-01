"""
PatchCore (Roth et al. 2022), written for the same data and config style as JADE.

  1. A frozen ImageNet CNN (model.backbone) gives feature maps of the chosen stages (layers).
  2. Every patch gets the average of its model.patch.neighbourhood x neighbourhood neighbours (local aggregation); all stages are brought to
     the grid of the finest one and every patch is squeezed to model.patch.embed_dim numbers.
  3. fit: the patches of ALL OK pictures are collected; a greedy k-center coreset (model.memory.coreset_ratio of them) is kept as the memory bank.
     The coreset keeps the patches that are far from all others first, so rare but normal details (like the embossing) are kept.
  4. score: a patch's error = the distance to its nearest patch in the memory bank. The error map is the patch errors enlarged to the picture
     size and blurred (model.blur_sigma). The score of a picture is the largest error in its map.
There is no training in the usual sense: nothing is learned by gradient steps.
"""
import time
import math
import torch
import torch.nn.functional as F
import torchvision


def gaussian_blur(m, sigma):
    """m (B, 1, H, W)."""
    if sigma <= 0:
        return m
    r = int(4 * sigma + 0.5)
    t = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    m = F.pad(m, (r, r, r, r), mode="reflect")
    m = F.conv2d(m, k.view(1, 1, 1, -1))
    return F.conv2d(m, k.view(1, 1, -1, 1))


class PatchCore:
    def __init__(self, cfg, device):
        mc = cfg["model"]
        bb = mc["backbone"]
        net = getattr(torchvision.models, bb["type"])(weights=None)
        net.load_state_dict(torch.load(bb["weights_path"], map_location="cpu"))
        for p in net.parameters():
            p.requires_grad = False
        self.net = net.to(device).eval()
        self.type = bb["type"]
        self.layers = list(bb["layers"])                           # stages 1..4 of the ResNet (stride 4, 8, 16, 32)
        if not self.layers or any(l not in (1, 2, 3, 4) for l in self.layers):
            raise ValueError("model.backbone.layers must be stages from 1 to 4, got %s" % self.layers)
        self.neigh = int(mc["patch"]["neighbourhood"])
        self.embed_dim = int(mc["patch"]["embed_dim"])
        self.sigma = float(mc.get("blur_sigma", 4))
        self.device = device
        self.bank = None

    def meta(self, size_hw):
        return {"backbone": self.type, "layers": self.layers, "neighbourhood": self.neigh, "embed_dim": self.embed_dim,
                "image_size": list(size_hw)}

    @torch.no_grad()
    def embed(self, x):
        """(B, 3, H, W) -> (B, h, w, embed_dim): one vector per patch of the finest chosen stage."""
        n = self.net
        y = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        got = {}
        for i, stage in enumerate([n.layer1, n.layer2, n.layer3, n.layer4], start=1):
            y = stage(y)
            if i in self.layers:
                got[i] = y
            if i >= max(self.layers):
                break
        maps = [got[i] for i in self.layers]
        maps = [F.avg_pool2d(m, self.neigh, stride=1, padding=self.neigh // 2) for m in maps]       # local neighbourhood
        h, w = max((m.shape[-2:] for m in maps), key=lambda s: s[0] * s[1])
        maps = [m if tuple(m.shape[-2:]) == (h, w) else F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False) for m in maps]
        B = x.shape[0]
        parts = []
        for m in maps:
            p = m.permute(0, 2, 3, 1).reshape(-1, 1, m.shape[1])                                   # (B*h*w, 1, C)
            parts.append(F.adaptive_avg_pool1d(p, self.embed_dim).squeeze(1))                      # (B*h*w, embed_dim)
        f = torch.cat(parts, 1).unsqueeze(1)                                                       # (B*h*w, 1, layers*embed_dim)
        f = F.adaptive_avg_pool1d(f, self.embed_dim).squeeze(1)                                    # (B*h*w, embed_dim)
        return f.reshape(B, h, w, -1)

    def set_bank(self, bank):
        self.bank = bank.to(self.device).float()

    @torch.no_grad()
    def nearest(self, f):
        """(B, h, w, D) -> (B, h, w): distance of every patch to its nearest patch in the memory bank."""
        B, h, w, D = f.shape
        q = f.reshape(-1, D)
        out = torch.empty(q.shape[0], device=q.device)
        for s in range(0, q.shape[0], 1024):
            out[s:s + 1024] = torch.cdist(q[s:s + 1024], self.bank).min(1).values
        return out.reshape(B, h, w)

    @torch.no_grad()
    def error_map(self, x, size_hw):
        """(B, 3, H, W) -> (B, H, W) error maps at the size of the picture."""
        s = self.nearest(self.embed(x))
        m = F.interpolate(s.unsqueeze(1), size=tuple(size_hw), mode="bilinear", align_corners=False)
        return gaussian_blur(m, self.sigma).squeeze(1)


def greedy_coreset(proj, n_select, seed):
    """Greedy k-center on the (randomly projected) patches: start from one random patch, then always add the patch that is farthest
    from all patches chosen so far. proj is (N, d) on the GPU. Returns the indices (CPU, long)."""
    N = proj.shape[0]
    sq = (proj * proj).sum(1)
    g = torch.Generator().manual_seed(seed)
    nxt = torch.randint(0, N, (1,), generator=g).to(proj.device).squeeze(0)
    min_d = torch.full((N,), float("inf"), device=proj.device)
    sel = torch.empty(n_select, dtype=torch.long, device=proj.device)
    t0 = time.time()
    step = max(1, n_select // 10)
    for it in range(n_select):
        sel[it] = nxt
        d = sq - 2.0 * torch.mv(proj, proj[nxt]) + sq[nxt]
        min_d = torch.minimum(min_d, d)
        nxt = torch.argmax(min_d)
        if (it + 1) % step == 0:
            print("[Fit] coreset %d / %d  (%.0f s)" % (it + 1, n_select, time.time() - t0), flush=True)
    return sel.cpu()


def random_projection(feats, dim, seed, device, chunk=100000):
    """(N, D) float16 on the CPU -> (N, dim) float32 on the GPU, by a fixed random matrix (only used to choose the coreset)."""
    g = torch.Generator().manual_seed(seed)
    R = (torch.randn(feats.shape[1], dim, generator=g) / math.sqrt(dim)).to(device)
    out = torch.empty((feats.shape[0], dim), device=device)
    for s in range(0, feats.shape[0], chunk):
        out[s:s + chunk] = feats[s:s + chunk].to(device).float() @ R
    return out
