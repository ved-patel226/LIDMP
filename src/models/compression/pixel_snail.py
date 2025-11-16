import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------
# Helper functions
# --------------------


def down_shift(x):
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]


def right_shift(x):
    return F.pad(x, (1, 0))[:, :, :, :-1]


def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))


# --------------------
# Model components
# --------------------


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)


class DownShiftedConv2d(Conv2d):
    def forward(self, x):
        Hk, Wk = self.kernel_size
        x = F.pad(x, ((Wk - 1) // 2, (Wk - 1) // 2, Hk - 1, 0))
        return super().forward(x)


class DownRightShiftedConv2d(Conv2d):
    def forward(self, x):
        Hk, Wk = self.kernel_size
        x = F.pad(x, (Wk - 1, 0, Hk - 1, 0))
        return super().forward(x)


class GatedResidualLayer(nn.Module):
    def __init__(
        self,
        conv,
        n_channels,
        kernel_size,
        drop_rate=0,
        shortcut_channels=None,
        n_cond_classes=None,
        relu_fn=concat_elu,
    ):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2 * n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2 * shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2 * n_channels, 2 * n_channels, kernel_size)
        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2 * n_channels)

    def forward(self, x, a=None, h=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, "dropout"):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if h is not None:
            c2 += self.proj_h(h)[:, :, None, None]
        a, b = c2.chunk(2, 1)
        out = x + a * torch.sigmoid(b)
        return out


def causal_attention(k, q, v, mask, nh, drop_rate, training):
    B, dq, H, W = q.shape
    _, dv, _, _ = v.shape

    flat_q = q.reshape(B, nh, dq // nh, H, W).flatten(3) * (dq // nh) ** -0.5
    flat_k = k.reshape(B, nh, dq // nh, H, W).flatten(3)
    flat_v = v.reshape(B, nh, dv // nh, H, W).flatten(3)

    logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
    logits = F.dropout(logits, p=drop_rate, training=training, inplace=True)
    logits = logits.masked_fill(mask == 0, -1e10)
    weights = F.softmax(logits, -1)

    attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
    attn_out = attn_out.transpose(2, 3)
    return attn_out.reshape(B, -1, H, W)


class AttentionGatedResidualBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        n_background_ch,
        n_res_layers,
        n_cond_classes,
        drop_rate,
        nh,
        dq,
        dv,
        attn_drop_rate,
    ):
        super().__init__()
        self.nh = nh
        self.dq = dq
        self.dv = dv
        self.attn_drop_rate = attn_drop_rate

        self.input_gated_resnet = nn.ModuleList(
            [
                GatedResidualLayer(
                    DownRightShiftedConv2d,
                    n_channels,
                    (2, 2),
                    drop_rate,
                    None,
                    n_cond_classes,
                )
                for _ in range(n_res_layers)
            ]
        )
        self.in_proj_kv = nn.Sequential(
            GatedResidualLayer(
                Conv2d,
                2 * n_channels + n_background_ch,
                1,
                drop_rate,
                None,
                n_cond_classes,
            ),
            Conv2d(2 * n_channels + n_background_ch, dq + dv, 1),
        )
        self.in_proj_q = nn.Sequential(
            GatedResidualLayer(
                Conv2d, n_channels + n_background_ch, 1, drop_rate, None, n_cond_classes
            ),
            Conv2d(n_channels + n_background_ch, dq, 1),
        )
        self.out_proj = GatedResidualLayer(
            Conv2d, n_channels, 1, drop_rate, dv, n_cond_classes
        )

    def forward(self, x, background, attn_mask, h=None):
        ul = x
        for m in self.input_gated_resnet:
            ul = m(ul, h=h)

        kv = self.in_proj_kv(torch.cat([x, ul, background], 1))
        k, v = kv.split([self.dq, self.dv], 1)
        q = self.in_proj_q(torch.cat([ul, background], 1))
        attn_out = causal_attention(
            k, q, v, attn_mask, self.nh, self.attn_drop_rate, self.training
        )
        return self.out_proj(ul, attn_out)
