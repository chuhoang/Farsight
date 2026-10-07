import torch

from farsight.modules.m3_encode.gait.biggait.model import get_body_valid


def _orig(mask, edge=5):   # OpenGait BigGait.get_body
    cm = torch.round(mask[..., 0]) - mask[..., 0].detach() + mask[..., 0]
    cond = (cm[:, :edge, :].sum(dim=(1, 2)) + cm[:, -edge:, :].sum(dim=(1, 2))) > cm.size(2) * edge
    mask[cond, :, :, 0] = mask[cond, :, :, 1]
    return mask[..., 0]


def test_same_as_original_without_padding():
    m = torch.softmax(torch.randn(8, 64, 32, 2), -1)
    assert torch.equal(get_body_valid(m.clone()), _orig(m.clone()))


def test_padded_background_channel_is_swapped():
    # channel 0 = background everywhere in a 12-column-wide valid band (rest padded = all zero), body in channel 1
    m = torch.zeros(1, 64, 32, 2)
    m[:, :, 10:22, 0] = 1.0                     # background over the whole valid width
    m[:, 20:60, 13:19, 0], m[:, 20:60, 13:19, 1] = 0.0, 1.0   # person in the middle, not touching top/bottom
    assert _orig(m.clone()).sum() > 300         # original keeps the background (cannot pass the full-width test)
    fixed = get_body_valid(m.clone())
    assert fixed[0, 30, 15] == 1 and fixed[0, 2, 11] == 0   # fixed: person in, background out
