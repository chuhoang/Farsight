"""Test-time clip split of CSCI (= CAL data/datasets MEVID._recombination_for_testset), on frame indices."""
import numpy as np


def recombine(n, seq_len=8, stride=4):
    """n frames -> list of clips (seq_len frame indices each); ceil(n / seq_len) clips, covering every frame."""
    clips, span = [], seq_len * stride
    for i in range(n // span):
        for j in range(stride):
            clips.append(list(range(i * span + j, (i + 1) * span, stride)))
    if n % span:
        base = n // span * span
        s = (n % span) // seq_len          # reduced stride for the remainder
        for i in range(s):
            clips.append(list(range(base + i, base + seq_len * s, s)))
        if n % seq_len:
            c = list(range(n // seq_len * seq_len, n))
            while len(c) < seq_len:        # pad by cycling the remainder
                c += c[:seq_len - len(c)]
            clips.append(c)
    assert len(clips) == -(-n // seq_len)
    return clips


def select(clips, max_clips):
    """Keep max_clips clips spread evenly over the track (plan B3); 0 = all."""
    if not max_clips or len(clips) <= max_clips:
        return clips
    return [clips[i] for i in np.linspace(0, len(clips) - 1, max_clips).round().astype(int)]


def model_clips(n, seq_len=8, stride=4, max_clips=8):
    """Model input per clip: every 2nd frame of each 8-frame clip (VID dataset clip[::2]) -> T=4."""
    return [c[::2] for c in select(recombine(n, seq_len, stride), max_clips)]
