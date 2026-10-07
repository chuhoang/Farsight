"""Video reading. PyAV (FFmpeg); frames as BGR uint8, yielded in batches."""
import av
import numpy as np


def read_frames(path, batch=8, stride=1):
    """Yield (start_frame_idx, [BGR frames]) batches."""
    buf, start, i = [], 0, 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            if i % stride == 0:
                if not buf:
                    start = i
                buf.append(f.to_ndarray(format="bgr24"))
                if len(buf) == batch:
                    yield start, buf
                    buf = []
            i += 1
    if buf:
        yield start, buf


def video_fps(path, default=25.0):
    """Average frame rate from the container (default when the stream doesn't say)."""
    with av.open(str(path)) as c:
        r = c.streams.video[0].average_rate
    return float(r) if r else default


def crop(frame, box, pad=0.0):
    """Crop x1y1x2y2 box with relative padding, clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    return np.ascontiguousarray(frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)])
