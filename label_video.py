import numpy as np
import sys
import grpc
import time
import os
import subprocess

import tensorflow as tf
import decord
from decord import VideoReader
from decord import cpu

from google.cloud import datastore

cachedir = "/cache"

labels = ["gameplay", "character_select", "not-gameplay"]

def frames_to_tc(frames):
    """Pretty print a frame count to a timecode `01:23:45`"""
    # Assuming 60fps
    seconds = int(frames / 60)
    sec = seconds % 60
    minutes = int(seconds / 60)
    minn = minutes % 60
    hours = int(minutes / 60)
    return f"{hours:02d}:{minn:02d}:{sec:02d}"

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print("loading model")
model = tf.keras.models.load_model("/model/")
print("loaded")
def inferLocal(frameIdxs, imgs):
    """Returns np array pairing frame idxs to the % likelyhood that it is gameplay"""
    results = np.empty((len(imgs),4), dtype=np.uint32)
    imgs = tf.image.resize(imgs, [480, 720])
    start = time.perf_counter()
    prediction = model.predict(imgs)
    print("Prediction Time:", time.perf_counter() - start)
    for idx, frame in enumerate(frameIdxs):
        results[idx,0] = frame
        results[idx,1:4] = (softmax(prediction[idx]) * 100).astype(np.uint32)
    return results


def test_video(video_name):
    """loads the given video and feeds frames through the inference engine"""
    f = os.path.join(cachedir, os.path.basename(os.path.splitext(video_name)[0]))
    if os.path.isfile(f+".npy"):
        print(f"FOUND EXISTING CLASSIFICATIONS: {f}.npy")
        return np.load(f+".npy")

    vr = VideoReader(video_name, ctx=cpu(0))

    frames = len(vr)
    print("video frames:", frames)
    decord.bridge.set_bridge('tensorflow')

    # Assuming 60 fps
    sample_rate = 60
    images_per_batch = 32
    samples = int(frames / sample_rate)
    batches = int(samples / images_per_batch)

    persample = np.empty((batches*images_per_batch,4), dtype=np.uint32)

    for i in range(batches):
        print("batch", i, "of", batches)
        # Create a collection of frame indexes at each sample rate within the batch
        frameIdxs = [(x * sample_rate) + (i * images_per_batch * sample_rate) for x in range(32)]
        frames = vr.get_batch(frameIdxs)

        res = inferLocal(frameIdxs, frames)
        persample[i*images_per_batch:(i+1)*images_per_batch,:] = res

    print("saving to", f)
    np.save(f, persample)
    return persample

def findEdges(d, beta=0.05, idx=1, threshold=50):
    val = d[0][idx]
    active = False
    startFrame = None
    sections = []

    for i in range(len(d)-1):
        # Weighted average of scores
        n = d[i+1][idx]
        val = (beta * n) + (1-beta) * val
        frame = d[i+1][0]
        if active and val < threshold:
            # Falling edge
            sections.append((startFrame, frame))
            active = False
        
        if not active and val > threshold:
            # rising edge
            startFrame = frame
            active = True
    return sections

def download_video(videoID):
    print("Downloading video...")
    filePath = os.path.join(cachedir, f"{videoID}.mp4")
    cmd = " ".join([
        "youtube-dl",
        "-f 720p60",
        "-w",
        f"https://www.twitch.tv/videos/{videoID}",
        f"-o {filePath}",
    ])
    print(cmd)
    download = subprocess.run(cmd, capture_output=True, shell=True, check=True)
    print(download.stdout.decode("utf-8"))
    print(download.stderr.decode("utf-8"))
    return filePath

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify video id")
    vid_id = sys.argv[1]
    f = download_video(vid_id)
    d = test_video(f)
    edges = findEdges(d)
    print(f"Found {len(edges)} games:")
    for i, edge in enumerate(edges):
        print(f"\t{i:02d}: {frames_to_tc(edge[1]-edge[0])} Starting @:{frames_to_tc(edge[0])}")

