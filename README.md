# Gameplay Classifier

## Training a valorant gameplay classifier.

I use nvidia's docker runtime to train.

You can run the training:
```
docker build . -f Dockerfile.train -t training:latest

docker run -it --rm --gpus all --mount type=bind,source="/path/to/training/data",target=/data -e TRAIN_MOUNT_POINT=/data training:latest
```

The training data required is a csv containing a path to the file and a label.

It is a simple single dense layer on top of a pre-trained resnet50.  We do 10 epochs training just our dense
layer and then do fine tuning.  This is exactly the method suggested by the tensorflow [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning) tutorial.  Based n the number I was seeing I would easily expect that I'm overfitting.

I trained using screenshots gathered using [Twitch Frame Capture](https://github.com/cconger/tw-fc) and manually labelled.  I gathered about 2000 labeled images and I specifically seeded it with images from tournaments so that it would be resilient to the UI of the alternate view.

## Annotating a video

Annotation is done by downloading a twitch vod, sampling a frame every second classifying it.  That dataframe is then softmaxed to find its probability of being gameplay.  I then create a simple weighted average of that percentage through time and I create "chunks" where the weighted average stays above a threshold.  The weighted average allows us to allow chunks to be resilient to occasional mis-categorized frames or pauses/cutaways.  The final output is a list of those chunks above the threshold.

```
docker build . -t annotate:latest

docker run -it --rm --gpus all --mount type=bind,source"/path/to/cache",target=/cache --ipc=host python label_video.py <VIDEO_ID>
```
