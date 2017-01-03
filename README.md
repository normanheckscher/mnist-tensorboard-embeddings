# mnist-tensorboard-embeddings
[![Build Status](https://travis-ci.org/normanheckscher/mnist-tensorboard-embeddings.svg?branch=master)](https://travis-ci.org/normanheckscher/mnist-tensorboard-embeddings)

TensorBoard is a suite of web applications for inspecting and understanding your
TensorFlow runs and graphs.

Mostly reused code from the official TensorFlow source tree r0.12

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Sample output

Run the files from within their directories to generate the embeddings and visualisation.

```

```

Once you have event files, run TensorBoard and provide the log directory. If
you're using a precompiled TensorFlow package (e.g. you installed via pip), run:

```
tensorboard --logdir=path/to/logs
```

This should print that TensorBoard has started. Next, connect to http://localhost:6006.

TensorBoard requires a `logdir` to read logs from. For info on configuring
TensorBoard, run `tensorboard --help`.

TensorBoard can be used in Google Chrome or Firefox. Other browsers might
work, but there may be bugs or performance issues.

# Contribution
Your comments (issues) and PRs are always welcome.
