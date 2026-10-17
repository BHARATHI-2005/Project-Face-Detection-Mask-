# Dataset layout
# ----------------
# Place training images like this (folder name = class name):
#
#   data/train/with_mask/*.jpg
#   data/train/without_mask/*.jpg
#
# The trainer uses validation_split=0.2 on data/train by default, so a
# separate data/val/ split is OPTIONAL. If you have one, use:
#
#   data/val/with_mask/*.jpg
#   data/val/without_mask/*.jpg
#
# and run:  python -m src.train --val-dir data/val
#
# NOTE: data/ is git-ignored (50+ MB of images don't belong in git).
# The dataset currently in use (1,376 real face images: 690 with_mask,
# 686 without_mask) was fetched with:
#
#   git clone --depth 1 https://github.com/Yousef-Hesham/COVID19-Face-Mask-Detector.git
#   copy dataset/with_mask    -> data/train/with_mask
#   copy dataset/without_mask -> data/train/without_mask
#
# (Prajna Bhandary's Face Mask Detection set, MIT-licensed tutorial repo.)
