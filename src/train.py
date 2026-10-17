"""Train the mask / no-mask classifier with MobileNetV2 transfer learning.

Usage:
    python -m src.train --data-dir data/train --epochs 10 --batch-size 32
    python -m src.train --data-dir data/train --val-dir data/val --epochs 15
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from src.utils import save_class_indices


def parse_args():
    p = argparse.ArgumentParser(description="Train face-mask classifier")
    p.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    p.add_argument("--val-dir", default=None,
                   help="Optional separate validation dir. Default: validation_split on data-dir.")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=cfg.IMG_SIZE[0])
    p.add_argument("--output", default=str(cfg.MODEL_PATH))
    p.add_argument("--labels", default=str(cfg.LABELS_PATH))
    p.add_argument("--curves", default=str(cfg.CURVES_PATH))
    return p.parse_args()


def main():
    args = parse_args()  # parse first so --help works without TensorFlow
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(
            f"Dataset not found: {data_dir}\n"
            f"Expected layout: {data_dir}/with_mask/*.jpg and "
            f"{data_dir}/without_mask/*.jpg\nSee data/README.md."
        )
    if args.val_dir and not Path(args.val_dir).is_dir():
        raise SystemExit(f"Validation dir not found: {args.val_dir}")

    # Heavy imports kept inside main so `--help` stays import-light.
    import matplotlib

    matplotlib.use("Agg")  # headless-safe; original plt.show() crashed without display
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = (args.img_size, args.img_size)

    print("Starting model training...")
    print(f"  train dir : {data_dir}")

    if args.val_dir:
        val_dir = Path(args.val_dir)
        train_gen_cfg = dict(rescale=1.0 / 255)
        train_flow = ImageDataGenerator(**train_gen_cfg).flow_from_directory(
            str(data_dir), target_size=img_size, batch_size=args.batch_size,
            class_mode="binary",
        )
        val_flow = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
            str(val_dir), target_size=img_size, batch_size=args.batch_size,
            class_mode="binary",
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=cfg.VALIDATION_SPLIT,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
        )
        common = dict(
            target_size=img_size, batch_size=args.batch_size,
            class_mode="binary", seed=cfg.SEED,
        )
        train_flow = datagen.flow_from_directory(
            str(data_dir), subset="training", **common)
        val_flow = datagen.flow_from_directory(
            str(data_dir), subset="validation", **common)
        print(f"  validation split: {cfg.VALIDATION_SPLIT}")

    print(f"  class indices: {train_flow.class_indices}")
    print(f"    -> prob < 0.5 means '{min(train_flow.class_indices, key=train_flow.class_indices.get)}', "
          f">= 0.5 means '{max(train_flow.class_indices, key=train_flow.class_indices.get)}'")

    base_model = MobileNetV2(weights="imagenet", include_top=False,
                              input_shape=(*img_size, 3))
    base_model.trainable = False  # transfer learning: keep ImageNet features fixed

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(train_flow, validation_data=val_flow, epochs=args.epochs)

    # --- Plots (saved to file; original plt.show() blocked headless runs) ---
    curves_path = Path(args.curves)
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train acc")
    axes[0].plot(history.history["val_accuracy"], label="val acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train loss")
    axes[1].plot(history.history["val_loss"], label="val loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(curves_path)
    print(f"Training curves saved to {curves_path}")

    # --- Save model + label mapping ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    save_class_indices(train_flow.class_indices, args.labels)
    print(f"Model saved to {out_path}")
    print(f"Class indices saved to {args.labels} -> "
          f"{json.dumps(train_flow.class_indices)}")

    # Quick final-epoch summary
    acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    print(f"Done. train_acc={acc:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
