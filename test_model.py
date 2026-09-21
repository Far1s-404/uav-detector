import time
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import DroneDetector
from dataset import DroneDataset, collate_fn


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nTraining device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

BATCH_SIZE = 4

EPOCHS = 200

LEARNING_RATE = 0.0005

NUM_WORKERS = 4

OUTPUT_DIR = Path(
    "D:/uav_detector/weights"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_DIR /
    "best_drone_detector.pth"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR /
    "last_drone_detector.pth"
)


# ============================================================
# TARGET BUILDER
# ============================================================

def build_targets(labels, batch_size, grid_h, grid_w, device):

    target = torch.zeros(
        batch_size,
        5 + NUM_CLASSES,
        grid_h,
        grid_w,
        device=device
    )

    for batch_index in range(len(labels)):

        for item in labels[batch_index]:

            class_id = int(item[0])

            x = float(item[1])
            y = float(item[2])
            width = float(item[3])
            height = float(item[4])

            # ------------------------------------------------
            # Clamp normalized coordinates
            # ------------------------------------------------

            x = min(max(x, 0.0), 0.999999)
            y = min(max(y, 0.0), 0.999999)

            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            # ------------------------------------------------
            # Find grid cell
            # ------------------------------------------------

            gx = min(
                int(x * grid_w),
                grid_w - 1
            )

            gy = min(
                int(y * grid_h),
                grid_h - 1
            )

            # ------------------------------------------------
            # Position INSIDE the cell
            # ------------------------------------------------

            cell_x = (
                x * grid_w
            ) - gx

            cell_y = (
                y * grid_h
            ) - gy

            # ------------------------------------------------
            # If multiple objects land in same cell,
            # keep the first one.
            #
            # This architecture supports one object per cell.
            # ------------------------------------------------

            if target[
                batch_index,
                4,
                gy,
                gx
            ] == 1:

                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            target[
                batch_index,
                0,
                gy,
                gx
            ] = cell_x

            target[
                batch_index,
                1,
                gy,
                gx
            ] = cell_y

            target[
                batch_index,
                2,
                gy,
                gx
            ] = width

            target[
                batch_index,
                3,
                gy,
                gx
            ] = height

            # ------------------------------------------------
            # Objectness
            # ------------------------------------------------

            target[
                batch_index,
                4,
                gy,
                gx
            ] = 1.0

            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            if 0 <= class_id < NUM_CLASSES:

                target[
                    batch_index,
                    5 + class_id,
                    gy,
                    gx
                ] = 1.0

    return target


# ============================================================
# LOSS
# ============================================================

def detection_loss(output, target):

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred_boxes = output[:, 0:4]

    pred_objectness = output[:, 4]

    pred_classes = output[:, 5:]

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target_boxes = target[:, 0:4]

    target_objectness = target[:, 4]

    target_classes = target[:, 5:]

    # --------------------------------------------------------
    # Object mask
    # --------------------------------------------------------

    object_mask = (
        target_objectness == 1
    )

    object_mask_float = object_mask.float()

    num_objects = (
        object_mask_float.sum()
        .clamp(min=1.0)
    )

    # ========================================================
    # BOX LOSS
    # ========================================================

    if object_mask.any():

        box_loss = F.smooth_l1_loss(
            pred_boxes,
            target_boxes,
            reduction="none"
        )

        # [B, 4, H, W]
        box_loss = (
            box_loss *
            object_mask_float.unsqueeze(1)
        ).sum()

        box_loss = (
            box_loss /
            (num_objects * 4.0)
        )

    else:

        box_loss = torch.tensor(
            0.0,
            device=output.device
        )

    # ========================================================
    # OBJECTNESS LOSS
    # ========================================================

    # Positive cells receive stronger weight.
    #
    # This is important because almost all 160x160 cells
    # contain background.
    # ========================================================

    positive_weight = 10.0
    negative_weight = 1.0

    objectness_weights = torch.where(
        target_objectness == 1,
        torch.full_like(
            target_objectness,
            positive_weight
        ),
        torch.full_like(
            target_objectness,
            negative_weight
        )
    )

    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness,
        target_objectness,
        weight=objectness_weights,
        reduction="mean"
    )

    # ========================================================
    # CLASS LOSS
    # ========================================================

    if object_mask.any():

        class_loss = F.binary_cross_entropy_with_logits(
            pred_classes,
            target_classes,
            reduction="none"
        )

        class_loss = (
            class_loss *
            object_mask_float.unsqueeze(1)
        ).sum()

        class_loss = (
            class_loss /
            (num_objects * NUM_CLASSES)
        )

    else:

        class_loss = torch.tensor(
            0.0,
            device=output.device
        )

    # ========================================================
    # TOTAL
    # ========================================================

    total_loss = (
        5.0 * box_loss
        + objectness_loss
        + class_loss
    )

    return total_loss, box_loss, objectness_loss, class_loss


# ============================================================
# TRAINING
# ============================================================

def main():

    print("\n======================================")
    print(" CUSTOM UAV DETECTOR TRAINING")
    print("======================================")

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = DroneDataset(
        split="train"
    )

    val_dataset = DroneDataset(
        split="val"
    )

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = DroneDetector(
        num_classes=NUM_CLASSES
    ).to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # --------------------------------------------------------
    # TRACK BEST MODEL
    # --------------------------------------------------------

    best_val_loss = float("inf")

    total_start_time = time.time()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_total = 0.0
        train_box = 0.0
        train_obj = 0.0
        train_cls = 0.0

        for batch_index, (
            images,
            labels
        ) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                output = model(images)

                batch_size, _, grid_h, grid_w = (
                    output.shape
                )

                target = build_targets(
                    labels,
                    batch_size,
                    grid_h,
                    grid_w,
                    device
                )

                loss, box_loss, obj_loss, cls_loss = (
                    detection_loss(
                        output,
                        target
                    )
                )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            train_total += loss.item()
            train_box += box_loss.item()
            train_obj += obj_loss.item()
            train_cls += cls_loss.item()

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if batch_index % 50 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        avg_train = (
            train_total /
            len(train_loader)
        )

        avg_train_box = (
            train_box /
            len(train_loader)
        )

        avg_train_obj = (
            train_obj /
            len(train_loader)
        )

        avg_train_cls = (
            train_cls /
            len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_total = 0.0

        val_box = 0.0
        val_obj = 0.0
        val_cls = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp
                ):

                    output = model(images)

                    batch_size, _, grid_h, grid_w = (
                        output.shape
                    )

                    target = build_targets(
                        labels,
                        batch_size,
                        grid_h,
                        grid_w,
                        device
                    )

                    loss, box_loss, obj_loss, cls_loss = (
                        detection_loss(
                            output,
                            target
                        )
                    )

                val_total += loss.item()
                val_box += box_loss.item()
                val_obj += obj_loss.item()
                val_cls += cls_loss.item()

        avg_val = (
            val_total /
            len(val_loader)
        )

        avg_val_box = (
            val_box /
            len(val_loader)
        )

        avg_val_obj = (
            val_obj /
            len(val_loader)
        )

        avg_val_cls = (
            val_cls /
            len(val_loader)
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        epoch_time = (
            time.time() -
            epoch_start
        )

        mins, secs = divmod(
            epoch_time,
            60
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print("\n" + "=" * 70)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Loss: {avg_train:.4f}"
        )

        print(
            f"  Box: {avg_train_box:.4f} | "
            f"Obj: {avg_train_obj:.4f} | "
            f"Cls: {avg_train_cls:.4f}"
        )

        print(
            f"Val Loss:   {avg_val:.4f}"
        )

        print(
            f"  Box: {avg_val_box:.4f} | "
            f"Obj: {avg_val_obj:.4f} | "
            f"Cls: {avg_val_cls:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

        print(
            f"Epoch Time: {int(mins)}m {int(secs)}s"
        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if avg_val < best_val_loss:

            best_val_loss = avg_val

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss
                },
                BEST_MODEL_PATH
            )

            print(
                f"\n[+] NEW BEST MODEL SAVED"
            )

            print(
                f"    Val Loss: {best_val_loss:.4f}"
            )

        # ====================================================
        # SAVE LAST
        # ====================================================

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val
            },
            LAST_MODEL_PATH
        )

        print("=" * 70 + "\n")

    # ========================================================
    # COMPLETE
    # ========================================================

    total_minutes = (
        time.time() -
        total_start_time
    ) / 60.0

    print(
        f"\nTraining complete!"
    )

    print(
        f"Total time: "
        f"{total_minutes:.2f} minutes"
    )

    print(
        f"Best model: "
        f"{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
