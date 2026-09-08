import torch
import torch.nn.functional as F
import time
from pathlib import Path
from model import DroneDetector
from dataset import DroneDataset, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grid_cell(x, y, width, height, class_id, grid_w, grid_h):
    x = min(max(float(x), 0.0), 0.999999)
    y = min(max(float(y), 0.0), 0.999999)
    grid_x = int(x * grid_w)
    grid_y = int(y * grid_h)
    cell_x = (x * grid_w) - grid_x
    cell_y = (y * grid_h) - grid_y
    return grid_x, grid_y, cell_x, cell_y, width, height, class_id

def detection_loss(output, target):
    pred_boxes = output[:, 0:4]
    target_boxes = target[:, 0:4]
    object_mask = target[:, 4:5] 

    box_loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none")
    box_loss = (box_loss * object_mask).sum() / (object_mask.sum() * 4 + 1e-6)

    objectness_loss = F.binary_cross_entropy_with_logits(
        output[:, 4], target[:, 4], reduction="none"
    )
    objectness_weight = torch.where(
        target[:, 4] == 1,
        torch.tensor(10.0, device=target.device),
        torch.tensor(5.0, device=target.device),
    )
    objectness_loss = (objectness_loss * objectness_weight).mean()

    class_loss = F.binary_cross_entropy_with_logits(
        output[:, 5:10], target[:, 5:10], reduction="none"
    )
    class_loss = (class_loss * object_mask).sum() / (object_mask.sum() * 5 + 1e-6)

    return (5.0 * box_loss) + objectness_loss + class_loss

def main():
    print(f"Initializing training on: {device}")
    
    train_dataset = DroneDataset(split="train")
    val_dataset = DroneDataset(split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, 
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, 
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    model = DroneDetector(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # --- MARATHON RUN CONFIGURATION ---
    epochs = 50
    best_val_loss = float('inf')
    
    output_dir = Path("D:/uav_detector/weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_drone_detector.pth"

    print(f"Training Images: {len(train_dataset)} | Validation Images: {len(val_dataset)}")
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        for batch_index, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time() # Start the batch timer
            
            images = images.to(device)
            output = model(images)
            batch_size, _, grid_h, grid_w = output.shape
            target = torch.zeros(batch_size, 10, grid_h, grid_w, device=device)

            # --- CPU TARGET BUILDER ---
            for i in range(len(labels)):
                for item in labels[i]:
                    class_id, x, y, width, height = item
                    gx, gy, cx, cy, w, h, cid = get_grid_cell(x, y, width, height, class_id, grid_w, grid_h)
                    target[i, 0, gy, gx] = cx
                    target[i, 1, gy, gx] = cy
                    target[i, 2, gy, gx] = min(max(float(w), 0.0), 1.0)
                    target[i, 3, gy, gx] = min(max(float(h), 0.0), 1.0)
                    target[i, 4, gy, gx] = 1.0
                    target[i, 5 + int(cid), gy, gx] = 1.0
            # -------------------------------------------------------------

            loss = detection_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            batch_duration = time.time() - batch_start_time # Stop the batch timer

            if batch_index % 10 == 0:
                print(f"  -> Processing Batch [{batch_index}/{len(train_loader)}] | Current Loss: {loss.item():.4f} | Time: {batch_duration:.3f}s")

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                output = model(images)
                batch_size, _, grid_h, grid_w = output.shape
                target = torch.zeros(batch_size, 10, grid_h, grid_w, device=device)

                for i in range(len(labels)):
                    for item in labels[i]:
                        class_id, x, y, width, height = item
                        gx, gy, cx, cy, w, h, cid = get_grid_cell(x, y, width, height, class_id, grid_w, grid_h)
                        target[i, 0, gy, gx] = cx
                        target[i, 1, gy, gx] = cy
                        target[i, 2, gy, gx] = min(max(float(w), 0.0), 1.0)
                        target[i, 3, gy, gx] = min(max(float(h), 0.0), 1.0)
                        target[i, 4, gy, gx] = 1.0
                        target[i, 5 + int(cid), gy, gx] = 1.0

                v_loss = detection_loss(output, target)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_duration = time.time() - epoch_start_time
        mins, secs = divmod(epoch_duration, 60)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {int(mins)}m {int(secs)}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   [+] New best model saved! (Val Loss: {best_val_loss:.4f})")

    total_time = (time.time() - total_start_time) / 60
    print(f"\nTraining complete in {total_time:.2f} minutes.")
    print(f"Best weights saved to: {best_model_path}")

if __name__ == "__main__":
    main()
