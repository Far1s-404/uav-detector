import torch
import time
from model import DroneDetector
from dataset import DroneDataset, DataLoader, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

def get_grid_cell(x, y, width, height, class_id):
    grid_x = int(x * 40)
    grid_y = int(y * 40)

    return grid_x, grid_y, x, y, width, height, class_id


# New loss function
def detection_loss(output, target):

    # Box coordinates
    box_loss = torch.nn.functional.mse_loss(
        output[:, 0:4],
        target[:, 0:4]
    )

    # Objectness
    objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output[:, 4],
        target[:, 4]
    )

    # Classes
    class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output[:, 5:10],
        target[:, 5:10]
    )

    loss = box_loss + objectness_loss + class_loss

    return loss


dataset = DroneDataset()

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

model = DroneDetector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adam:the optimization algorithm., lr = learning rate , model parameters = parameters to update

loss_function = detection_loss

start_time = time.time()


# Training loop
for epoch in range(10):

    total_loss = 0

    for batch_index, (images, labels) in enumerate(dataloader):

        images = images.to(device)

        output = model(images)

        target = torch.zeros(images.size(0), 10, 40, 40).to(device)

        for image_index in range(len(labels)):

            for item in labels[image_index]:

                class_id, x, y, width, height = item

                grid_x, grid_y, x, y, width, height, class_id = get_grid_cell(
                    x, y, width, height, class_id
                )

                target[image_index, 0, grid_y, grid_x] = x
                target[image_index, 1, grid_y, grid_x] = y
                target[image_index, 2, grid_y, grid_x] = width
                target[image_index, 3, grid_y, grid_x] = height
                target[image_index, 4, grid_y, grid_x] = 1

                target[image_index, 5 + int(class_id), grid_y, grid_x] = 1

        loss = loss_function(output, target)

        optimizer.zero_grad() # clear old gradients
        loss.backward()       # calculate how each weight contributed to the error
        optimizer.step()      # update the weights

        total_loss += loss.item()

        if batch_index % 100 == 0:
            print("Batch:", batch_index, "Loss:", loss.item())

    print("Epoch:", epoch + 1, "Loss:", total_loss / len(dataloader))


end_time = time.time()
training_time = end_time - start_time

print("Training time:", training_time / 60, "minutes")

torch.save(model.state_dict(), "drone_detector.pth")
print("Model saved as drone_detector.pth")
