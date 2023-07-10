import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = retinanet_resnet50_fpn(weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    image = read_image("images/pedestrian_night.jpg")
    transform = torchvision.transforms.Lambda(lambda x: x[:3])
    image = transform(image)
    norm_image = torch.div(image, 255)
    images = [norm_image]
    predictions = model(images)
    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]
    ped_boxes = []
    ped_scores = []
    for b, l, s in zip(boxes, labels, scores):
        if l == 1 and s > 0.5:
            ped_boxes.append(b)
            ped_scores.append(str(s.item())[0:4])
            print(f"Pedestrians score: {s:.3f}.")
    boxes_image = draw_bounding_boxes(image, torch.stack(ped_boxes),
                                      labels = ped_scores,
                                      font = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
                                      font_size = 20,
                                      colors = "red",
                                      width = 2)
    plt.imshow(boxes_image.permute(1, 2, 0))
    plt.show()