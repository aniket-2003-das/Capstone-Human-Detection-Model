import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class HumanDetector(nn.Module):
    def __init__(self):
        super(HumanDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self._initialize_fc_layer()

    def _initialize_fc_layer(self):
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = self.forward_conv_layers(dummy_input)
        output_size = output.numel()
        self.fc1 = nn.Linear(output_size, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward_conv_layers(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        return x

    def forward(self, x):
        x = self.forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HumanDetectionSystem:
    def __init__(self):
        self.model = HumanDetector()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} not found.")
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        return image

    def detect_human(self, image_path):
        image = self.load_image(image_path)
        image = image.to(self.device)
        output = self.model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        return predicted.item()

    def draw_rectangle(self, image_path, x, y, w, h):
        image = cv2.imread(image_path)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('output_image.jpg', image)



# Prem Human detection system
human_detection_system = HumanDetectionSystem()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(human_detection_system.model.parameters(), lr=0.001)



image_path = 'test.png'
for epoch in range(10):
    image = human_detection_system.load_image(image_path)
    label = torch.tensor([1]).to(human_detection_system.device)
    optimizer.zero_grad()
    output = human_detection_system.model(image.unsqueeze(0))
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



# Test non Prem Model(same image)
image_path2 = 'test.png'
predicted = human_detection_system.detect_human(image_path)
print(f'Predicted: {predicted}')  # 0 for non-human, 1 for human
if predicted == 1:
    human_detection_system.draw_rectangle(image_path, 10, 10, 100, 100)
image = cv2.imread('output_image.jpg')
cv2.imshow('Human Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
