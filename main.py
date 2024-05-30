from uuid import uuid4
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import toml


def get_cifar10(batch_size=16):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    return trainloader


def log_batch(dataloader, figure_title="", filename=str(uuid4())):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    batch_size = len(images)
    num_columns = 4
    num_rows = int(batch_size / num_columns) + 1
    fig = plt.figure(figsize=(2 * 2 * num_columns, 2 * num_rows))
    fig.suptitle(figure_title)
    for idx in range(batch_size):
        ax = fig.add_subplot(num_rows, 2 * num_columns, 2 * idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx].numpy().transpose((1, 2, 0)))
    plt.savefig(f'logs/{filename}.png')


class NGramModel(nn.Module):
    def __init__(self, n, hidden_size):
        super(NGramModel, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(n * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def generate_image(self):
        self.eval()
        input_size = 3  # RGB channels
        img_size = 32  # Image size (32x32)
        total_pixels = img_size * img_size
        
        # Initialize the first n-1 pixels with random values
        random_init_pixel = torch.rand((self.n - 1, input_size))
        
        ngram = torch.zeros((self.n, input_size))
        ngram[-(self.n-1):] = random_init_pixel
        
        output = torch.zeros((input_size, total_pixels))
        
        print('Generating an image...')
        for i in tqdm(range(total_pixels)):
            flat_ngram = ngram.view(-1)  # Flatten ngram to pass into the model
            next_pixel = self.forward(flat_ngram)
            output[:, i] = next_pixel
            
            # Update n-gram
            if i < total_pixels - 1:
                ngram[:-1] = ngram[1:].clone()
                ngram[-1] = next_pixel
        
        output_image = output.view(1, input_size, img_size, img_size)
        return output_image


def train_ngram(dataloader, model, optimizer, criterion, epochs=3):
    print("Training the n-gram model...")
    
    model.train()
    n = model.n
    
    for epoch in range(epochs):
        for images, _ in tqdm(dataloader):
            images = images.view(images.shape[0], 3, -1).permute(0, 2, 1)  # (batch_size, total_pixels, 3)
            padded_images = torch.cat([torch.zeros(images.shape[0], n-1, 3), images, torch.zeros(images.shape[0], n, 3)], dim=1)
            
            for pixel_idx in range(images.shape[1]):
                ngram = padded_images[:, pixel_idx:pixel_idx+n].reshape(images.shape[0], -1)  # (batch_size, n*3)
                gt_pixel = padded_images[:, pixel_idx+n]  # (batch_size, 3)
                prediction = model(ngram)
                loss = criterion(prediction, gt_pixel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    print("Done!")
    
    return model


if __name__ == '__main__':
    config = toml.load("config.toml")
    trainloader = get_cifar10(config["batch_size"])
    
    ngram_model = NGramModel(n=10, hidden_size=128)
    
    # Sanity check; Generate image with untrained model
    # generated_image = ngram_model.generate_image()
    # generated_image_np = generated_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # plt.clf()
    # plt.imshow((generated_image_np * 255).astype(int))
    # plt.show()
    
    # Train ngram
    optimizer = optim.Adam(ngram_model.parameters())
    criterion = nn.MSELoss()
    train_ngram(trainloader, ngram_model, optimizer, criterion)
    
    # Generate and visualize an image with the trained model
    generated_image = ngram_model.generate_image()
    generated_image_np = generated_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    plt.clf()
    plt.imshow((generated_image_np * 255).astype(int))
    plt.show()
