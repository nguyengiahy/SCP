import os
import sys
#Insert current working directory to path variables => No relative path usage from Python 3.
sys.path.insert(0, os.getcwd())
import torch
import imageio
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTImageProcessor
from torch.utils.data import random_split
from src.utils.misc import get_device_available
class KTHDataset(Dataset):
    def __init__(self, root_dir, N):
        '''
        root_dir: directory of the dataset
        N: number of past frames
        '''
        self.root_dir = root_dir
        self.N = N
        self.samples = []
        #KTH video properties 
        #Room for customizations
        self.fps = 25

        #Init embedding model when create instance of dataset -> Avoid model retrieval loops
        model_name = 'google/vit-base-patch16-224'
        self.emb_model = ViTModel.from_pretrained(model_name)
        self.emb_processor = ViTImageProcessor.from_pretrained(model_name)
        self.device = get_device_available()
        self.emb_model.to(self.device)

        # Iterate through each category folder and collect video paths
        for category in os.listdir(root_dir):
            category_folder = os.path.join(root_dir, category)
            if os.path.isdir(category_folder):
                for video_file in os.listdir(category_folder):
                    video_path = os.path.join(category_folder, video_file)
                    # Get the total number of frames of the current video
                    num_frames = self.get_total_frames(video_path)
                    # Add each possible sequence in this video
                    for i in range(0, num_frames - self.fps, self.fps):
                        #Take one sample every one second in video
                        self.samples.append((video_path, i))              

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_frame = self.samples[idx]

        # Get N+1 frames of the video_path, starting from the "start_frame" index
        frames = self.read_frames(video_path, start_frame, self.N + 1)

        # Process each frame to have the shape [channels, width, height]
        processed_frames = [self.process_frame(frame) for frame in frames]
        
        # Stack N frames for input data
        data = torch.stack(processed_frames[:-1]).type(torch.float64)   # data = [N, channels, width, height]

        # Output data (label) is the ViT processed features of the last frame
        last_frame = processed_frames[-1]                # last_frame = [channels, width, height]
        inputs = self.emb_processor(images=last_frame, return_tensors='pt', dtype=torch.float64)
        pixel_values = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            label = self.emb_model(pixel_values)
            # Representation of the entire frame
            label = label.last_hidden_state.mean(dim=1)     # label = [1, 768]
            label = label.squeeze()                         # label = [768]

        return data, label

    def read_frames(self, video_path, start_frame, num_frames):
        frames = []
        try:
            reader = imageio.get_reader(video_path)

            # Skipping to the start frame
            for _ in range(start_frame):
                _ = reader.get_next_data()

            # Reading the required number of frames
            # distance shows how far would it be from one frame to another
            distance = self.fps // num_frames
            for i in range(self.fps):
                frame = reader.get_next_data()
                if i % distance == 0:
                    # If the current frame has the proper distance from the previous frame => Take the frame
                    frames.append(frame)
                if len(frames) == num_frames:
                    #Ensure that the number of frames is exact num_frames
                    break

        except Exception as e:
            print(f"Error reading frames from {video_path}: {e}")

        return frames

    def get_total_frames(self, video_path):
        num_frames = 0
        try:
            reader = imageio.get_reader(video_path)
            for _ in reader:
                num_frames += 1
        except Exception as e:
            print(f"Error counting frames in video file {video_path}: {e}")

        return num_frames

    def process_frame(self, frame):
        '''
        Process a frame to have the shape of [channels, width, height]
        Args:
            frame: the frame to be processed
        '''
        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.from_numpy(frame)

        # The frame is originally in (H, W, C) format => convert to (C, W, H)
        frame_tensor = frame_tensor.permute(2, 0, 1)

        return frame_tensor

def get_dataloader(dataset, batch_size, train_split=0.7, val_split=0.15, test_split=0.15, shuffle_train=True):
    """
    Create DataLoaders for training, validation, and testing.
    Args:
        dataset: the processed dataset. Each data instance is a tuple of (data, label).
        batch_size: Batch size for the DataLoaders.
        train_split: Proportion of data to use for training.
        val_split: Proportion of data to use for validation.
        test_split: Proportion of data to use for testing.
        shuffle_train: Whether to shuffle the training dataset.
    return:
        A tuple of DataLoaders (train_loader, val_loader, test_loader).
    """
    # Calculate the sizes of each split
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    test_size = dataset_size - (train_size + val_size)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader