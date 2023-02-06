import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
#import cv2
#from imutils.video import count_frames
#import sys
from skvideo.io import FFmpegReader
from itertools import islice
from frame_transforms import FrameTransforms
from random import sample

import numpy as np
import pandas as pd
from collections import OrderedDict
#from vit_pytorch.vivit import ViT


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key) -> int:
        if key not in self.cache:
            return None

        # Move the key to the back of the cache
        self.cache.move_to_end(key)

        return self.cache[key]

    def put(self, key, value) -> None:
        if key in self.cache:
            # Update the value and move the key to the back of the cache
            self.cache[key] = value
            self.cache.move_to_end(key)
            
        else:
            if len(self.cache) == self.capacity:
                # Evict the least recently used key
                self.cache.popitem(last=False)
            # Add the new key to the cache and move it to the back of the cache
            self.cache[key] = value
            self.cache.move_to_end(key)

class VideoFrameSequencer(Dataset):
    def __init__(self, total_frames=256) -> None:
        super().__init__()
        self.frame_transform = FrameTransforms(frame_shape=512)
        self.total_frames = total_frames
        

    def get_frame_set(self, start_timestamp, start_frame, end_frame):
        pass
    
    def pad_frames(self, frame_seq:torch.Tensor) -> torch.Tensor:
        frame_t, height, width, channels = frame_seq.shape
        frame_dif = self.total_frames - frame_t

        frame_pad = torch.zeros(frame_dif, height, width, channels, dtype=torch.float)
        return torch.cat((frame_pad, frame_seq))

class VideoFrameSequencerCV2(VideoFrameSequencer):
    def __init__(self, video_dir:str) -> None:
        super().__init__()
        self.reader = cv2.VideoCapture(video_dir)
    
    def get_frame_set(self, start_timestamp, start_frame, end_frame):
        count = start_frame
        # Set read head to start_frame
        # self.reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

        # Get frames
        combined_frames = None
        success = True
        bad_frame = 0
        for frame_idx in range(start_frame-1, end_frame):
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            # Read single frame
            success, frame = self.reader.read()
            if frame is None:
                print("frame_idx:", frame_idx)
                bad_frame += 1
            if frame is not None:
                temp = torch.tensor(frame.T)
                temp = torch.unsqueeze(temp, 0)
                if combined_frames is None:
                    combined_frames = temp
                else:
                    combined_frames = torch.cat((combined_frames, temp))

        """ while success and count <= end_frame:
            # Read single frame
            success, frame = self.reader.read()
            if success:
                temp = torch.tensor(frame.T)
                temp = torch.unsqueeze(temp, 0)
                if combined_frames is None:
                    combined_frames = temp
                else:
                    combined_frames = torch.cat((combined_frames, temp)) """
        print('Bad frames:', bad_frame)
        return combined_frames

class VideoFrameSequencerSKVideo(VideoFrameSequencer):
    def __init__(self, video_dir:str, total_frames=256) -> None:
        super().__init__(total_frames)
        self.reader = FFmpegReader(video_dir)
    
    def get_frame_subset(self, start_frame, end_frame):
        combined_frames = None
        for frame in islice(self.reader.nextFrame(), start_frame-1, end_frame):
            temp = torch.tensor(frame, dtype=torch.float)
            
            # Preprocess frame
            temp = self.frame_transform(temp)

            temp = torch.unsqueeze(temp, 0)
            #print(temp.shape)

            if combined_frames is None:
                combined_frames = temp
            else:
                combined_frames = torch.cat((combined_frames, temp))
        
        print(combined_frames.shape)
        combined_frames = self.pad_frames(combined_frames)
        
        combined_frames = combined_frames.permute(3, 0, 1, 2)
        combined_frames = combined_frames.unsqueeze(0)
        return combined_frames



class VideoFrameSequencerTorchVision(VideoFrameSequencer):
    def __init__(self, video_dir:str) -> None:
        super().__init__()
        #self.vidCap = cv2.VideoCapture(video_dir)
        #self.total_frames = count_frames(video_dir)
        self.reader = torchvision.io.VideoReader(video_dir, stream="video")
        #self.frames = self.get_all_frames()
    
    def get_all_frames(self):
        combined_frames = None
        for frame in self.reader:          
            # Concat all the frames
            current_frame = torch.unsqueeze(frame['data'], 0)
            if combined_frames is None:
                combined_frames = current_frame
            else:
                combined_frames = torch.cat((combined_frames, current_frame))
        return combined_frames

    def get_frame_subset(self, start_frame, end_frame):
        # First frame is 1, change to starting at 0
        start_frame -= 1
        end_frame -= 1

        idx = torch.LongTensor(range(start_frame, end_frame + 1))
        return self.frames.index_select(0, idx)

    def get_frame_set(self, start_timestamp, start_frame, end_frame):
        # Move seek to start of section
        self.reader.seek(start_timestamp)
        # Get # of frames
        total_frames = end_frame - start_frame + 1

        count = 1
        combined_frames = None
        for frame in self.reader:
            # Exit loop if read all the frames
            if count > total_frames:
                break
            
            # Concat all the frames
            current_frame = torch.unsqueeze(frame['data'], 0)
            if combined_frames is None:
                combined_frames = current_frame
            else:
                combined_frames = torch.cat((combined_frames, current_frame))
            
            count += 1
        
        return combined_frames
            

class VideoFrameDataset(Dataset):
    def __init__(self, dataset_csv:str, cache_capacity:int = 15) -> None:
        super().__init__()
        self.dataset_df = pd.read_csv(dataset_csv)
        self.narration_ids = self.dataset_df['narration_id']
        self.dataset_df.set_index('narration_id', inplace=True)
        self.cache = LRUCache(capacity=cache_capacity)
    
    def __getitem__(self, index) -> torch.Tensor:
        if index < 0 or index >= len(self.narration_ids):
            raise IndexError()
        
        seq = self.cache.get(index)
        if seq is None:
            nar_row = self.dataset_df.loc[self.narration_ids[index]]

            seq = self._get_video_seq(row=nar_row)

            self.cache.put(index, seq)
        
        return seq

    def _get_video_seq(self, row:pd.Series) -> torch.Tensor:

        #timestamp = self._convert_timestamp(row['start_timestamp'])
        start_frame = row['start_frame']
        stop_frame = row['stop_frame']
        video_file = row['video_id']+".MP4"
        narration = row['narration']

        vid_sequencer = self._get_vid_sequencer(video_file)

        return vid_sequencer.get_frame_subset(start_frame, stop_frame), narration

        
    
    def _convert_timestamp(self, formatted_ts:str) -> float:
        sections = list(map(float, formatted_ts.split(":")))
        sections[0] *= 60
        return sum(sections)

    def _get_vid_sequencer(self, video_file:str)->VideoFrameSequencer:
        
        return VideoFrameSequencerSKVideo(video_file)
    
    def get_random_batch(self, batch_size:int=256):
        if batch_size < 0 or batch_size > len(self.narration_ids):
            raise IndexError()

        batch_idxs = sample(range(len(self.narration_ids)), batch_size)

        for idx in batch_idxs:
            yield self.__getitem__(idx)


if __name__ == '__main__':
    #print(sys.getsizeof(VideoFrameSequencer('P01_11.MP4')))
    #print(sys.getsizeof(17))

    ds = VideoFrameDataset("EPIC_100_VALIDATION.csv")
    print(ds[0])
    """
    v = ViT(
        image_size = 128,          # image size
        frames = 16,               # number of frames
        image_patch_size = 16,     # image patch size
        frame_patch_size = 2,      # frame patch size
        num_classes = 1000,
        dim = 1024,
        spatial_depth = 6,         # depth of the spatial transformer
        temporal_depth = 6,        # depth of the temporal transformer
        heads = 8,
        mlp_dim = 2048
    )
    print(v)

    with open('vit_output.txt', 'w') as f:
        print(v, file=f) """

    """ reader = torchvision.io.VideoReader("P01_11.MP4")
    first_frame = next(reader)['data']
    reader = torchvision.io.VideoReader("P01_11.MP4")
    count = 0
    for frame in reader:
        if count % 100 == 0:
            print(count)
        if count == 11234 - 1:
            next_frame = frame['data']
            break
        count += 1

    #next_frame = next(reader)['data']

    print(torch.equal(first_frame, next_frame)) """




