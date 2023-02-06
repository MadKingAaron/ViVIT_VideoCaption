import torch
import skvideo.io
import random
import os
import pandas as pd
from torch.utils.data import Dataset
from frame_transforms import FrameTransforms
from itertools import islice


class VideoFrameSequencer(Dataset):
    def __init__(self, total_frames:int=256, frame_shape:int=512) -> None:
        super().__init__()
        self.frame_transform = FrameTransforms(frame_shape=frame_shape)
        self.total_frames = total_frames
        
    
    def pad_frames(self, frame_seq:torch.Tensor) -> torch.Tensor:
        frame_t, height, width, channels = frame_seq.shape
        frame_dif = self.total_frames - frame_t

        frame_pad = torch.zeros(frame_dif, height, width, channels, dtype=torch.float)
        return torch.cat((frame_pad, frame_seq))
    
    def ensure_size(self, frame_seq:torch.Tensor) -> torch.Tensor:
        frame_t, height, width, channels = frame_seq.shape
        if frame_t > self.total_frames:
            frame_seq = frame_seq[:self.total_frames]
        return frame_seq


class VideoFrameSequencerSKVideo(VideoFrameSequencer):
    def __init__(self, total_frames:int=256, frame_shape:int=512) -> None:
        super().__init__(total_frames, frame_shape)
        self.reader = None
    
    def load_video(self, video_dir:str):
        self.reader = skvideo.io.vreader(video_dir)

    def get_frame_subset(self, start_frame, end_frame) -> torch.Tensor:
        combined_frames = None
        if self.reader is None:
            return combined_frames # TODO Have throw an error
        for frame in islice(self.reader, start_frame-1, end_frame):
            temp = torch.tensor(frame, dtype=torch.float)
            
            # Preprocess frame
            temp = self.frame_transform(temp)

            temp = torch.unsqueeze(temp, 0)
            #print(temp.shape)

            if combined_frames is None:
                combined_frames = temp
            else:
                combined_frames = torch.cat((combined_frames, temp))
        
        #print(combined_frames.shape)
        combined_frames = self.ensure_size(combined_frames)
        combined_frames = self.pad_frames(combined_frames)
        
        combined_frames = combined_frames.permute(3, 0, 1, 2)
        combined_frames = combined_frames.unsqueeze(0)
        return combined_frames

class PretrainDataset(Dataset):
    def __init__(self, dataset_csv:str, frame_shape:int, total_frames:int, video_file_col:str = 'video_file', video_folder_dir:str = './pretrain_data') -> None:
        super().__init__()
       
        self.video_folder_dir = video_folder_dir
        self.video_file_col = video_file_col
        
        self.dataset_df = self.get_df(dataset_csv)
        self.dataset_size = self.dataset_df.shape[0]
        
        self.reader = VideoFrameSequencerSKVideo(frame_shape=frame_shape, total_frames=total_frames)
        
    
    def get_df(self, dataset_csv:str):
        files_in_videos_dir = [f for f in os.listdir(self.video_folder_dir) if f.endswith(".mp4")]
       
        df = pd.read_csv(dataset_csv)
        df = df[df[self.video_file_col].isin(files_in_videos_dir)].reset_index()

        df = df.drop(['index', 'Unnamed: 0'], axis=1)

        return df
        

    def _get_clip(self, index:int):
        video_dir = self.video_folder_dir + '/' + self.dataset_df.loc[index, self.video_file_col]
        start_frame = self.dataset_df.loc[index, 'start_frame']
        end_frame = self.dataset_df.loc[index, 'stop_frame']
        caption = self.dataset_df.loc[index, 'annotation']

        self.reader.load_video(video_dir=video_dir)

        frame_seq = self.reader.get_frame_subset(start_frame=start_frame, end_frame=end_frame)

        return frame_seq, caption
    
    def __getitem__(self, index:int) -> torch.Tensor:
        if index >= self.dataset_size:
            print("Idx:"+str(index)+" Size:"+str(self.dataset_size))
            raise IndexError()
        return self._get_clip(index)
    
    def get_batch(self, batch_size:int = 256) -> tuple[torch.Tensor, list]:
        samples = random.sample(range(self.dataset_size), batch_size)
        batch = None
        captions = []
        for sample in samples:
            if batch is None:
                batch, caption = self[sample]
                captions.append(caption)
            else:
                frame, caption = self[sample]
                captions.append(caption)

                batch = torch.cat((batch, frame))
                del frame
        
        return batch, captions

if __name__ == "__main__":
    dataset_csv = './pretrain_data/training_ds.csv'
    ds = PretrainDataset(dataset_csv=dataset_csv, frame_shape=64, total_frames=128)
    print(ds.dataset_df)
    batch, captions = ds.get_batch(5)
    print(batch.shape, len(captions))
    

