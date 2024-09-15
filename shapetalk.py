import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud

NUM_POINTS = 1024
PROMPTS = "prompts"
UTTERANCE = "utterance"
SOURCE_UID = "source_uid"
TARGET_UID = "target_uid"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"


class ShapeTalk(Dataset):
    def __init__(
        self,
        batch_size: int,
        df: pd.DataFrame,
        partnet_dict: dict,
        device: torch.device,
    ):
        super().__init__()
        self.prompts = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(
            df.iterrows(), total=len(df), desc="Creating ShapeTalk dataset"
        ):
            self._append_sample(row, device, partnet_dict)
        self.set_length(batch_size)

    def _append_sample(self, row, device, partnet_dict):
        source_uid, target_uid, prompt = (
            row[SOURCE_UID],
            row[TARGET_UID],
            row[UTTERANCE],
        )
        self.prompts.append(prompt)
        source_pc = PointCloud.load_partnet(partnet_dict[source_uid], source_uid)
        target_pc = PointCloud.load_partnet(partnet_dict[target_uid], target_uid)
        source_pc = source_pc.farthest_point_sample(NUM_POINTS)
        target_pc = target_pc.farthest_point_sample(NUM_POINTS)
        self.source_latents.append(source_pc.encode().to(device))
        self.target_latents.append(target_pc.encode().to(device))

    def set_length(self, batch_size, length=None):
        if length is None:
            self.length = len(self.prompts)
        else:
            assert length <= len(self.prompts)
            self.length = length
        r = self.length % batch_size
        if r == 0:
            self.logical_length = self.length
        else:
            q = batch_size - r
            self.logical_length = self.length + q

    def __len__(self):
        return self.logical_length

    def __getitem__(self, logical_index):
        index = logical_index % self.length
        return {
            PROMPTS: self.prompts[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
