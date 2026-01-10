import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import pandas as pd
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
import pickle
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH

class Text2CAD_Dataset(Dataset):
    def __init__(
        self,
        cad_seq_dir: str,
        prompt_path: str,
        split_filepath: str,
        subset: str,
        max_workers: int,
        debug: bool = False,
    ):
        """
        Args:
            cad_seq_dir (string): Directory with all the .pth files.
            prompt_path (string): Directory with all the .npz files.
            split_filepath (string): Train_Test_Val json file path.
            subset (string): "train", "test" or "val"
        """
        super(Text2CAD_Dataset, self).__init__()
        self.cad_seq_dir = cad_seq_dir
        self.prompt_path = prompt_path
        self.all_prompt_choices = ["abstract", "beginner", "intermediate", "expert"]
        self.substrings_to_remove = ["*", "\n", '"', "\_", "\\", "\t", "-", ":"]
        
        # open spilt json
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)

        self.uid_pair = self.split[subset]
        if debug:
            self.uid_pair = self.uid_pair[:10] # Limit to 10 samples in debug mode

        self.prompt_data = {}
        self.keys = []

        try:
            data = pd.read_pickle(prompt_path)
            if isinstance(data, dict):
                # It's already the pre-processed prompt_data!
                # Split data by ratio (80% train, 20% validation) instead of using split file
                all_data = data
                all_keys = sorted(list(all_data.keys()))  # Sort for reproducibility
                
                # Use fixed seed for reproducible splits
                import random
                rng = random.Random(42)
                rng.shuffle(all_keys)
                
                total_samples = len(all_keys)
                train_ratio = 0.8
                train_size = int(total_samples * train_ratio)
                
                if subset == "train":
                    selected_keys = all_keys[:train_size]
                elif subset == "validation":
                    selected_keys = all_keys[train_size:]
                else:  # test - use validation set
                    selected_keys = all_keys[train_size:]
                
                self.prompt_data = {k: all_data[k] for k in selected_keys}
                self.keys = list(self.prompt_data.keys())
                
                if debug:
                    self.keys = self.keys[:10]
                print(f"Found {len(self.keys)} samples for {subset} split (auto-split from {total_samples} total).")
                return
            
            self.prompt_df = data
            self.prompt_df = self.prompt_df[
                self.prompt_df["abstract"].notnull()
                & self.prompt_df["beginner"].notnull()
                & self.prompt_df["intermediate"].notnull()
                & self.prompt_df["expert"].notnull()
            ]
        except (MemoryError, KeyError) as e:
            if debug:
                print(f"Error caught while loading {prompt_path}: {type(e).__name__}. Creating mock data for debug mode.")
                # Create mock data
                import numpy as np
                mock_cad_vec = torch.zeros((10, 2)) # Mock shape
                mock_mask = {"mask": torch.ones(10)}
                for uid in self.uid_pair:
                    for choice in self.all_prompt_choices:
                        key = f"{uid}_{choice}"
                        self.prompt_data[key] = (mock_cad_vec, f"Mock {choice} prompt for {uid}", mock_mask)
                self.keys = list(self.prompt_data.keys())
                print(f"Found {len(self.keys)} samples for {subset} split (Mock mode).")
                return
            else:
                raise

        func = self._prepare_data
        
        # Load the prompt data using ThreadPoolExecutor and _prepare_data function
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use ThreadPoolExecutor to process the prompt data in parallel
            for data in tqdm(
                executor.map(func, self.uid_pair),
                total=len(self.uid_pair),
                desc=f"Loading {subset} split",
            ):
                if data is not None:
                    uid, cad_vec, prompt, mask_cad_dict = data
                    if isinstance(prompt, dict):
                        for key, val in prompt.items():
                            self.prompt_data[uid + f"_{key}"] = (
                                cad_vec,
                                val,
                                mask_cad_dict,
                            )  # "0000/00001234" -> "0000/00001234_beginner"

        self.keys = list(self.prompt_data.keys())
        print(f"Found {len(self.prompt_data)} samples for {subset} split.")

    def __len__(self):
        return len(self.keys)

    def _prepare_data(self, uid):
        root_id, chunk_id = uid.split("/")
        if len(self.prompt_df[self.prompt_df["uid"] == uid]) == 0:
            return None
        try:
            cad_vec_dict = torch.load(
                os.path.join(self.cad_seq_dir, root_id, chunk_id, "seq", f"{chunk_id}.pth"),
                weights_only=True,
            )
        except Exception as e:
            # print(f"Error loading CAD seq for {uid}: {e}")
            return None
            
        level_data = {}
        for prompt_choice in self.all_prompt_choices:
            prompt = self.prompt_df[self.prompt_df["uid"] == uid][prompt_choice].iloc[0]
            if isinstance(prompt, str):
                level_data[prompt_choice] = self.remove_substrings(prompt, self.substrings_to_remove).lower()
        
        if len(level_data) == 0:
            return None
            
        # Generate flag_vec and index_vec using CADSequence
        try:
            # Check if vec is already a dict with processed data
            if isinstance(cad_vec_dict["vec"], dict):
                vec_dict = cad_vec_dict["vec"]
            else:
                cad_seq_obj = CADSequence.from_vec(cad_vec_dict["vec"], denumericalize=False)
                cad_seq_obj.to_vec(padding=True, max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH)
                
                vec_dict = {
                    "cad_vec": cad_seq_obj.cad_vec,
                    "flag_vec": cad_seq_obj.flag_vec,
                    "index_vec": cad_seq_obj.index_vec,
                }
        except Exception as e:
            # Fallback if CADSequence fails
            if isinstance(cad_vec_dict["vec"], dict):
                vec_dict = cad_vec_dict["vec"]
            else:
                vec_dict = {
                    "cad_vec": cad_vec_dict["vec"],
                    "flag_vec": torch.zeros(cad_vec_dict["vec"].shape[0], dtype=torch.int32),
                    "index_vec": torch.zeros(cad_vec_dict["vec"].shape[0], dtype=torch.int32),
                }

        # Fix attn_mask dimension if needed
        mask_cad_dict = cad_vec_dict["mask_cad_dict"]
        seq_len = vec_dict["cad_vec"].shape[0]
        if mask_cad_dict["attn_mask"].shape[0] != seq_len:
            # Regenerate attn_mask with correct size
            mask_cad_dict["attn_mask"] = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
            )
        
        return uid, vec_dict, level_data, mask_cad_dict

    def remove_substrings(self, text, substrings):
        """
        Remove specified substrings from the input text.

        Args:
            text (str): The input text to be cleaned.
            substrings (list): A list of substrings to be removed.

        Returns:
            str: The cleaned text with specified substrings removed.
        """
        # Escape special characters in substrings and join them to form the regex pattern
        regex_pattern = "|".join(re.escape(substring) for substring in substrings)
        # Use re.sub to replace occurrences of any substrings with an empty string
        cleaned_text = re.sub(regex_pattern, " ", text)
        # Remove extra white spaces
        cleaned_text = re.sub(" +", " ", cleaned_text)
        return cleaned_text

    def __getitem__(self, idx):
        uid = self.keys[idx]
        return uid, *self.prompt_data[uid]


def get_dataloaders(
    cad_seq_dir: str,
    prompt_path: str,
    split_filepath: str,
    subsets: list[str],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
    prefetch_factor: int,
    debug: bool = False,
):
    """
    Generate a DataLoader for the NL2CADDataset.

    Args:
    - cad_seq_dir (str): The directory containing the CAD sequence files.
    - prompt_path (str): The path to the CSV file containing the prompts.
    - split_filepath (str): The path to the JSON file containing the train/test/validation split.
    - subsets (list[str]): The subset to use ("train", "test", or "val").
    - batch_size (int): The batch size.
    - shuffle (bool): Whether to shuffle the data.
    - pin_memory (bool): Whether to pin memory.
    - num_workers (int): The number of workers.
    - prefetch_factor (int): The prefetch factor.
    - debug (bool): Whether to use debug mode.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object.
    """

    all_dataloaders = []

    for subset in subsets:
        # Create an instance of the NL2CADDataset
        dataset = Text2CAD_Dataset(
            cad_seq_dir=cad_seq_dir,
            prompt_path=prompt_path,
            split_filepath=split_filepath,
            subset=subset,
            max_workers=num_workers,
            debug=debug,
        )

        # Create a DataLoader with the specified parameters
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,  # You can set this to True if you want to shuffle the data
            num_workers=num_workers,
            pin_memory=pin_memory,  # Set to True if using CUDA
            prefetch_factor=prefetch_factor,
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders


if __name__ == "__main__":
    pass
