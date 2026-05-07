import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from human_plan.preprocessing.preprocessing import preprocess_multimodal_vla, preprocess_vla
from human_plan.preprocessing.prompting_format import preprocess_language_instruction
from llava.mm_utils import process_image_ndarray_v2
from llava.train.args import DataArguments


@dataclass
class _EpisodeData:
    video_path: str
    states: np.ndarray
    actions: np.ndarray
    frame_indices: np.ndarray
    task_indices: np.ndarray
    done_flags: np.ndarray
    task_descriptions: Dict[int, str]
    state_slices: Optional[Sequence[Tuple[int, int]]]
    action_slices: Optional[Sequence[Tuple[int, int]]]


class SimpleLeRobotVLADataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        data_root: str,
        task_dir: Optional[str] = None,
        split: str = "train",
        data_skip: int = 1,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_root = os.path.expanduser(str(data_root))
        self.task_dir = str(task_dir or "").strip()
        self.split = split
        self.data_skip = int(data_skip)
        self.max_samples = None if max_samples in (None, "") else int(max_samples)

        if self.data_skip < 1:
            raise ValueError(f"data_skip must be >= 1, got {self.data_skip}")
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"SIMPLE LeRobot data_root not found: {self.data_root}")
        if not hasattr(self.data_args, "action_tokenizer"):
            raise ValueError("data_args.action_tokenizer must be initialized before building SIMPLE LeRobot data")

        self.episodes: List[_EpisodeData] = []
        self.samples: List[Tuple[int, int]] = []
        for task_path in self._iter_task_paths():
            self._load_task(task_path)

        self.samples = self.samples[:: self.data_skip]
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]
        if not self.samples:
            raise ValueError(
                f"No usable SIMPLE LeRobot samples found under data_root={self.data_root}, task_dir={self.task_dir!r}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_idx, row_idx = self.samples[index]
        episode = self.episodes[episode_idx]

        state = self._reorder(episode.states[row_idx], episode.state_slices)
        actions = np.stack([self._reorder(action, episode.action_slices) for action in episode.actions], axis=0)
        future_actions, future_mask = self._future_actions(actions, row_idx)

        self._validate_vector_dim("states", state, getattr(self.data_args, "proprio_size", None))
        self._validate_vector_dim("action", future_actions[0], getattr(self.data_args, "traj_action_output_dim", None))

        language_label = episode.task_descriptions.get(
            int(episode.task_indices[row_idx]),
            "finish the task",
        )
        language_instruction = preprocess_language_instruction(language_label, 0, self.data_args)
        language_instruction = preprocess_multimodal_vla(language_instruction, self.data_args)

        data_dict = preprocess_vla(
            language_instruction,
            torch.tensor(state, dtype=torch.float32).reshape(1, -1),
            torch.tensor(future_actions, dtype=torch.float32),
            torch.tensor(future_mask, dtype=torch.bool),
            self.data_args.action_tokenizer,
            self.tokenizer,
            mask_input=getattr(self.data_args, "mask_input", False),
            mask_ignore=getattr(self.data_args, "mask_ignore", False),
            raw_action_label=getattr(self.data_args, "raw_action_label", False),
            traj_action_output_dim=getattr(self.data_args, "traj_action_output_dim", future_actions.shape[-1]),
            input_placeholder_diff_index=getattr(self.data_args, "input_placeholder_diff_index", False),
            sep_query_token=getattr(self.data_args, "sep_query_token", False),
            language_response=None,
            include_response=getattr(self.data_args, "include_response", False),
            include_repeat_instruction=getattr(self.data_args, "include_repeat_instruction", False),
            raw_language_label=language_label,
        )

        image = self._read_rgb_frame(episode.video_path, int(episode.frame_indices[row_idx]))
        image_tensor = process_image_ndarray_v2(
            image,
            self.data_args,
            reverse_channel_order=False,
        )
        data_dict["image"] = torch.stack([image_tensor], dim=0)
        data_dict["proprio_input_2d"] = torch.zeros((1, 4), dtype=torch.float32)
        data_dict["proprio_input_3d"] = torch.zeros((1, 6), dtype=torch.float32)
        data_dict["proprio_input_rot"] = torch.zeros((1, 6), dtype=torch.float32)
        data_dict["proprio_input_handdof"] = torch.zeros((1, 30), dtype=torch.float32)
        data_dict["proprio_input_hand_finger_tip"] = torch.zeros((1, 30), dtype=torch.float32)
        data_dict["ee_movement_mask"] = torch.ones((1, 2), dtype=torch.float32)
        data_dict["raw_width"] = image.shape[1]
        data_dict["raw_height"] = image.shape[0]
        data_dict["language_label"] = language_label
        return data_dict

    def _iter_task_paths(self) -> Sequence[str]:
        if self.task_dir:
            task_path = os.path.join(self.data_root, self.task_dir)
            if not os.path.isdir(task_path):
                raise FileNotFoundError(f"SIMPLE LeRobot task_dir not found: {task_path}")
            return [task_path]
        return [
            os.path.join(self.data_root, name)
            for name in sorted(os.listdir(self.data_root))
            if os.path.isfile(os.path.join(self.data_root, name, "meta", "info.json"))
        ]

    def _load_task(self, task_path: str) -> None:
        info_path = os.path.join(task_path, "meta", "info.json")
        with open(info_path, "r") as f:
            info = json.load(f)

        task_descriptions = self._load_task_descriptions(task_path)
        state_slices = self._load_modality_slices(task_path, "state", "states")
        action_slices = self._load_modality_slices(task_path, "action", "action")
        data_glob = os.path.join(task_path, "data", "chunk-*", "episode_*.parquet")

        for parquet_path in sorted(glob.glob(data_glob)):
            episode_index = self._episode_index(parquet_path)
            episode_chunk = self._episode_chunk(parquet_path)
            video_path = os.path.join(
                task_path,
                info["video_path"].format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                ),
            )
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"SIMPLE LeRobot video not found for {parquet_path}: {video_path}")

            episode = self._read_episode(parquet_path, video_path, task_descriptions, state_slices, action_slices)
            current_episode_idx = len(self.episodes)
            self.episodes.append(episode)
            self.samples.extend(
                (current_episode_idx, row_idx)
                for row_idx, done in enumerate(episode.done_flags.tolist())
                if not done
            )

    def _read_episode(
        self,
        parquet_path: str,
        video_path: str,
        task_descriptions: Dict[int, str],
        state_slices: Optional[Sequence[Tuple[int, int]]],
        action_slices: Optional[Sequence[Tuple[int, int]]],
    ) -> _EpisodeData:
        table = pq.read_table(
            parquet_path,
            columns=["states", "action", "frame_index", "task_index", "next.done"],
        )
        states = np.asarray(table.column("states").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        task_indices = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
        done_flags = np.asarray(table.column("next.done").to_pylist(), dtype=bool)

        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"Expected 2D states/actions in {parquet_path}, got {states.shape} and {actions.shape}")

        return _EpisodeData(
            video_path=video_path,
            states=states,
            actions=actions,
            frame_indices=frame_indices,
            task_indices=task_indices,
            done_flags=done_flags,
            task_descriptions=task_descriptions,
            state_slices=state_slices,
            action_slices=action_slices,
        )

    def _future_actions(self, actions: np.ndarray, row_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        future_index = int(getattr(self.data_args, "future_index", 0))
        horizon = int(getattr(self.data_args, "predict_future_step", 1))
        action_dim = actions.shape[-1]
        start = min(row_idx + future_index, len(actions) - 1)
        end = min(start + horizon, len(actions))
        valid = actions[start:end]

        labels = np.zeros((horizon, action_dim), dtype=np.float32)
        mask = np.zeros((horizon, action_dim), dtype=bool)
        labels[: len(valid)] = valid
        mask[: len(valid)] = True
        if len(valid) < horizon:
            labels[len(valid) :] = valid[-1]
        return labels, mask

    @staticmethod
    def _load_task_descriptions(task_path: str) -> Dict[int, str]:
        tasks_path = os.path.join(task_path, "meta", "tasks.jsonl")
        descriptions: Dict[int, str] = {}
        if not os.path.isfile(tasks_path):
            return descriptions
        with open(tasks_path, "r") as f:
            for line in f:
                item = json.loads(line)
                descriptions[int(item["task_index"])] = item.get("description", item.get("task", "finish the task"))
        return descriptions

    @staticmethod
    def _load_modality_slices(
        task_path: str,
        section: str,
        expected_key: str,
    ) -> Optional[Sequence[Tuple[int, int]]]:
        modality_path = os.path.join(task_path, "meta", "modality.json")
        if not os.path.isfile(modality_path):
            return None
        with open(modality_path, "r") as f:
            modality = json.load(f)
        entries = modality.get(section, {})
        if not entries:
            return None
        original_keys = {meta.get("original_key") for meta in entries.values()}
        if original_keys != {expected_key}:
            return None
        return [(int(meta["start"]), int(meta["end"])) for meta in entries.values()]

    @staticmethod
    def _reorder(vector: np.ndarray, slices: Optional[Sequence[Tuple[int, int]]]) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        if slices is None:
            return vector
        return np.concatenate([vector[start:end] for start, end in slices], axis=0).astype(np.float32)

    @staticmethod
    def _read_rgb_frame(video_path: str, frame_idx: int) -> np.ndarray:
        capture = cv2.VideoCapture(video_path)
        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open SIMPLE LeRobot video: {video_path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {frame_idx} from SIMPLE LeRobot video: {video_path}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    @staticmethod
    def _episode_index(parquet_path: str) -> int:
        match = re.search(r"episode_(\d+)\.parquet$", parquet_path)
        if match is None:
            raise ValueError(f"Could not parse episode index from {parquet_path}")
        return int(match.group(1))

    @staticmethod
    def _episode_chunk(parquet_path: str) -> int:
        match = re.search(r"chunk-(\d+)", parquet_path)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _validate_vector_dim(name: str, value: np.ndarray, expected_dim: Optional[int]) -> None:
        if expected_dim in (None, 0):
            return
        if value.shape[-1] != int(expected_dim):
            raise ValueError(f"SIMPLE LeRobot {name} dim mismatch: expected {expected_dim}, got {value.shape[-1]}")
