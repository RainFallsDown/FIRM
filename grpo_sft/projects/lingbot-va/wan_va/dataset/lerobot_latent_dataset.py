# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
try:
    from lerobot.datasets.utils import get_episode_data_index as _lerobot_get_episode_data_index
except ImportError:
    _lerobot_get_episode_data_index = None
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from datasets import Dataset as HFDataset
import json
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
try:
    from lerobot.constants import HF_LEROBOT_HOME
except ModuleNotFoundError:
    from lerobot.utils.constants import HF_LEROBOT_HOME

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
            print(f"Error: {e}")
    return result


def _episode_rows_from_meta(meta_episodes):
    if hasattr(meta_episodes, "to_list"):
        return meta_episodes.to_list()
    if isinstance(meta_episodes, dict):
        if all(isinstance(v, dict) for v in meta_episodes.values()):
            return list(meta_episodes.values())
        keys = list(meta_episodes)
        if keys and all(hasattr(meta_episodes[k], "__len__") for k in keys):
            return [
                {k: meta_episodes[k][idx] for k in keys}
                for idx in range(len(meta_episodes[keys[0]]))
            ]
    return list(meta_episodes)


def _load_action_episode_rows(root, fallback_rows):
    episodes_jsonl = Path(root) / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        rows = []
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if any("action_config" in row for row in rows):
            return rows
    return fallback_rows


def _build_episode_data_index(meta_rows, episodes=None, hf_dataset=None):
    if _lerobot_get_episode_data_index is not None:
        try:
            return _lerobot_get_episode_data_index(meta_rows, episodes)
        except Exception:
            pass

    allowed = set(episodes) if episodes is not None else None
    from_map = {}
    to_map = {}
    for row in meta_rows:
        episode_index = int(row["episode_index"])
        if allowed is not None and episode_index not in allowed:
            continue
        if "dataset_from_index" in row and "dataset_to_index" in row:
            from_map[episode_index] = int(row["dataset_from_index"])
            to_map[episode_index] = int(row["dataset_to_index"])

    if from_map:
        return {"from": from_map, "to": to_map}

    if hf_dataset is None or "episode_index" not in hf_dataset.column_names:
        raise RuntimeError("Cannot build episode_data_index from metadata or hf_dataset")

    current_episode = None
    for idx, episode_index in enumerate(hf_dataset["episode_index"]):
        episode_index = int(episode_index)
        if allowed is not None and episode_index not in allowed:
            continue
        if episode_index != current_episode:
            if current_episode is not None:
                to_map[current_episode] = idx
            from_map[episode_index] = idx
            current_episode = episode_index
    if current_episode is not None:
        to_map[current_episode] = len(hf_dataset)
    return {"from": from_map, "to": to_map}


class _LocalLeRobotDatasetMetadata:
    """Minimal local metadata reader for LeRobot datasets without episodes_stats."""

    def __init__(self, root):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(f"Missing local LeRobot metadata under {self.root / 'meta'}")

        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.episodes = []
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))

        self.chunks_size = int(self.info.get("chunks_size", 1000) or 1000)
        self._version = self.info.get("codebase_version", "v3.0")

    def get_episode_chunk(self, episode_index):
        episode_index = int(episode_index)
        candidates = [episode_index, episode_index // max(self.chunks_size, 1)]
        for chunk_index in candidates:
            if (self.root / "latents" / f"chunk-{chunk_index:03d}").exists():
                return chunk_index
            if (self.root / "data" / f"chunk-{chunk_index:03d}").exists():
                return chunk_index
        return candidates[-1]


def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    info_files = [Path(v) for v in recursive_find_file(config.dataset_path, 'info.json')]
    repo_list = sorted({
        str(path.parent.parent)
        for path in info_files
        if path.name == 'info.json' and path.parent.name == 'meta'
    })
    with Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        try:
            self.meta = LeRobotDatasetMetadata(
                self.repo_id, self.root, self.revision, force_cache_sync=False
            )
        except Exception as exc:
            if not (self.root / "meta" / "episodes.jsonl").exists():
                raise
            print(f"Using local LeRobot metadata fallback for {self.root}: {exc}")
            self.meta = _LocalLeRobotDatasetMetadata(self.root)
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            self.hf_dataset = self.load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError, KeyError) as exc:
            raise RuntimeError(f"Failed to load local LeRobot data from {self.root}") from exc
        self.meta_episode_rows = _episode_rows_from_meta(self.meta.episodes)
        self.action_episode_rows = _load_action_episode_rows(self.root, self.meta_episode_rows)
        self.episode_row_by_index = {
            int(row["episode_index"]): row
            for row in self.meta_episode_rows
        }
        self.episode_data_index = _build_episode_data_index(
            self.meta_episode_rows,
            self.episodes,
            self.hf_dataset,
        )
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def load_hf_dataset(self):
        parquet_paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}")
        return HFDataset.from_parquet([str(path) for path in parquet_paths])

    def parse_meta(self):
        out = []
        for value in self.action_episode_rows:
            episode_index = int(value["episode_index"])
            tasks = value.get("tasks") or []
            action_config = value.get("action_config")
            if not action_config:
                action_config = [{
                    "start_frame": 0,
                    "end_frame": int(value["length"]),
                    "action_text": " ".join(tasks),
                }]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _get_episode_chunk(self, episode_index):
        if hasattr(self.meta, "get_episode_chunk"):
            return int(self.meta.get_episode_chunk(episode_index))
        row = self.episode_row_by_index.get(int(episode_index), {})
        for key in ("data/chunk_index", "meta/episodes/chunk_index"):
            if key in row:
                return int(row[key])
        chunks_size = int(getattr(self.meta, "chunks_size", 1000) or 1000)
        return int(episode_index) // max(chunks_size, 1)

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self._get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        episode_from = self.episode_data_index["from"]
        if isinstance(episode_from, dict):
            ep_start = episode_from[int(episode_index)]
        else:
            ep_start = episode_from[int(episode_index)]
            if torch.is_tensor(ep_start):
                ep_start = ep_start.item()
        return int(local_index) + int(ep_start)

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self._get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        latent_layout = getattr(
            self.config,
            'latent_layout',
            'tshape' if self.config.env_type == 'robotwin_tshape' else 'concat_width',
        )
        if latent_layout == 'tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        action_pose_mode = getattr(
            self.config,
            'action_pose_mode',
            'relative' if self.config.env_type == 'robotwin_tshape' else 'raw',
        )
        if action_pose_mode == 'relative': ## TODO support get_relative_pose for other dataset, currently only support robotwin
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
