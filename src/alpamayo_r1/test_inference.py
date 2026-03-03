# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import torch
import numpy as np
import time

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
t_start_total = time.perf_counter()
t_start_load = t_start_total
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, "cuda")
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_end_load = time.perf_counter()
timing_load_ms = (t_end_load - t_start_load) * 1000.0

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_end_total = time.perf_counter()
timing_total_ms = (t_end_total - t_start_total) * 1000.0
timing_cot_ms = float(
    extra.get(
        "timing_cot_ms",
        float(extra.get("timing_cot_s", float("nan"))) * 1000.0,
    )
)
timing_vision_ms = float(extra.get("timing_vision_ms", float("nan")))
timing_traj_gen_ms = float(
    extra.get(
        "timing_traj_gen_ms",
        float(extra.get("timing_traj_gen_s", float("nan"))) * 1000.0,
    )
)

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
print("\n=== 耗时统计 ===")
print(f"数据加载与预处理: {timing_load_ms:.2f} ms")
print(f"Vision Encoder:   {timing_vision_ms:.2f} ms")
print(f"CoT 推理:         {timing_cot_ms:.2f} ms")
print(f"轨迹生成:         {timing_traj_gen_ms:.2f} ms")
print(f"总耗时:           {timing_total_ms:.2f} ms")

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)
