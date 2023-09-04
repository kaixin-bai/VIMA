from __future__ import annotations

import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange
import cv2
from vima.utils import *
from vima import create_policy_from_ckpt
from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import torch
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

PLACEHOLDER_TOKENS = [
    AddedToken("{base_obj}", **_kwargs),
    AddedToken("{base_obj_1}", **_kwargs),
    AddedToken("{base_obj_2}", **_kwargs),
    AddedToken("{dragged_obj}", **_kwargs),
    AddedToken("{dragged_obj_1}", **_kwargs),
    AddedToken("{dragged_obj_2}", **_kwargs),
    AddedToken("{dragged_obj_3}", **_kwargs),
    AddedToken("{dragged_obj_4}", **_kwargs),
    AddedToken("{dragged_obj_5}", **_kwargs),
    AddedToken("{swept_obj}", **_kwargs),
    AddedToken("{bounds}", **_kwargs),
    AddedToken("{constraint}", **_kwargs),
    AddedToken("{scene}", **_kwargs),
    AddedToken("{demo_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_blicker_obj_3}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
    AddedToken("{start_scene}", **_kwargs),
    AddedToken("{end_scene}", **_kwargs),
    AddedToken("{before_twist_1}", **_kwargs),
    AddedToken("{after_twist_1}", **_kwargs),
    AddedToken("{before_twist_2}", **_kwargs),
    AddedToken("{after_twist_2}", **_kwargs),
    AddedToken("{before_twist_3}", **_kwargs),
    AddedToken("{after_twist_3}", **_kwargs),
    AddedToken("{frame_0}", **_kwargs),
    AddedToken("{frame_1}", **_kwargs),
    AddedToken("{frame_2}", **_kwargs),
    AddedToken("{frame_3}", **_kwargs),
    AddedToken("{frame_4}", **_kwargs),
    AddedToken("{frame_5}", **_kwargs),
    AddedToken("{frame_6}", **_kwargs),
    AddedToken("{ring}", **_kwargs),
    AddedToken("{hanoi_stand}", **_kwargs),
    AddedToken("{start_scene_1}", **_kwargs),
    AddedToken("{end_scene_1}", **_kwargs),
    AddedToken("{start_scene_2}", **_kwargs),
    AddedToken("{end_scene_2}", **_kwargs),
    AddedToken("{start_scene_3}", **_kwargs),
    AddedToken("{end_scene_3}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)


@torch.no_grad()
def main(cfg):
    assert cfg.partition in ALL_PARTITIONS
    assert cfg.task in PARTITION_TO_SPECS["test"][cfg.partition]

    seed = 42
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device)
    env = TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                cfg.task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
                seed=seed,
                render_prompt=True,
                display_debug_window=True,
                hide_arm_rgb=False,
            )
        ),
        bonus_steps=2,
    )

    while True:
        env.global_seed = seed

        obs = env.reset()

        # ==============================================================================================================
        # ============= bkx debug ======================================================================================
        """
        可视化obs
        """
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(221)
        plt.title('segm.front')
        plt.axis('OFF')
        plt.imshow(obs['segm']['front'])
        plt.subplot(222)
        plt.title('segm.top')
        plt.axis('OFF')
        plt.imshow(obs['segm']['top'])
        plt.subplot(223)
        plt.title('rgb.front')
        plt.axis('OFF')
        plt.imshow(obs['rgb']['front'].transpose(1, 2, 0))
        plt.subplot(224)
        plt.title('rgb.top')
        plt.axis('OFF')
        plt.imshow(obs['rgb']['top'].transpose(1, 2, 0))
        # plt.show()
        # ============= bkx debug ======================================================================================
        # ==============================================================================================================

        env.render()

        meta_info = env.meta_info  # {'end_effector_type': 'suction', 'n_objects': 3, 'difficulty': 'easy', 'views': ['front', 'top'], 'modalities': ['segm', 'rgb'], 'seed': 42, 'action_bounds': {'low': array([ 0.25, -0.5 ], dtype=float32), 'high': array([0.75, 0.5 ], dtype=float32)}, 'robot_components': [2, 3, 4], 'obj_id_to_info': {5: {'obj_name': 'container', 'obj_assets': 'container/container-template.urdf', 'obj_size_range': SizeRange(low=(0.15, 0.15, 0.05), high=(0.17, 0.17, 0.05)), 'obj_from_template': True, 'obj_replace_fn': <function container_replace_fn at 0x7f15cc4b04c0>, 'obj_pose_transform_fn': None, 'obj_alias': None, 'obj_novel_name': None, 'obj_template_file': None, 'obj_symmetry': None, 'obj_profile': <ProfilePedia.SQUARE_LIKE: 0>, 'texture_name': 'purple', 'texture_color_value': (0.6901960784313725, 0.47843137254901963, 0.6313725490196078), 'texture_texture_asset': None, 'texture_alias': None, 'texture_novel_name': None}, 6: {'obj_name': 'block', 'obj_assets': 'stacking/block.urdf', 'obj_size_range': SizeRange(low=(0.07, 0.07, 0.07), high=(0.07, 0.07, 0.07)), 'obj_from_template': True, 'obj_replace_fn': None, 'obj_pose_transform_fn': None, 'obj_alias': ['cube'], 'obj_novel_name': None, 'obj_template_file': None, 'obj_symmetry': 0.7853981633974483, 'obj_profile': <ProfilePedia.SQUARE_LIKE: 0>, 'texture_name': 'red swirl', 'texture_color_value': None, 'texture_texture_asset': '/home/kb/gpu02_project/VIMA/VIMABench/vima_bench/tasks/assets/textures/swirls/red_swirl.jpg', 'texture_alias': None, 'texture_novel_name': None}, 7: {'obj_name': 'bowl', 'obj_assets': 'bowl/bowl.urdf', 'obj_size_range': SizeRange(low=(0.17, 0.17, 0), high=(0.17, 0.17, 0)), 'obj_from_template': False, 'obj_replace_fn': None, 'obj_pose_transform_fn': None, 'obj_alias': None, 'obj_novel_name': None, 'obj_template_file': None, 'obj_symmetry': 0, 'obj_profile': <ProfilePedia.CIRCLE_LIKE: 1>, 'texture_name': 'yellow', 'texture_color_value': (0.9294117647058824, 0.788235294117647, 0.2823529411764706), 'texture_texture_asset': None, 'texture_alias': None, 'texture_novel_name': None}}}
        prompt = env.prompt  # 一个例子：{str}'Put the {dragged_obj_1} into the {base_obj}.'
        prompt_assets = env.prompt_assets

        # ==============================================================================================================
        # ============= bkx debug ======================================================================================
        """
        可视化prompt_assets中的内容，为什么prompt_assets和上面的图像对不上？
        """
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(241)
        plt.title('base_obj.rgb.front')
        plt.axis('OFF')
        plt.imshow(prompt_assets['base_obj']['rgb']['front'].transpose(1, 2, 0))
        plt.subplot(242)
        plt.title('base_obj.rgb.top')
        plt.axis('OFF')
        plt.imshow(prompt_assets['base_obj']['rgb']['top'].transpose(1, 2, 0))
        plt.subplot(243)
        plt.title('base_obj.segm.front')
        plt.axis('OFF')
        plt.imshow(prompt_assets['base_obj']['segm']['front'])
        plt.subplot(244)
        plt.title('base_obj.segm.top')
        plt.axis('OFF')
        plt.imshow(prompt_assets['base_obj']['segm']['top'])
        plt.subplot(245)
        plt.title('dragged_obj_1.rgb.front')
        plt.axis('OFF')
        plt.imshow(prompt_assets['dragged_obj_1']['rgb']['front'].transpose(1, 2, 0))
        plt.subplot(246)
        plt.title('dragged_obj_1.rgb.top')
        plt.axis('OFF')
        plt.imshow(prompt_assets['dragged_obj_1']['rgb']['top'].transpose(1, 2, 0))
        plt.subplot(247)
        plt.title('dragged_obj_1.segm.front')
        plt.axis('OFF')
        plt.imshow(prompt_assets['dragged_obj_1']['segm']['front'])
        plt.subplot(248)
        plt.title('dragged_obj_1.segm.top')
        plt.axis('OFF')
        plt.imshow(prompt_assets['dragged_obj_1']['segm']['top'])
        plt.show()
        # ============= bkx debug ======================================================================================
        # ==============================================================================================================

        elapsed_steps = 0
        inference_cache = {}
        while True:
            if elapsed_steps == 0:
                # todo: prepare_prompt这个函数在做什么
                """
                prompt: str, 'Put the {dragged_obj_1} into the {base_obj}.'
                prompt_assets: 这是个字典，key有两个，分别是'base_obj'和'dragged_obj_1'，'base_obj'是箱子盒子，'dragged_obj_1'是要放到箱子里的物体
                               每个key下面又是三个字典，其中key为'placeholder_type'的value为'object'
                                                    key为'rgb'和'segm'的下面有两个图分别是'front'和'top'，rgb为3通道的图像，segm为单通道的图像
                               在'base_obj'-'segm'下面又多了个'obj_info'，内容为：{'obj_id': 0, 'obj_name': 'container', 'obj_color': 'purple'}
                               在'dragged_obj_1'-'segm'下面的'obj_info'，内容为：{'obj_id': 0, 'obj_name': 'block', 'obj_color': 'red swirl'}
                """
                print("debug prompt: {}".format(prompt))
                prompt_token_type, word_batch, image_batch = prepare_prompt(
                    prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
                )
                word_batch = word_batch.to(cfg.device)
                image_batch = image_batch.to_torch_tensor(device=cfg.device)
                # todo: forward_prompt_assembly这个函数的作用，输入和输出
                prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                    (prompt_token_type, word_batch, image_batch)
                )

                inference_cache["obs_tokens"] = []
                inference_cache["obs_masks"] = []
                inference_cache["action_tokens"] = []
            obs["ee"] = np.asarray(obs["ee"])
            obs = add_batch_dim(obs)
            obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
                device=cfg.device
            )
            obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
            obs_token_this_step = obs_token_this_step.squeeze(0)
            obs_mask_this_step = obs_mask_this_step.squeeze(0)
            inference_cache["obs_tokens"].append(obs_token_this_step[0])
            inference_cache["obs_masks"].append(obs_mask_this_step[0])
            max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
            obs_tokens_to_forward, obs_masks_to_forward = [], []
            obs_tokens_this_env, obs_masks_this_env = [], []
            for idx in range(len(inference_cache["obs_tokens"])):
                obs_this_env_this_step = inference_cache["obs_tokens"][idx]
                obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
                required_pad = max_objs - obs_this_env_this_step.shape[0]
                obs_tokens_this_env.append(
                    any_concat(
                        [
                            obs_this_env_this_step,
                            torch.zeros(
                                required_pad,
                                obs_this_env_this_step.shape[1],
                                device=cfg.device,
                                dtype=obs_this_env_this_step.dtype,
                            ),
                        ],
                        dim=0,
                    )
                )
                obs_masks_this_env.append(
                    any_concat(
                        [
                            obs_mask_this_env_this_step,
                            torch.zeros(
                                required_pad,
                                device=cfg.device,
                                dtype=obs_mask_this_env_this_step.dtype,
                            ),
                        ],
                        dim=0,
                    )
                )
            obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
            obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
            obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
            obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
            obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
            obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

            if elapsed_steps == 0:
                action_tokens_to_forward = None
            else:
                action_tokens_to_forward = any_stack(
                    [any_stack(inference_cache["action_tokens"], dim=0)],
                    dim=0,
                )
                action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
            predicted_action_tokens = policy.forward(
                obs_token=obs_tokens_to_forward,
                action_token=action_tokens_to_forward,
                prompt_token=prompt_tokens,
                prompt_token_mask=prompt_masks,
                obs_mask=obs_masks_to_forward,
            )  # (L, B, E)
            predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(
                0
            )  # (1, B, E)
            dist_dict = policy.forward_action_decoder(predicted_action_tokens)
            # todo: actions里面东西的物理含义，position和rotation是什么坐标系下的
            actions = {k: v.mode() for k, v in dist_dict.items()}
            action_tokens = policy.forward_action_token(actions)  # (1, B, E)
            action_tokens = action_tokens.squeeze(0)  # (B, E)
            inference_cache["action_tokens"].append(action_tokens[0])
            # todo: 这个函数在做什么
            actions = policy._de_discretize_actions(actions)
            action_bounds = [meta_info["action_bounds"]]
            action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
            action_bounds_high = [
                action_bound["high"] for action_bound in action_bounds
            ]
            action_bounds_low = np.asarray(action_bounds_low)
            action_bounds_high = np.asarray(action_bounds_high)
            action_bounds_low = torch.tensor(
                action_bounds_low, dtype=torch.float32, device=cfg.device
            )
            action_bounds_high = torch.tensor(
                action_bounds_high, dtype=torch.float32, device=cfg.device
            )
            actions["pose0_position"] = (
                    actions["pose0_position"] * (action_bounds_high - action_bounds_low)
                    + action_bounds_low
            )
            actions["pose1_position"] = (
                    actions["pose1_position"] * (action_bounds_high - action_bounds_low)
                    + action_bounds_low
            )
            actions["pose0_position"] = torch.clamp(
                actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose1_position"] = torch.clamp(
                actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
            actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
            actions["pose0_rotation"] = torch.clamp(
                actions["pose0_rotation"], min=-1, max=1
            )
            actions["pose1_rotation"] = torch.clamp(
                actions["pose1_rotation"], min=-1, max=1
            )
            actions = {k: v.cpu().numpy() for k, v in actions.items()}
            actions = any_slice(actions, np.s_[0, 0])
            # 动作是在下面这行被执行的，todo: 看actions在step中是如何被使用的
            obs, _, done, info = env.step(actions)
            elapsed_steps += 1
            if done:
                break


def prepare_prompt(*, prompt: str, prompt_assets: dict, views: list[str]):
    """
    输入
    prompt是个str:'Put the {dragged_obj_1} into the {base_obj}.'
    prompt_assets: 字典，里面有两个元素'base_obj'和'dragged_obj_1'.
            每个元素里面又有三个元素，分别是
                    'rgb': 里面是三通道的图，分别是'front'和'top'
                    'segm': 里面是单通道的图'front'和'top',对于分割图，物体是0背景是255. 和字典'obj_info'
                            'obj_info': 例子{'obj_id': 0, 'obj_name': 'container', 'obj_color': 'purple'} 和 {'obj_id': 0, 'obj_name': 'block', 'obj_color': 'red swirl'}
                    'placeholder_type': 'object'
    views: 列表['front', 'top']
    ---
    输出
    raw_prompt_token_type: list1 [[0, 0, 1, 0, 0, 1, 0, 0, 0]]
    word_batch: tensor([5306,    8,  139,    8,    3,    5,    1])
            word_batch是个list，里面append的是token，token来源是filled_prompt，
    image_batch: 字典，存储一些图像的tensor
    """
    views = sorted(views)  # ['front', 'top']
    encoding = tokenizer.encode(prompt, add_special_tokens=True)  # {list:9} '_Put','_the','{dragged_obj_1}','_into','_the','{base_obj}',''_'','.','</s>'
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens  # prompt_ids:[5306, 8, 32104, 139, 8, 32100, 3, 5, 1], prompt_tokens和上面的encoding一致
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:  # PLACEHOLDERS是个list，列表中的内容是一大堆的{}包进去的内容
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]  # asset中有'rgb'和'segm'，'segm'中有'front','top','obj_info'，其中'obj_info':{'obj_id': 0, 'obj_name': 'block', 'obj_color': 'red swirl'}
            obj_info = asset["segm"]["obj_info"]  # {'obj_id': 0, 'obj_name': 'block', 'obj_color': 'red swirl'}
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                objects = [obj_info["obj_id"]]
            elif placeholder_type == "scene":
                objects = [each_info["obj_id"] for each_info in obj_info]
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    """
                    可视化检查
                    """
                    # -- bkx debug --------------------------------------------
                    from matplotlib import pyplot as plt
                    plt.figure()
                    plt.title('segm_this_view')
                    plt.imshow(segm_this_view, cmap='jet')
                    plt.show()
                    # -- bkx debug --------------------------------------------

                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    cropped_img = rgb_this_view[:, ymin: ymax + 1, xmin: xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch


def prepare_obs(
        *,
        obs: dict,
        rgb_dict: dict | None = None,
        meta: dict,
):
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views = sorted(rgb_dict.keys())
    assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = list(meta["obj_id_to_info"].keys())

    L_obs = get_batch_size(obs)

    obs_list = {
        "ee": obs["ee"],
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for l in range(L_obs):
        rgb_dict_this_step = any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step = any_slice(segm_dict, np.s_[l])
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            segm_this_view = segm_dict_this_step[view]
            bboxes = []
            cropped_imgs = []
            n_pad = 0
            for obj_id in objects:
                ys, xs = np.nonzero(segm_this_view == obj_id)
                if len(xs) < 2 or len(ys) < 2:
                    n_pad += 1
                    continue
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                cropped_img = rgb_this_view[:, ymin: ymax + 1, xmin: xmax + 1]
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img)
            bboxes = np.asarray(bboxes)
            cropped_imgs = np.asarray(cropped_imgs)
            mask = np.ones(len(bboxes), dtype=bool)
            if n_pad > 0:
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, 32, 32),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )

    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs = any_transpose_first_two_axes(obs)
    return obs


class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )


class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


if __name__ == "__main__":
    # python3 scripts/example.py --ckpt='/data/net/dl_data/ProjectDatasets_bkx/VIMA_pretrained_models/20M.ckpt'
    #                            --device='cpu'
    #                            --partition='placement_generalization'
    #                            --task='visual_manipulation'
    arg = argparse.ArgumentParser()
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str,
                     default='/data/net/dl_data/ProjectDatasets_bkx/VIMA_pretrained_models/20M.ckpt')
    arg.add_argument("--device", default='cpu')
    arg = arg.parse_args()
    main(arg)
