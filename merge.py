import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Set
import safetensors.torch
import torch
from . import merge_methods
from .merge_utils import WeightClass
from .merge_rebasin import (
	apply_permutation,
	update_model_a,
	weight_matching,
)
from .merge_PermSpec import sdunet_permutation_spec
from .merge_PermSpec_SDXL import sdxl_permutation_spec

from tqdm import tqdm

import comfy.utils
import comfy.model_management

MAX_TOKENS = 77


KEY_POSITION_IDS = ".".join(
	[
		"cond_stage_model",
		"transformer",
		"text_model",
		"embeddings",
		"position_ids",
	]
)


def fix_clip(model: Dict) -> Dict:
	if KEY_POSITION_IDS in model.keys():
		model[KEY_POSITION_IDS] = torch.tensor(
			[list(range(MAX_TOKENS))],
			dtype=torch.int64,
			device=model[KEY_POSITION_IDS].device,
		)

	return model


def prune_sd_model(model: Dict, keyset: Set) -> Dict:
	keys = list(model.keys())
	for k in keys:
		if (
			not k.startswith("model.diffusion_model.") # UNET 
			# and not k.startswith("first_stage_model.") # VAE
			and not k.startswith("cond_stage_model.") # CLIP
			and not k.startswith("conditioner.embedders.") # SDXL CLIP
		) or k not in keyset:
			del model[k]
	return model


def restore_sd_model(original_model: Dict, merged_model: Dict) -> Dict:
	for k in original_model:
		if k not in merged_model:
			merged_model[k] = original_model[k]
	return merged_model

def load_thetas(
	model_paths: Dict[str, os.PathLike],
	should_prune: bool,
	target_device: torch.device,
	precision: str,
) -> Dict:
	"""
	Load and process model parameters from given paths.

	Args:
	model_paths: Dictionary of model names and their file paths
	should_prune: Flag to determine if models should be pruned
	target_device: The device to load the models onto
	precision: The precision to use for the model parameters

	Returns:
	Dictionary of processed model parameters
	"""
	# Load model parameters from files
	model_params = {
		model_name: comfy.utils.load_torch_file(model_path)
		for model_name, model_path in model_paths.items()
	}

	if should_prune:
		# Find common keys across all models
		common_keys = set.intersection(*[set(model.keys()) for model in model_params.values() if len(model.keys())])
		# Prune models to keep only common parameters
		model_params = {
			model_name: prune_sd_model(model, common_keys)
			for model_name, model in model_params.items()
		}

	# Process each model's parameters
	for model_name, model in model_params.items():
		for param_name, param_tensor in model.items():
			if precision == "fp16":
				# Convert to half precision and move to target device
				model_params[model_name].update({param_name: param_tensor.to(target_device).half()})
			else:
				# Move to target device maintaining original precision
				model_params[model_name].update({param_name: param_tensor.to(target_device)})

	print("Models loaded successfully")
	return model_params

def merge_models(
	models: Dict[str, os.PathLike],
	merge_mode: str,
	precision: str = "fp16",
	weights_clip: bool = False,
	device: torch.device = None,
	work_device: torch.device = None,
	prune: bool = False,
	threads: int = 4,
	optional_model_a = None,
	optional_clip_a = None,
	optional_model_b = None,
	optional_clip_b = None,
	optional_model_c = None,
	optional_clip_c = None,
	**kwargs,
) -> Dict:
	print("Alpha:")
	print(kwargs["alpha"])
	
	if models == { }:
		thetas = { }
	else:
		thetas = load_thetas(models, prune, device, precision)
 
	if "model_a" not in thetas:
		thetas["model_a"] = {}

	if "model_b" not in thetas:
		thetas["model_b"] = {}

	if optional_model_a is not None:
		key_patches = optional_model_a.get_key_patches()
		for key in key_patches:
			if "diffusion_model." in key:
				thetas["model_a"]["model." + key] = key_patches[key][0]

	if optional_clip_a is not None:
		key_patches = optional_clip_a.get_key_patches()
		for key in key_patches:
			if "transformer." in key and "text_projection" not in key:
				thetas["model_a"][key.replace("clip_l", "cond_stage_model")] = key_patches[key][0]

	if optional_model_b is not None:
		key_patches = optional_model_b.get_key_patches()
		for key in key_patches:
			if "diffusion_model." in key:
				thetas["model_b"]["model." + key] = key_patches[key][0]

	if optional_clip_b is not None:
		key_patches = optional_clip_b.get_key_patches()
		for key in key_patches:
			if "transformer." in key and "text_projection" not in key:
				thetas["model_b"][key.replace("clip_l", "cond_stage_model")] = key_patches[key][0]

	if optional_model_c is not None:
		if "model_c" not in thetas:
			thetas["model_c"] = {}
		key_patches = optional_model_c.get_key_patches()
		for key in key_patches:
			if "diffusion_model." in key:
				thetas["model_c"]["model." + key] = key_patches[key][0]

	if optional_clip_c is not None:
		if "model_c" not in thetas:
				thetas["model_c"] = {}
		key_patches = optional_clip_c.get_key_patches()
		for key in key_patches:
			if "transformer." in key and "text_projection" not in key:
				thetas["model_c"][key.replace("clip_l", "cond_stage_model")] = key_patches[key][0]
	
	print(f'Merge start: models={models.values()} precision={precision} clip={weights_clip} prune={prune} threads={threads}')
	weight_matcher = WeightClass(thetas["model_a"], **kwargs)
	if kwargs.get("re_basin", False):
		merged = rebasin_merge(
			thetas,
			weight_matcher,
			merge_mode,
			precision=precision,
			weights_clip=weights_clip,
			iterations=kwargs.get("re_basin_iterations", 1),
			device=device,
			work_device=work_device,
			threads=threads,
		)
	else:
		merged = simple_merge(
			thetas,
			weight_matcher,
			merge_mode,
			precision=precision,
			weights_clip=weights_clip,
			device=device,
			work_device=work_device,
			threads=threads,
		)

	return fix_clip(merged)

def simple_merge(
	thetas: Dict[str, Dict],
	weight_matcher: WeightClass,
	merge_mode: str,
	precision: str = "fp16",
	weights_clip: bool = False,
	device: torch.device = None,
	work_device: torch.device = None,
	threads: int = 4,
) -> Dict:
	futures = []
	with tqdm(thetas["model_a"].keys(), desc="Merge") as progress:
		with ThreadPoolExecutor(max_workers=threads) as executor:
			for key in thetas["model_a"].keys():
				future = executor.submit(
					simple_merge_key,
					progress,
					key,
					thetas,
					weight_matcher,
					merge_mode,
					precision,
					weights_clip,
					device,
					work_device,
				)
				futures.append(future)

		for res in futures:
			res.result()

	if len(thetas["model_b"]) > 0:
		print(f'Merge update thetas: keys={len(thetas["model_b"])}')
		for key in thetas["model_b"].keys():
			if KEY_POSITION_IDS in key:
				continue
			if "model" in key and key not in thetas["model_a"].keys():
				thetas["model_a"].update({key: thetas["model_b"][key]})
				if precision == "fp16":
					thetas["model_a"].update({key: thetas["model_a"][key].half()})

	return fix_clip(thetas["model_a"])


def rebasin_merge(
	thetas: Dict[str, os.PathLike],
	weight_matcher: WeightClass,
	merge_mode: str,
	precision: str = "fp16",
	weights_clip: bool = False,
	iterations: int = 1,
	device: torch.device = None,
	work_device: torch.device = None,
	threads: int = 1,
):
	# not sure how this does when 3 models are involved...
	model_a = thetas["model_a"]
	if weight_matcher.SDXL:
		perm_spec = sdxl_permutation_spec()
	else:
		perm_spec = sdunet_permutation_spec()

	for it in range(iterations):
		print(f"rebasin: iteration={it+1}")
		weight_matcher.set_it(it)

		# normal block merge we already know and love
		thetas["model_a"] = simple_merge(
			thetas,
			weight_matcher,
			merge_mode,
			precision,
			False,
			device,
			work_device,
			threads,
		)

		# find permutations
		perm_1, y = weight_matching(
			perm_spec,
			model_a,
			thetas["model_a"],
			max_iter=it,
			init_perm=None,
			usefp16=precision == "fp16",
			device=device,
		)
		thetas["model_a"] = apply_permutation(perm_spec, perm_1, thetas["model_a"])

		perm_2, z = weight_matching(
			perm_spec,
			thetas["model_b"],
			thetas["model_a"],
			max_iter=it,
			init_perm=None,
			usefp16=precision == "fp16",
			device=device,
		)

		new_alpha = torch.nn.functional.normalize(
			torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0
		).tolist()[0]
		thetas["model_a"] = update_model_a(
			perm_spec, perm_2, thetas["model_a"], new_alpha
		)

	if weights_clip:
		clip_thetas = thetas.copy()
		clip_thetas["model_a"] = model_a
		thetas["model_a"] = clip_weights(thetas, thetas["model_a"])

	return thetas["model_a"]


def simple_merge_key(progress, key, thetas, *args, **kwargs):
	with merge_key_context(key, thetas, *args, **kwargs) as result:
		if result is not None:
			thetas["model_a"].update({key: result.detach().clone()})
	progress.update(1)


def merge_key(  # pylint: disable=inconsistent-return-statements
	key: str,
	thetas: Dict,
	weight_matcher: WeightClass,
	merge_mode: str,
	precision: str = "fp16",
	weights_clip: bool = False,
	device: torch.device = None,
	work_device: torch.device = None,
) -> Optional[Tuple[str, Dict]]:
	if work_device is None:
		work_device = device

	if KEY_POSITION_IDS in key:
		return

	for theta in thetas.values():
		if key not in theta.keys():
			return thetas["model_a"][key]

		current_bases = weight_matcher(key)
		try:
			merge_method = getattr(merge_methods, merge_mode)
		except AttributeError as e:
			raise ValueError(f"{merge_mode} not implemented, aborting merge!") from e

		merge_args = get_merge_method_args(current_bases, thetas, key, work_device)

		# dealing with pix2pix and inpainting models
		if (a_size := merge_args["a"].size()) != (b_size := merge_args["b"].size()):
			if a_size[1] > b_size[1]:
				merged_key = merge_args["a"]
			else:
				merged_key = merge_args["b"]
		else:
			merged_key = merge_method(**merge_args).to(device)

		if weights_clip:
			merged_key = clip_weights_key(thetas, merged_key, key)

		if precision == "fp16":
			merged_key = merged_key.half()

		return merged_key


def clip_weights(thetas, merged):
	for k in thetas["model_a"].keys():
		if k in thetas["model_b"].keys():
			merged.update({k: clip_weights_key(thetas, merged[k], k)})
	return merged

def clip_weights_key(thetas, merged_weights, key):
	# Determine the device of the merged_weights
	device = merged_weights.device

	# Move all tensors to the same device
	t0 = thetas["model_a"][key].to(device)
	t1 = thetas["model_b"][key].to(device)
	
	maximums = torch.maximum(t0, t1)
	minimums = torch.minimum(t0, t1)
	
	return torch.minimum(torch.maximum(merged_weights, minimums), maximums)

@contextmanager
def merge_key_context(*args, **kwargs):
	result = merge_key(*args, **kwargs)
	try:
		yield result
	finally:
		if result is not None:
			del result


def get_merge_method_args(
	current_bases: Dict,
	thetas: Dict,
	key: str,
	work_device: torch.device,
) -> Dict:
	merge_method_args = {
		"a": thetas["model_a"][key].to(work_device),
		"b": thetas["model_b"][key].to(work_device),
		**current_bases,
	}

	if "model_c" in thetas:
		merge_method_args["c"] = thetas["model_c"][key].to(work_device)

	return merge_method_args
