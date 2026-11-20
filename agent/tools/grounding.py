"""
grounding.py — Corrected version of grounding.py with all bugs fixed.
"""

import json
import os

import json_repair
import torch
import re
import glob
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from agent.prompts import (
    QUERY_PARSE_SYSTEM_PROMPT, SCENE_FILTER_SYSTEM_PROMPT,
    VLM_SCORE_SYSTEM_PROMPT, EXPANSION_SYSTEM_PROMPT
)
from agent.tools.base import BaseTool
from mm_utils.utils import *
from pcd.pcd_loader import load_o3d_pcd
from agent.tools.sub_tools import (
    segment_mask, segment_mask_with_point, get_random_max_index,
    vlm_identify_id, get_image_with_segment_and_marker,
    adjust_frame_index, generate_point_cloud,
    load_info_file, load_extrinsics,
)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


# ==============================================================================
# Phase 1 tools (steps 1-7) — unchanged from grounding.py
# ==============================================================================

class QueryParseArgs(BaseModel):
    query: str = Field(..., description="The natural language description of the target object.")

class QueryParseTool(BaseTool):
    name = "query_parse"
    description = "Parse the natural language query into a string with structured JSON format, including keys like: target_class, attributes, conditions and scene_feature."
    return_description = "The parsed JSON string containing keys: target_class, attributes, conditions, and scene_feature."
    args_schema = QueryParseArgs

    def run(self, query: str) -> str:
        print('\nThe raw query: ', query)
        sys_prompt = QUERY_PARSE_SYSTEM_PROMPT
        user_prompt = f"Query: {query}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        parsed_query = self.context.client.response(messages)
        match = re.search(r'\{.*\}', parsed_query, re.DOTALL)
        parsed_query = match.group()
        print('Success! The Analysed Query: ', parsed_query)
        return parsed_query


class ReadImageFilesArgs(BaseModel):
    scene_id: str = Field(..., description="The unique ID of the scene (e.g., 'scene0095_00').")

class ReadImageFilesTool(BaseTool):
    name = "read_image_files"
    description = "Scan the directory for a specific scene and save all image file paths to a JSON file in the cache directory."
    return_description = "The absolute path to the saved JSON file containing image paths."
    args_schema = ReadImageFilesArgs

    def run(self, scene_id: str) -> str:
        scene_dir = os.path.join(self.context.scannet_video_path, scene_id)
        image_files = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
        file_name_path = os.path.join(self.context.cache_dir, 'image_files.json')
        save_json(image_files, file_name_path)
        return str(file_name_path)


class MasksFilterArgs(BaseModel):
    image_files_path: str = Field(..., description="The path to the JSON file containing the list of image file paths (usually the output of 'read_image_files').")
    parsed_query: str = Field(..., description="The structured JSON string (usually the output of 'query_parse').")
    threshold: float = Field(..., description="Confidence threshold for mask generation (default 0.5). Lower this if no masks are found.")

class MasksFilterTool(BaseTool):
    name = "masks_filter"
    description = "Filter the image list using SAM. It reads a JSON file of image paths, detects objects matching the 'target_class' in the parsed query, and saves the filtered list to a new JSON file."
    return_description = "A tuple containing: 'path_to_filtered_images_json, count_of_filtered_images'."
    args_schema = MasksFilterArgs

    def run(self, image_files_path: str, parsed_query: str, threshold: float = 0.5) -> str:
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        current_filtered_images = load_json(image_files_path)
        mask_filtered_images = []

        for idx, file in enumerate(tqdm(current_filtered_images)):
            raw_image = Image.open(file).convert("RGB")
            mask, _ = segment_mask(
                self.context.sam_processor, self.context.sam_model,
                raw_image, parsed_query['target_class'], threshold=threshold
            )
            if mask is not None:
                mask_filtered_images.append(file)

        max_keep = 100
        if len(mask_filtered_images) > max_keep:
            indices = np.linspace(0, len(mask_filtered_images) - 1, max_keep, dtype=int)
            mask_filtered_images = [mask_filtered_images[i] for i in indices]

        file_name_path = os.path.join(self.context.cache_dir, 'mask_filtered_image_files.json')
        save_json(mask_filtered_images, file_name_path)
        return f"{file_name_path}, {len(mask_filtered_images)}"


class VlmFilterArgs(BaseModel):
    image_files_path: str = Field(..., description="The path to the JSON file containing the list of image file paths (usually the output of 'masks_filter' or 'read_image_files').")
    parsed_query: str = Field(..., description="The structured JSON string (output of 'query_parse').")

class VlmFilterTool(BaseTool):
    name = "vlm_filter"
    description = 'Filter the image list using a Vision-Language Model based on the scene description.'
    return_description = "A tuple containing: 'path_to_filtered_images_json, count_of_filtered_images'."
    args_schema = VlmFilterArgs

    def run(self, image_files_path: str, parsed_query: str) -> str:
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        current_filtered_images = load_json(image_files_path)
        sys_prompt = SCENE_FILTER_SYSTEM_PROMPT
        user_prompt = f"""The scene description is: \"{parsed_query['scene_feature']}\".
        Does the image strictly satisfy all the constraints in this description?
        Please directly answer yes or no."""

        vlm_filtered_images = []
        for idx, file in enumerate(tqdm(current_filtered_images)):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": file},
                    {"type": "text", "text": user_prompt},
                ]}
            ]
            output = self.context.client.response(messages)
            if 'yes' in output.lower():
                vlm_filtered_images.append(file)

        file_name_path = os.path.join(self.context.cache_dir, 'vlm_filtered_image_files.json')
        save_json(vlm_filtered_images, file_name_path)
        return f"{file_name_path}, {len(vlm_filtered_images)}"


class VlmScoreArgs(BaseModel):
    image_files_path: str = Field(..., description="The path to the JSON file containing the list of image file paths (usually the output of 'vlm_filter').")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string (output of 'query_parse').")

class VlmScoreTool(BaseTool):
    name = "vlm_score"
    description = "Score and rank the remaining images based on how well they match the query details."
    return_description = "A tuple containing: 'path_to_scores_json, path_to_ranked_image_files_json'."
    args_schema = VlmScoreArgs

    def run(self, image_files_path: str, query: str, parsed_query: str) -> str:
        sys_prompt = VLM_SCORE_SYSTEM_PROMPT
        current_filtered_images = load_json(image_files_path)
        user_prompt = f"""Here is the query data for the current image:
        1. **Query**: "{query}"
        2. **Parsed Query**:
        {str(parsed_query)}

        Remember, pay attention to the existence of the reference objects. Do not give a high scroe if reference objects missing.

        Please score each image based on the system instructions.
        """
        print('\nThe VLM begins scoring each filtered frame...')
        scores = []
        for idx, file in enumerate(tqdm(current_filtered_images)):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": file},
                    {"type": "text", "text": user_prompt},
                ]}
            ]
            output = self.context.client.response(messages)
            res = json.loads(output)
            scores.append(res['score'])
        print('Success! The scores are:', scores)
        if max(scores) == 0:
            print('WARNING: The max score is 0')

        print('Rank images according to the scores...')
        sorted_pairs = sorted(
            zip(current_filtered_images, scores),
            key=lambda x: (x[1], random.random()),
            reverse=True,
        )
        current_filtered_images = [img for img, _ in sorted_pairs]
        ranked_image_file_path = os.path.join(self.context.cache_dir, 'vlm_ranked_image_files.json')
        save_json(current_filtered_images, ranked_image_file_path)

        scores = [score for _, score in sorted_pairs]
        score_file_name_path = os.path.join(self.context.cache_dir, 'vlm_scores.json')
        save_json(scores, score_file_name_path)

        return score_file_name_path, ranked_image_file_path


class ArgmaxImageAndSegIdArgs(BaseModel):
    scores_path: str = Field(..., description="The path to the JSON file containing image scores (output of 'vlm_score').")
    image_files_path: str = Field(..., description="The path to the JSON file containing sorted image paths (output of 'vlm_score').")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string (usually the output of 'query_parse').")
    threshold: float = Field(..., description="Segmentation confidence threshold.")

class ArgmaxImageAndSegIdTool(BaseTool):
    name = "argmax_image_and_seg_id"
    description = 'Select the best candidate image and identify the specific target object ID within it.'
    return_description = "A tuple containing (reference_image_file_path, target_object_id, reference_image_mask_results_path)."
    args_schema = ArgmaxImageAndSegIdArgs

    def run(self, scores_path: str, image_files_path: str, query: str, parsed_query: str, threshold: float = 0.5):
        if isinstance(image_files_path, str):
            current_filtered_images = load_json(image_files_path)
        else:
            current_filtered_images = image_files_path

        if isinstance(scores_path, str):
            scores = load_json(scores_path)
        else:
            scores = scores_path

        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)
        else:
            parsed_query = parsed_query

        temp_scores = scores.copy()
        target_id = -1
        max_retry_count = len(scores)
        current_try = 0

        while target_id == -1 and current_try < max_retry_count:
            current_try += 1
            if max(temp_scores) == -float('inf'):
                print("No valid images left to check.")
                return None, None, None

            max_index = get_random_max_index(temp_scores)
            print(f"\nFind the image with max score. Attempt {current_try}: Trying image index {max_index}...")
            reference_image_file = current_filtered_images[max_index]
            print(f"The file name is {reference_image_file}")
            reference_image = Image.open(reference_image_file).convert("RGB")
            image_with_mask, image_with_mask_box_id, results = get_image_with_segment_and_marker(
                self.context.sam_processor, self.context.sam_model, reference_image,
                parsed_query['target_class'], threshold=float(threshold),
                draw_mask=False, draw_id=True,
            )
            reference_image_with_mask_box_id_file = os.path.join(
                self.context.cache_dir, 'reference_image_with_mask_box_id.png')
            image_with_mask_box_id.save(reference_image_with_mask_box_id_file)
            print(f'Saved image with masks, boxes and ids, to {reference_image_with_mask_box_id_file}')

            print(f"\nThe VLM is trying to identify the target object...")
            target_id = vlm_identify_id(reference_image_with_mask_box_id_file, self.context.client, query, parsed_query)
            print(f"VLM Parsed ID: {target_id}")

            if target_id == -1:
                print(f"Target not found in image index {max_index}. Marking as failed and retrying...")
                temp_scores[max_index] = -float('inf')
            else:
                print(f"Success! Target found at image index {max_index} with ID {target_id}")

        reference_image_mask_results_path = os.path.join(
            self.context.cache_dir, 'reference_image_mask_results.pth')
        torch.save(results, reference_image_mask_results_path)
        return reference_image_file, target_id, reference_image_mask_results_path


class SegmentTargetInReferenceArgs(BaseModel):
    reference_image_file: str = Field(..., description="Path to the reference image file (output of 'argmax_image_and_seg_id').")
    target_id: int = Field(..., description="The specific integer ID of the target object (output of 'argmax_image_and_seg_id').")
    reference_image_mask_results_path: str = Field(..., description="Path to the saved mask results file (output of 'argmax_image_and_seg_id').")
    parsed_query: str = Field(..., description="The structured JSON string (usually the output of 'query_parse').")
    threshold: float = Field(..., description="Segmentation confidence threshold.")

class SegmentTargetInReferenceTool(BaseTool):
    name = "segment_target_in_reference"
    description = "Isolate the specific target object in the reference image."
    return_description = "The file path to the visualized reference image with the target object boxed/masked."
    args_schema = SegmentTargetInReferenceArgs

    def run(self, reference_image_file: str, target_id: int,
            reference_image_mask_results_path: str, parsed_query: str,
            threshold: float = 0.5) -> str:
        print("\nSegment only target object in the referecne image...")
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)
        target_id = int(target_id)

        reference_image = Image.open(reference_image_file).convert("RGB")
        reference_image_mask_results = torch.load(reference_image_mask_results_path,
                                                   weights_only=True)

        if target_id + 1 > reference_image_mask_results['masks'].shape[0]:
            print(f'Current target id {target_id} is out of the index of mask results, reset to 0')
            target_id = 0

        ref_results = {
            'masks': reference_image_mask_results['masks'][target_id].unsqueeze(0),
            'boxes': reference_image_mask_results['boxes'][target_id].unsqueeze(0),
            'scores': reference_image_mask_results['scores'][target_id].unsqueeze(0),
        }
        image_with_target_mask, image_with_target_mask_box, results = get_image_with_segment_and_marker(
            self.context.sam_processor, self.context.sam_model, reference_image,
            parsed_query['target_class'], threshold=float(threshold),
            results=ref_results, draw_id=False, draw_mask=False,
        )
        reference_image_with_target_mask_box_file = os.path.join(
            self.context.cache_dir, 'reference_image_with_target_mask_box.png')
        image_with_target_mask_box.save(reference_image_with_target_mask_box_file)
        return reference_image_with_target_mask_box_file


# ==============================================================================
# V1 Temporal Expansion (original VLM-based, no optical flow)
# ==============================================================================

class VlmFrameExpansionArgs(BaseModel):
    reference_image_file: str = Field(..., description="Path to the raw reference image file.")
    reference_image_with_target_mask_box_file: str = Field(..., description="Path to the reference image with the target masked/boxed.")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string.")
    max_filtered_num: int = Field(..., description="Maximum number of frames to expand/track (default 16).")
    threshold: float = Field(..., description="Segmentation confidence threshold.")

class VlmFrameExpansionTool(BaseTool):
    name = "vlm_frame_expansion"
    description = "Expand the target object search temporally from a reference frame using VLM verification and SAM masks."
    return_description = "A tuple containing: 'path_to_expanded_image_files_json, count_of_expanded_images'."
    args_schema = VlmFrameExpansionArgs

    def run(self, reference_image_file: str, reference_image_with_target_mask_box_file: str,
            query: str, parsed_query: str, max_filtered_num: int = 16,
            threshold: float = 0.5):
        print("\n\nStart expanding frames from the reference...")
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        reference_video_clip = [reference_image_with_target_mask_box_file]
        candidate_image_files = [reference_image_file]

        sys_prompt = EXPANSION_SYSTEM_PROMPT
        user_prompt = f"""
        <TaskInstruction>
        Compare the candidate image with the reference video context.
        Strictly follow the criteria to decide if the target is present.
        </TaskInstruction>

        <Instructions>
        1. Analyze the appearance of the object inside the bbox in the Reference Video.
        2. Scan the Candidate Image for this specific object.
        3. Output the decision in JSON.
        </Instructions>
        """

        print(f"Your reference file is {reference_image_file}")
        output_dir = os.path.join(self.context.cache_dir, 'expansion')
        os.makedirs(output_dir, exist_ok=True)

        for delta in [1, -1]:
            direction_str = "right" if delta == 1 else "left"
            print(f"\nThen, from reference to {direction_str}")
            start_image_file = reference_image_file

            while True:
                current_image_file = adjust_frame_index(start_image_file, delta,
                                                        min_idx=0, max_idx=299)
                if current_image_file is None:
                    break

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "### REFERENCE CONTEXT (Video Clip with BBox)"},
                        {"type": "video", "video": reference_video_clip, 'sample_fps': 1},
                        {"type": "text", "text": "### CANDIDATE IMAGE (To Evaluate)"},
                        {"type": "image", "image": current_image_file},
                        {"type": "text", "text": user_prompt},
                    ]}
                ]
                output = self.context.client.response(messages)
                print(current_image_file, output)

                if 'yes' in output.lower():
                    current_image = Image.open(current_image_file).convert("RGB")
                    _, current_image_with_mask_box_id, results = get_image_with_segment_and_marker(
                        self.context.sam_processor, self.context.sam_model,
                        current_image, parsed_query['target_class'],
                        threshold=float(threshold), draw_mask=False, draw_id=True,
                    )

                    if results['masks'].shape[0] == 0:
                        print(f"No [{parsed_query['target_class']}] is found by SAM in {current_image_file}, so skip this image")
                        break

                    current_image_with_mask_box_id_file = os.path.join(
                        output_dir,
                        f"expansion_current_image_with_mask_box_id_{current_image_file.split('/')[-1]}")
                    current_image_with_mask_box_id.save(current_image_with_mask_box_id_file)

                    target_id = vlm_identify_id(current_image_with_mask_box_id_file,
                                                self.context.client, query, parsed_query)
                    if target_id == -1:
                        print(f"target_id: {target_id}, so skip this image")
                        break

                    ref_results = {
                        'masks': results['masks'][target_id].unsqueeze(0),
                        'boxes': results['boxes'][target_id].unsqueeze(0),
                        'scores': results['scores'][target_id].unsqueeze(0),
                    }
                    _, current_image_with_target_mask_box, _ = get_image_with_segment_and_marker(
                        self.context.sam_processor, self.context.sam_model,
                        current_image, parsed_query['target_class'],
                        threshold=float(threshold), results=ref_results,
                        draw_id=False, draw_mask=False,
                    )
                    current_image_with_target_mask_box_file = os.path.join(
                        output_dir,
                        f"expansion_current_image_with_target_mask_box_{current_image_file.split('/')[-1]}")
                    current_image_with_target_mask_box.save(current_image_with_target_mask_box_file)

                    if delta == 1:
                        candidate_image_files = candidate_image_files + [current_image_file]
                        reference_video_clip = reference_video_clip + [current_image_with_target_mask_box_file]
                    else:
                        candidate_image_files = [current_image_file] + candidate_image_files
                        reference_video_clip = [current_image_with_target_mask_box_file] + reference_video_clip
                else:
                    break

                if len(candidate_image_files) >= max_filtered_num:
                    break
                start_image_file = current_image_file

        save_json(reference_video_clip,
                  os.path.join(self.context.cache_dir, "expanded_image_files_with_box.json"))
        file_name_path = os.path.join(self.context.cache_dir, "expanded_image_files.json")
        save_json(candidate_image_files, file_name_path)
        return file_name_path, len(candidate_image_files)


# ==============================================================================
# V1 Expand From Secondary View
# ==============================================================================

class ExpandFromSecondaryViewArgs(BaseModel):
    expanded_image_file_path: str = Field(..., description="Path to the current JSON list of expanded images.")
    scores_path: str = Field(..., description="Path to the VLM scores JSON file.")
    ranked_image_file_path: str = Field(..., description="Path to the ranked image files JSON list.")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string.")
    max_filtered_num: int = Field(..., description="Maximum number of frames to expand/track (default 16).")
    threshold: float = Field(..., description="Segmentation confidence threshold.")

class ExpandFromSecondaryViewTool(BaseTool):
    name = "expand_from_secondary_view"
    description = "Iteratively improve view coverage by searching for new high-scoring reference frames."
    return_description = "A tuple containing: 'path_to_updated_image_files_json, total_count_of_images'."
    args_schema = ExpandFromSecondaryViewArgs

    def run(self, expanded_image_file_path: str, scores_path: str,
            ranked_image_file_path: str, query: str, parsed_query: str,
            max_filtered_num: int = 16, threshold: float = 0.5):
        save_json(load_json(expanded_image_file_path),
                  os.path.join(self.context.cache_dir, 'old_expanded_image_files.json'))
        save_json(load_json(scores_path),
                  os.path.join(self.context.cache_dir, 'old_vlm_scores.json'))

        def get_another_reference(image_files_path, scores_path, ranked_image_file_path):
            expanded_image_files = load_json(image_files_path)
            scores = load_json(scores_path)
            ranked_image_files = load_json(ranked_image_file_path)

            max_score_frames = []
            for i in range(len(scores)):
                if scores[i] == max(scores) or scores[i] >= 4:
                    max_score_frames.append(ranked_image_files[i])
            candidate_frames = [img for img in max_score_frames
                                if img not in expanded_image_files]
            if len(candidate_frames) == 0:
                print("No other candidate frames")
                return None

            def get_frame_index(filename):
                basename = os.path.basename(filename)
                match = re.findall(r'(\d+)', basename)
                return int(match[-1]) if match else 0

            expanded_indices = [get_frame_index(f) for f in expanded_image_files]
            best_candidate = None
            max_min_distance = -1

            for candidate in candidate_frames:
                cand_idx = get_frame_index(candidate)
                min_dist_to_set = min([abs(cand_idx - exist_idx)
                                       for exist_idx in expanded_indices])
                if min_dist_to_set > max_min_distance:
                    max_min_distance = min_dist_to_set
                    best_candidate = candidate

            if best_candidate:
                print(f"Found new reference frame: {best_candidate} (Distance: {max_min_distance})")
                return best_candidate
            else:
                return None

        expanded_image_files = load_json(expanded_image_file_path)
        another_reference_file = get_another_reference(
            expanded_image_file_path, scores_path, ranked_image_file_path)

        if another_reference_file is None:
            return expanded_image_file_path, "No other candidate frames, use the original expanded_image_files"

        argmax_image_and_seg_id_tool = ArgmaxImageAndSegIdTool(self.context)
        reference_image_file, target_id, reference_image_mask_results_path = \
            argmax_image_and_seg_id_tool.run(
                [5], [another_reference_file], query, parsed_query, threshold)
        if reference_image_file is None:
            return expanded_image_file_path, "No other candidate frames, use the original expanded_image_files"

        segment_target_in_reference_tool = SegmentTargetInReferenceTool(self.context)
        reference_image_with_target_mask_box_file = segment_target_in_reference_tool.run(
            reference_image_file, target_id, reference_image_mask_results_path,
            parsed_query, threshold)

        vlm_frame_expansion_tool = VlmFrameExpansionTool(self.context)
        another_expanded_image_file_path, num = vlm_frame_expansion_tool.run(
            reference_image_file, reference_image_with_target_mask_box_file,
            query, parsed_query, max_filtered_num, threshold)

        new_expanded_image_files = expanded_image_files + load_json(another_expanded_image_file_path)
        file_name_path = os.path.join(self.context.cache_dir, "expanded_image_files.json")
        save_json(new_expanded_image_files, file_name_path)
        return file_name_path, len(new_expanded_image_files)


# ==============================================================================
# Phase 2-3 tools (steps 9-14) — unchanged
# ==============================================================================

class SegmentAllTargetObjectArgs(BaseModel):
    image_files_path: str = Field(..., description="Path to the JSON list of verified candidate images.")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string.")
    threshold: float = Field(..., description="Segmentation confidence threshold.")

class SegmentAllTargetObjectTool(BaseTool):
    name = "segment_all_target_object"
    description = "Perform final segmentation on all candidate images for reconstruction."
    return_description = "A tuple containing: 'path_to_final_images_json, path_to_final_masks_pth'."
    args_schema = SegmentAllTargetObjectArgs

    def run(self, image_files_path: str, query: str, parsed_query: str,
            threshold: float = 0.5):
        candidate_image_files = load_json(image_files_path)
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        final_masks = []
        final_images = []

        output_dir = os.path.join(self.context.cache_dir, 'candidate')
        os.makedirs(output_dir, exist_ok=True)

        for idx, image_file in tqdm(enumerate(candidate_image_files)):
            target_image = Image.open(image_file).convert("RGB")
            _, candidate_image_with_mask_box_id, results = get_image_with_segment_and_marker(
                self.context.sam_processor, self.context.sam_model,
                target_image, parsed_query['target_class'],
                threshold=float(threshold), draw_mask=False, draw_id=True,
            )
            candidate_image_with_mask_box_id_file = os.path.join(
                output_dir, f"candidate_image_with_mask_box_id_{idx}.png")
            candidate_image_with_mask_box_id.save(candidate_image_with_mask_box_id_file)

            target_id = vlm_identify_id(candidate_image_with_mask_box_id_file,
                                        self.context.client, query, parsed_query)
            print(f"For image {candidate_image_with_mask_box_id_file}, VLM Parsed ID: {target_id}. Raw image: {image_file}")

            if target_id == -1:
                print("Failed this image")
                continue
            else:
                target_results = {
                    'masks': results['masks'][target_id].unsqueeze(0),
                    'boxes': results['boxes'][target_id].unsqueeze(0),
                    'scores': results['scores'][target_id].unsqueeze(0),
                }
                final_masks.append(target_results['masks'][0])
                final_images.append(image_file)

                _, candidate_image_with_target_mask_box, results = get_image_with_segment_and_marker(
                    self.context.sam_processor, self.context.sam_model,
                    target_image, parsed_query['target_class'],
                    threshold=float(threshold), results=target_results,
                    draw_mask=True, draw_id=False,
                )
                candidate_image_with_target_mask_box.save(
                    os.path.join(output_dir, f'candidate_image_with_target_mask_box_{idx}.png'))

        final_images_file = os.path.join(self.context.cache_dir, 'final_images.json')
        final_masks_file = os.path.join(self.context.cache_dir, 'final_masks.pth')
        save_json(final_images, final_images_file)
        torch.save(final_masks, final_masks_file)

        return final_images_file, final_masks_file


class ReconstructPointCloudArgs(BaseModel):
    final_images_file: str = Field(..., description="Path to the JSON list of final images.")
    final_masks_file: str = Field(..., description="Path to the saved masks .pth file.")
    scene_id: str = Field(..., description="Unique scene ID.")

class ReconstructPointCloudTool(BaseTool):
    name = "reconstruct_point_cloud"
    description = "Generate the final 3D point cloud from the segmented target images."
    return_description = "The absolute file path to the generated PLY point cloud file."
    args_schema = ReconstructPointCloudArgs

    def run(self, final_images_file: str, final_masks_file: str, scene_id: str) -> str:
        print("Reconstruct Object...")
        final_images = load_json(final_images_file)
        final_masks = torch.load(final_masks_file, weights_only=True)
        output_file = os.path.join(self.context.cache_dir, "pred_pcd.ply")

        rgb_files = []
        depth_files = []
        masks = []

        scene_dir = os.path.join(self.context.scannet_video_path, scene_id)
        info_path = os.path.join(self.context.scannet_info_path, scene_id,
                                 scene_id + '.txt')
        for i in range(len(final_images)):
            rgb_files.append(final_images[i])
            depth_files.append(final_images[i].replace('.jpg', '.png'))
            masks.append(final_masks[i].cpu().numpy())

        pcd = generate_point_cloud(
            scene_dir=scene_dir, info_path=info_path,
            rgb_files=rgb_files, depth_files=depth_files,
            num_frames_to_sample=300, target_n=100000,
            masks=masks, indices=None
        )
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Saved point cloud to {output_file}")
        return output_file


class CentroidCompleteArgs(BaseModel):
    mask_filtered_images_file: str = Field(..., description="Path to the JSON list of mask filtered images.")
    pcd_path: str = Field(..., description="File path to the point cloud (e.g., 'pred_pcd.ply').")
    parsed_query: str = Field(..., description="The structured JSON string.")
    scene_id: str = Field(..., description="Unique scene ID.")
    threshold: float = Field(..., description="Confidence threshold for mask generation (default 0.5).")

class CentroidCompleteTool(BaseTool):
    name = "centroid_complete"
    description = "Uses 3D geometric projection and depth-based occlusion checks to track the target across video frames."
    return_description = "A tuple containing: 'path_to_centroid_final_images_json, path_to_centroid_final_masks_pth'."
    args_schema = CentroidCompleteArgs

    def run(self, mask_filtered_images_file: str, pcd_path: str, parsed_query: str,
            scene_id: str, threshold: float) -> str:

        info_path = os.path.join(self.context.scannet_info_path, scene_id,
                                 scene_id + '.txt')
        scene_dir = os.path.join(self.context.scannet_video_path, scene_id)
        info = load_info_file(info_path)
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        fx_d, fy_d = float(info['fx_depth']), float(info['fy_depth'])
        cx_d, cy_d = float(info['mx_depth']), float(info['my_depth'])
        fx_c, fy_c = float(info['fx_color']), float(info['fy_color'])
        cx_c, cy_c = float(info['mx_color']), float(info['my_color'])
        color_w, color_h = int(info['colorWidth']), int(info['colorHeight'])
        axisAlignment = info['axisAlignment'].reshape(4, 4)
        inv_axisAlignment = np.linalg.inv(axisAlignment)
        if 'colorToDepthExtrinsics' in info.keys():
            T_c2d = info['colorToDepthExtrinsics'].reshape(4, 4)
            T_d2c = np.linalg.inv(T_c2d)
        else:
            T_d2c = np.eye(4)

        print(f"\n--- 1. Calculate the 3D Centroid of current pcd: {pcd_path} ---")
        pcd = o3d.io.read_point_cloud(pcd_path)
        centroid_o3d = pcd.get_center()
        C_world = centroid_o3d

        print("\n--- 2. Retrieving Valid Frames (Geometry & Occlusion Check) ---")
        rgb_files = load_json(mask_filtered_images_file)
        retrieved_frames = []
        depth_w, depth_h = int(info['depthWidth']), int(info['depthHeight'])
        depth_tolerance = 0.4

        C_w_aligned_h = np.append(C_world, 1.0)

        for rgb_path in rgb_files:
            idx = os.path.splitext(os.path.basename(rgb_path))[0]
            extr_path = os.path.join(scene_dir, f"{idx}.txt")
            depth_path = os.path.join(scene_dir, f"{idx}.png")

            if not os.path.exists(extr_path) or not os.path.exists(depth_path):
                continue

            T_d2w_target = load_extrinsics(extr_path)
            T_w2d_target = np.linalg.inv(T_d2w_target)

            C_w_unaligned_h = inv_axisAlignment @ C_w_aligned_h
            C_d_target_h = T_w2d_target @ C_w_unaligned_h
            x_d, y_d, z_d = C_d_target_h[:3]

            C_c_target_h = T_d2c @ C_d_target_h
            x_c, y_c, z_predict = C_c_target_h[:3]

            if z_predict <= 0.1 or z_d <= 0.1:
                continue

            u_proj = int(round((x_c * fx_c / z_predict) + cx_c))
            v_proj = int(round((y_c * fy_c / z_predict) + cy_c))

            u_depth = int(round((x_d * fx_d / z_d) + cx_d))
            v_depth = int(round((y_d * fy_d / z_d) + cy_d))

            if 0 <= u_proj < color_w and 0 <= v_proj < color_h:
                if 0 <= u_depth < depth_w and 0 <= v_depth < depth_h:
                    depth_target = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    actual_z = depth_target[v_depth, u_depth] / 1000.0
                    if actual_z > 0 and actual_z < (z_predict - depth_tolerance):
                        continue
                else:
                    continue

                retrieved_frames.append({
                    "frame_id": idx,
                    "rgb_path": rgb_path,
                    "point_prompt": [[u_proj, v_proj]],
                    "z_predict": z_predict
                })

        print(f"\nRetrieval Complete! Found {len(retrieved_frames)} valid frames out of {len(rgb_files)}.")

        print("\n--- 3. Visualizing Centroid on Retrieved Frames ---")
        output_dir = os.path.join(self.context.cache_dir, 'centroid_vis')
        os.makedirs(output_dir, exist_ok=True)
        for i, frame_data in enumerate(retrieved_frames):
            rgb_path = frame_data['rgb_path']
            frame_id = frame_data['frame_id']
            u_proj, v_proj = frame_data['point_prompt'][0]
            img = cv2.imread(rgb_path)
            cv2.circle(img, (u_proj, v_proj), radius=15, color=(255, 255, 255), thickness=4)
            cv2.circle(img, (u_proj, v_proj), radius=12, color=(0, 0, 255), thickness=-1)
            cv2.drawMarker(img, (u_proj, v_proj), color=(0, 0, 0),
                          markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
            save_path = os.path.join(output_dir, f"centroid_vis_{frame_id}.jpg")
            cv2.imwrite(save_path, img)

        print("\n--- 4. Segmenting and Filtering Masks using Centroid ---")
        centroid_final_images = []
        centroid_final_masks = []
        mask_output_dir = os.path.join(self.context.cache_dir, 'centroid_vis_masks')
        os.makedirs(mask_output_dir, exist_ok=True)

        for i, frame_data in enumerate(retrieved_frames):
            rgb_path = frame_data['rgb_path']
            frame_id = frame_data['frame_id']
            u_proj, v_proj = frame_data['point_prompt'][0]

            target_image = Image.open(rgb_path).convert("RGB")
            image_np = np.array(target_image)

            _, results = segment_mask(self.context.sam_processor, self.context.sam_model,
                                      target_image, parsed_query['target_class'],
                                      threshold=threshold)

            valid_mask_found = False
            final_mask = None
            target_id = 0
            min_distance = float('inf')
            fallback_mask = None
            fallback_target_id = -1

            if results is not None and 'masks' in results and len(results['masks']) > 0:
                for j in range(results['masks'].shape[0]):
                    mask_np_current = results['masks'][j].cpu().numpy()
                    bool_mask = (mask_np_current > 0.5)

                    if v_proj < bool_mask.shape[0] and u_proj < bool_mask.shape[1]:
                        if results['masks'].shape[0] == 1:
                            final_mask = bool_mask
                            valid_mask_found = True
                            target_id = j
                            break
                        else:
                            if bool_mask[v_proj, u_proj]:
                                final_mask = bool_mask
                                valid_mask_found = True
                                target_id = j
                                break
                            else:
                                y_idx, x_idx = np.where(bool_mask)
                                if len(y_idx) > 0:
                                    distances = np.sqrt((x_idx - u_proj) ** 2 + (y_idx - v_proj) ** 2)
                                    current_min_dist = np.min(distances)
                                    if current_min_dist < min_distance:
                                        min_distance = current_min_dist
                                        fallback_mask = bool_mask
                                        fallback_target_id = j

                if not valid_mask_found and fallback_target_id != -1:
                    final_mask = fallback_mask
                    valid_mask_found = True
                    target_id = fallback_target_id
                    print(f"[{frame_id}] Fallback activated: Matched mask {target_id} at distance {min_distance:.2f}px")

            if valid_mask_found:
                overlay_image_np = image_np.copy()
                overlay_color = np.array([0, 255, 0])
                alpha = 0.5
                original_pixels = image_np[final_mask]
                blended_pixels = (1 - alpha) * original_pixels.astype(np.float32) + \
                                 alpha * overlay_color.astype(np.float32)
                overlay_image_np[final_mask] = np.clip(blended_pixels, 0, 255).astype(np.uint8)
                cv2.circle(overlay_image_np, (u_proj, v_proj), radius=6,
                          color=(255, 0, 0), thickness=-1)
                final_image = Image.fromarray(overlay_image_np)
                save_path = os.path.join(mask_output_dir, f"filtered_mask_{frame_id}.jpg")
                final_image.save(save_path)
                print(f"[{frame_id}] Target mask validated & saved.")
                centroid_final_images.append(rgb_path)
                centroid_final_masks.append(results['masks'][target_id])
            else:
                print(f"[{frame_id}] No mask contains the centroid point ({u_proj}, {v_proj}).")

        centroid_final_images_file = os.path.join(self.context.cache_dir,
                                                   'centroid_final_images.json')
        centroid_final_masks_file = os.path.join(self.context.cache_dir,
                                                  'centroid_final_masks.pth')
        save_json(centroid_final_images, centroid_final_images_file)
        torch.save(centroid_final_masks, centroid_final_masks_file)

        return centroid_final_images_file, centroid_final_masks_file


class CalculateBboxArgs(BaseModel):
    pcd_path: str = Field(..., description="File path to the point cloud (e.g., 'pred_pcd.ply').")

class CalculateBboxTool(BaseTool):
    name = "calculate_bbox"
    description = "Calculate the axis-aligned 3D bounding box of a point cloud file."
    return_description = 'A string representation of the bounding box "[cx cy cz dx dy dz]".'
    args_schema = CalculateBboxArgs

    def run(self, pcd_path: str) -> str:
        pcd = load_o3d_pcd(pcd_path)
        points = np.asarray(pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        bbox_center = (min_bound + max_bound) / 2.0
        bbox_size = max_bound - min_bound
        bbox_result = np.concatenate([bbox_center, bbox_size]).tolist()
        return str(bbox_result)


# ==============================================================================
# V2: RAFT Optical Flow + TSDF — ALL BUGS FIXED BELOW
# ==============================================================================

# --- FIX 1: RAFTFlowEstimator with keyword-argument forward call ---

class RAFTFlowEstimator:
    """Optical flow estimator using torchvision RAFT.
    """

    def __init__(self, device='cuda'):
        self.device = device
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(device).eval()
        self.transform = weights.transforms()

    def estimate_flow(self, img1, img2):
        """Compute optical flow from img1 to img2.

        Args:
            img1, img2: numpy arrays (H, W, 3) in BGR or RGB.
        Returns:
            flow: numpy array (H, W, 2)
        """

        if self.model is None:
            h, w = img1.shape[:2]
            return np.zeros((h, w, 2), dtype=np.float32)

        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)

        try:
            img1_t, img2_t = self.transform(img1_pil, img2_pil)
        except Exception as e:
            print(f"[ERROR] transform failed: {e}")
            raise e

        img1_t = img1_t.unsqueeze(0).to(self.device)
        img2_t = img2_t.unsqueeze(0).to(self.device)
        # print('img1_t=>', img1_t.shape, 'img2_t=>', img2_t.shape)   # img1_t=> torch.Size([1, 3, 968, 1296]) img2_t=> torch.Size([1, 3, 968, 1296])

        with torch.no_grad():
            flow_list = self.model(img1_t, img2_t)

        # print('flow_list[-1]=>', flow_list[-1].shape)   # flow_list[-1]=> torch.Size([1, 2, 968, 1296])
        flow = flow_list[-1][0]
        return flow.permute(1, 2, 0).cpu().numpy()


class VlmFrameExpansionArgsV2(BaseModel):
    reference_image_file: str = Field(..., description="Path to the raw reference image file.")
    reference_image_with_target_mask_box_file: str = Field(..., description="Path to the reference image with target masked/boxed.")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string.")
    max_filtered_num: int = Field(16, description="Maximum number of frames to expand/track.")
    threshold: float = Field(0.5, description="Segmentation confidence threshold.")

class VlmFrameExpansionToolV2(BaseTool):
    name = "vlm_frame_expansion"
    description = "Expand target search temporally using RAFT optical flow + full VLM verification pipeline (compatible with V1 VLM logic). Tracks object frame-by-frame, uses global reference clip VLM check + SAM refine + instance ID VLM match for high accuracy."
    return_description = "A tuple containing: 'path_to_expanded_image_files_json, count_of_expanded_images'."
    args_schema = VlmFrameExpansionArgsV2

    def __init__(self, context):
        super().__init__(context)
        self.raft = RAFTFlowEstimator()
        self.ref_annot_clip = []  # 存储带mask bbox标注图路径，对齐V1 reference_video_clip

    def warp_mask(self, mask, flow):
        """Warp a binary mask using optical flow."""
        h, w = mask.shape
        flow_t = torch.from_numpy(flow).unsqueeze(0).float()
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack([x, y], dim=-1).unsqueeze(0).float()
        grid = grid + flow_t

        grid[..., 0] = 2.0 * grid[..., 0] / (w - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (h - 1) - 1.0

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        warped_mask = F.grid_sample(
            mask_t, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        return (warped_mask[0, 0] > 0.5).numpy()

    @staticmethod
    def _get_frame_index(filename):
        basename = os.path.basename(filename)
        match = re.findall(r'(\d+)', basename)
        return int(match[-1]) if match else 0

    def _load_reference_mask(self, reference_image_file, parsed_query, threshold):
        ref_pil = Image.open(reference_image_file).convert("RGB")
        mask, results = segment_mask(
            self.context.sam_processor,
            self.context.sam_model,
            ref_pil,
            parsed_query['target_class'],
            threshold=threshold,
        )
        if mask is not None and mask.shape[0] > 0:
            best_idx = torch.argmax(results['scores']).item()
            return (mask[best_idx].cpu().numpy() > 0.5).astype(np.uint8)
        h, w = ref_pil.height, ref_pil.width
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[h//4:3*h//4, w//4:3*w//4] = 1
        print("WARNING: _load_reference_mask: SAM found no masks, using fallback center mask")
        return fallback

    def _vlm_global_check(self, cand_img_path: str, query: str, parsed_query: dict) -> bool:
        """复刻V1全局VLM校验：对比参考标注clip判断目标是否存在"""
        sys_prompt = EXPANSION_SYSTEM_PROMPT
        user_prompt = """
        <TaskInstruction>
        Compare the candidate image with the reference video context.
        Strictly follow the criteria to decide if the target is present.
        </TaskInstruction>

        <Instructions>
        1. Analyze the appearance of the object inside the bbox in the Reference Video.
        2. Scan the Candidate Image for this specific object.
        3. Output only "yes" or "no".
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": "### REFERENCE CONTEXT (Video Clip with BBox)"},
                {"type": "video", "video": self.ref_annot_clip, 'sample_fps': 1},
                {"type": "text", "text": "### CANDIDATE IMAGE (To Evaluate)"},
                {"type": "image", "image": cand_img_path},
                {"type": "text", "text": user_prompt},
            ]}
        ]
        output = self.context.client.response(messages)
        return "yes" in output.lower()

    def _refine_with_sam_vlm(self, img_bgr, warped_mask, frame_file: str,
                              query, parsed_query, threshold):
        # 第一层：全局VLM存在校验（V1完整逻辑）
        exist_ok = self._vlm_global_check(frame_file, query, parsed_query)
        if not exist_ok:
            print(f"Global VLM reject {os.path.basename(frame_file)}")
            return False, None, None

        # 第二层：光流重心作为SAM点prompt分割
        y_idx, x_idx = np.where(warped_mask)
        if len(y_idx) == 0:
            return False, None, None
        center = [[int(np.mean(x_idx)), int(np.mean(y_idx))]]
        target_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        _, results = segment_mask_with_point(
            self.context.sam_processor,
            self.context.sam_model,
            target_image,
            parsed_query['target_class'],
            point_prompts=center,
            threshold=threshold,
        )
        if results is None or 'masks' not in results or len(results['masks']) == 0:
            return False, None, None

        # 匹配重叠度最高掩码
        best_overlap = 0.0
        best_idx = 0
        warped_float = warped_mask.astype(np.float32)
        for j in range(results['masks'].shape[0]):
            cand = (results['masks'][j].cpu().numpy() > 0.5).astype(np.float32)
            if cand.shape != warped_float.shape:
                warped_resized = cv2.resize(
                    warped_float, (cand.shape[1], cand.shape[0]), cv2.INTER_NEAREST
                )
            else:
                warped_resized = warped_float
            overlap = np.sum(cand * warped_resized) / (np.sum(cand) + 1e-8)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = j
        final_mask = (results['masks'][best_idx].cpu().numpy() > 0.5)

        # 第三层：多掩码时实例ID VLM核验（保留V2轻量化策略）
        instance_pass = True
        if results['masks'].shape[0] > 1:
            _, annotated, _ = get_image_with_segment_and_marker(
                self.context.sam_processor, self.context.sam_model,
                target_image, parsed_query['target_class'],
                threshold=threshold,
                results={
                    'masks': results['masks'][best_idx].unsqueeze(0),
                    'boxes': results['boxes'][best_idx].unsqueeze(0),
                    'scores': results['scores'][best_idx].unsqueeze(0),
                },
                draw_mask=True, draw_id=False,
            )
            target_id = vlm_identify_id(annotated, self.context.client, query, parsed_query)
            if target_id == -1:
                print(f"Instance VLM reject {os.path.basename(frame_file)}")
                instance_pass = False
        if not instance_pass:
            return False, None, None

        # 生成当前帧带bbox标注图，加入参考clip（对齐V1动态扩充上下文）
        output_dir = os.path.join(self.context.cache_dir, 'expansion')
        os.makedirs(output_dir, exist_ok=True)
        res_single = {
            'masks': results['masks'][best_idx].unsqueeze(0),
            'boxes': results['boxes'][best_idx].unsqueeze(0),
            'scores': results['scores'][best_idx].unsqueeze(0),
        }
        _, img_with_box, _ = get_image_with_segment_and_marker(
            self.context.sam_processor, self.context.sam_model,
            target_image, parsed_query['target_class'],
            threshold=threshold,
            results=res_single,
            draw_mask=False, draw_id=False
        )
        annot_save_path = os.path.join(output_dir, f"ref_annot_{os.path.basename(frame_file)}")
        img_with_box.save(annot_save_path)

        return True, final_mask, annot_save_path

    def run(self, reference_image_file: str, reference_image_with_target_mask_box_file: str,
            query: str, parsed_query: str, max_filtered_num: int = 16,
            threshold: float = 0.5):

        print(f"\n--- Optical Flow Assisted Temporal Expansion (V2+Full V1 VLM Pipeline) ---")
        print(f"Reference frame: {os.path.basename(reference_image_file)}")

        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        # 初始化参考标注clip，和V1完全同源起点
        self.ref_annot_clip = [reference_image_with_target_mask_box_file]

        ref_img = cv2.imread(reference_image_file)
        ref_mask = self._load_reference_mask(reference_image_file, parsed_query, threshold)
        ref_idx = self._get_frame_index(reference_image_file)
        scene_dir = os.path.dirname(reference_image_file)

        expanded_frames = [reference_image_file]
        expanded_masks = [ref_mask]

        # ========== 前向追踪（索引更小，时间更早） ==========
        print(f"\n[Forward Tracking (earlier frames)]")
        current_img = ref_img
        current_mask = ref_mask
        forward_count = 0
        half_max = max_filtered_num // 2

        for i in range(1, half_max + 1):
            prev_idx = ref_idx - i
            prev_file = os.path.join(scene_dir, f"{prev_idx:05d}.jpg")
            if not os.path.exists(prev_file):
                print(f"  Frame {prev_idx:05d} missing, stop forward")
                break
            prev_img = cv2.imread(prev_file)

            flow = self.raft.estimate_flow(current_img, prev_img)
            warped_mask = self.warp_mask(current_mask, flow)

            valid, final_mask, annot_path = self._refine_with_sam_vlm(
                prev_img, warped_mask, prev_file, query, parsed_query, threshold
            )
            if valid:
                expanded_frames.insert(0, prev_file)
                expanded_masks.insert(0, final_mask)
                self.ref_annot_clip.insert(0, annot_path)
                current_img = prev_img
                current_mask = final_mask
                forward_count += 1
                # print(f"  Added frame {prev_idx:05d}, total={len(expanded_frames)}")
            else:
                print(f"  Frame {prev_idx:05d} validation failed, stop forward")
                break

        # ========== 后向追踪（索引更大，时间更晚） ==========
        print(f"\n[Backward Tracking (later frames)]")
        current_img = ref_img
        current_mask = ref_mask
        backward_count = 0

        for i in range(1, half_max + 1):
            next_idx = ref_idx + i
            next_file = os.path.join(scene_dir, f"{next_idx:05d}.jpg")
            if not os.path.exists(next_file):
                print(f"  Frame {next_idx:05d} missing, stop backward")
                break
            next_img = cv2.imread(next_file)

            flow = self.raft.estimate_flow(current_img, next_img)
            warped_mask = self.warp_mask(current_mask, flow)

            valid, final_mask, annot_path = self._refine_with_sam_vlm(
                next_img, warped_mask, next_file, query, parsed_query, threshold
            )
            if valid:
                expanded_frames.append(next_file)
                expanded_masks.append(final_mask)
                self.ref_annot_clip.append(annot_path)
                current_img = next_img
                current_mask = final_mask
                backward_count += 1
                # print(f"  Added frame {next_idx:05d}, total={len(expanded_frames)}")
            else:
                print(f"  Frame {next_idx:05d} validation failed, stop backward")
                break

        # ========== 保存输出，完全对齐V1双JSON文件 ==========
        file_name_path = os.path.join(self.context.cache_dir, "expanded_image_files.json")
        save_json(expanded_frames, file_name_path)

        annot_json_path = os.path.join(self.context.cache_dir, "expanded_image_files_with_box.json")
        save_json(self.ref_annot_clip, annot_json_path)

        print(f"\nExpansion finished | total={len(expanded_frames)} (forward:{forward_count}, backward:{backward_count})")
        print(f"Raw list saved to {file_name_path}, annotated clip saved to {annot_json_path}")
        return file_name_path, len(expanded_frames)


class ExpandFromSecondaryViewArgsV2(BaseModel):
    expanded_image_file_path: str = Field(..., description="Path to the current JSON list of expanded images.")
    ranked_image_file_path: str = Field(..., description="Path to the ranked image files JSON list.")
    scores_path: str = Field(..., description="The path to the JSON file containing image scores (output of 'vlm_score').")
    image_files_path: str = Field(..., description="The path to the JSON file containing sorted image paths (output of 'vlm_score').")
    query: str = Field(..., description="The raw natural language query.")
    parsed_query: str = Field(..., description="The structured JSON string.")
    max_filtered_num: int = Field(16, description="Maximum number of frames to expand/track.")
    threshold: float = Field(0.5, description="Segmentation confidence threshold.")


class ExpandFromSecondaryViewToolV2(BaseTool):
    name = "expand_from_secondary_view"
    description = "Iteratively improve view coverage. Searches for a new high-scoring reference frame that is temporally/spatially farthest from the currently collected frames."
    return_description = "A tuple containing: 'path_to_updated_image_files_json, total_count_of_images'."
    args_schema = ExpandFromSecondaryViewArgsV2

    def run(self, expanded_image_file_path: str, scores_path: str,
            ranked_image_file_path: str, image_files_path: str,
            query: str, parsed_query: str, max_filtered_num: int = 32,
            threshold: float = 0.5):

        # Backup old files
        save_json(load_json(expanded_image_file_path),
                  os.path.join(self.context.cache_dir, 'old_expanded_image_files.json'))
        save_json(load_json(scores_path),
                  os.path.join(self.context.cache_dir, 'old_vlm_scores.json'))

        if isinstance(image_files_path, str):
            all_image_files = load_json(image_files_path)
        else:
            all_image_files = image_files_path

        def get_another_reference():
            expanded_image_files = load_json(expanded_image_file_path)
            scores = load_json(scores_path)
            ranked_image_files = load_json(ranked_image_file_path)

            max_score_frames = []
            for i in range(len(scores)):
                if scores[i] == max(scores) or scores[i] >= 4:
                    max_score_frames.append(ranked_image_files[i])
            candidate_frames = [img for img in max_score_frames
                                if img not in expanded_image_files]
            if len(candidate_frames) == 0:
                print("No other candidate frames")
                return None

            def get_frame_index(filename):
                basename = os.path.basename(filename)
                match = re.findall(r'(\d+)', basename)
                return int(match[-1]) if match else 0

            expanded_indices = [get_frame_index(f) for f in expanded_image_files]
            best_candidate = None
            max_min_distance = -1

            for candidate in candidate_frames:
                cand_idx = get_frame_index(candidate)
                min_dist_to_set = min([abs(cand_idx - exist_idx)
                                       for exist_idx in expanded_indices])
                if min_dist_to_set > max_min_distance:
                    max_min_distance = min_dist_to_set
                    best_candidate = candidate

            if best_candidate:
                print(f"Found new reference frame: {best_candidate} (Distance: {max_min_distance})")
                return best_candidate
            else:
                return None

        expanded_image_files = load_json(expanded_image_file_path)
        another_reference_file = get_another_reference()

        if another_reference_file is None:
            return expanded_image_file_path, len(expanded_image_files)

        argmax_image_and_seg_id_tool = ArgmaxImageAndSegIdTool(self.context)

        try:
            ref_index = all_image_files.index(another_reference_file)
        except ValueError:
            ref_index = 0

        scores_for_argmax = [-float('inf')] * len(all_image_files)
        scores_for_argmax[ref_index] = 1.0

        reference_image_file, target_id, reference_image_mask_results_path = \
            argmax_image_and_seg_id_tool.run(
                scores_for_argmax, image_files_path, query, parsed_query, threshold)

        if reference_image_file is None:
            print("Argmax tool failed, using original expanded images")
            return expanded_image_file_path, len(expanded_image_files)

        segment_target_in_reference_tool = SegmentTargetInReferenceTool(self.context)
        reference_image_with_target_mask_box_file = segment_target_in_reference_tool.run(
            reference_image_file, target_id, reference_image_mask_results_path,
            parsed_query, threshold)

        vlm_frame_expansion_tool = VlmFrameExpansionToolV2(self.context)
        another_expanded_image_file_path, num = vlm_frame_expansion_tool.run(
            reference_image_file, reference_image_with_target_mask_box_file,
            query, parsed_query, max_filtered_num, threshold)

        # Merge and deduplicate
        new_expanded_image_files = expanded_image_files + \
                                   load_json(another_expanded_image_file_path)
        final_files = list(dict.fromkeys(new_expanded_image_files))

        file_name_path = os.path.join(self.context.cache_dir, "expanded_image_files.json")
        save_json(final_files, file_name_path)

        return file_name_path, len(final_files)

# ==============================================================================
# V2: TSDF-based Centroid Complete
# ==============================================================================

class TSDFVolumeManager:
    def __init__(self, voxel_size=0.02, sdf_trunc=0.08):
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.volume = None
        self.pcd = None
        self.kdtree = None
        self.has_valid_data = False

    def build_volume(self, rgb_files, depth_files, intrinsics, extrinsics,
                     color_w, color_h, depth_w, depth_h):
        print(f"Building TSDF volume with voxel_size={self.voxel_size}...")
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            color_w, color_h, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        )

        count = 0
        for rgb_path, depth_path, extr in zip(rgb_files, depth_files, extrinsics):
            try:
                rgb = Image.open(rgb_path).convert("RGB")
                depth = np.array(Image.open(depth_path), dtype=np.uint16)

                rgb_o3d = o3d.geometry.Image(np.array(rgb))
                depth_o3d = o3d.geometry.Image(depth)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d,
                    depth_scale=1000.0,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False
                )

                self.volume.integrate(rgbd, intrinsic, np.linalg.inv(extr))
                count += 1
            except Exception as e:
                continue

        print(f"TSDF volume built from {count}/{len(rgb_files)} frames")

        if count == 0:
            self.has_valid_data = False
            return

        self.pcd = self.volume.extract_point_cloud()
        if len(self.pcd.points) == 0:
            self.has_valid_data = False
            return

        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.has_valid_data = True
        print(f"   Point cloud has {len(self.pcd.points)} points")

    def is_visible(self, p_3d, camera_extrinsic, check_radius=0.05):
        if not self.has_valid_data:
            return True

        try:
            [k, idx, _] = self.kdtree.search_radius_vector_3d(p_3d, check_radius)
            if k == 0:
                return False

            camera_center = camera_extrinsic[:3, 3]
            direction = p_3d - camera_center
            distance = np.linalg.norm(direction)
            if distance < 0.1:
                return True

            direction = direction / distance
            step = self.voxel_size * 2.0

            for t in np.arange(0.1, distance - 0.1, step):
                p_check = camera_center + t * direction
                [k, _, _] = self.kdtree.search_radius_vector_3d(
                    p_check, self.voxel_size * 1.5)
                if k > 0:
                    return False

            return True
        except Exception:
            return True


class CentroidCompleteArgsV2(BaseModel):
    mask_filtered_images_file: str = Field(..., description="Path to the JSON list of mask filtered images.")
    pcd_path: str = Field(..., description="File path to the point cloud.")
    parsed_query: str = Field(..., description="The structured JSON string.")
    scene_id: str = Field(..., description="Unique scene ID.")
    threshold: float = Field(0.5, description="Confidence threshold for mask generation.")


class CentroidCompleteToolV2(BaseTool):
    name = "centroid_complete"
    description = "Uses 3D geometric projection, TSDF voxelization and depth-based occlusion checks to track the target across video frames."
    return_description = "A tuple containing: 'path_to_centroid_final_images_json, path_to_centroid_final_masks_pth'."
    args_schema = CentroidCompleteArgsV2

    def run(self, mask_filtered_images_file: str, pcd_path: str, parsed_query: str,
            scene_id: str, threshold: float):

        info_path = os.path.join(self.context.scannet_info_path, scene_id,
                                 scene_id + '.txt')
        scene_dir = os.path.join(self.context.scannet_video_path, scene_id)
        info = load_info_file(info_path)
        if isinstance(parsed_query, str):
            parsed_query = json.loads(parsed_query)

        fx_d, fy_d = float(info['fx_depth']), float(info['fy_depth'])
        cx_d, cy_d = float(info['mx_depth']), float(info['my_depth'])
        fx_c, fy_c = float(info['fx_color']), float(info['fy_color'])
        cx_c, cy_c = float(info['mx_color']), float(info['my_color'])
        color_w, color_h = int(info['colorWidth']), int(info['colorHeight'])
        depth_w, depth_h = int(info['depthWidth']), int(info['depthHeight'])
        axisAlignment = info['axisAlignment'].reshape(4, 4)
        inv_axisAlignment = np.linalg.inv(axisAlignment)
        if 'colorToDepthExtrinsics' in info.keys():
            T_c2d = info['colorToDepthExtrinsics'].reshape(4, 4)
            T_d2c = np.linalg.inv(T_c2d)
        else:
            T_d2c = np.eye(4)

        print(f"\n--- 1. Calculate 3D Bounding Box & Multi-Key-Points ---")
        pcd = o3d.io.read_point_cloud(pcd_path)
        bbox = pcd.get_axis_aligned_bounding_box()
        box_points = bbox.get_box_points()
        centroid = pcd.get_center()
        points_3d = np.vstack([box_points, centroid])
        print(f"Using {len(points_3d)} 3D points (8 corners + 1 centroid)")
        print(f"Centroid: {centroid}")

        print(f"\n--- 2. Build TSDF Volume for Visibility Check ---")
        rgb_files = load_json(mask_filtered_images_file)
        depth_files = []
        extrinsics = []
        valid_rgb_files = []

        for rgb_path in rgb_files:
            idx = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(scene_dir, f"{idx}.png")
            extr_path = os.path.join(scene_dir, f"{idx}.txt")

            if os.path.exists(depth_path) and os.path.exists(extr_path):
                valid_rgb_files.append(rgb_path)
                depth_files.append(depth_path)
                extrinsics.append(load_extrinsics(extr_path))

        tsdf_manager = TSDFVolumeManager()
        intrinsics = [fx_c, fy_c, cx_c, cy_c]
        tsdf_manager.build_volume(valid_rgb_files, depth_files, intrinsics,
                                  extrinsics, color_w, color_h, depth_w, depth_h)

        print(f"\n--- 3. Retrieving Valid Frames (TSDF Visibility Check) ---")
        retrieved_frames = []

        for rgb_path in rgb_files:
            idx = os.path.splitext(os.path.basename(rgb_path))[0]
            extr_path = os.path.join(scene_dir, f"{idx}.txt")
            depth_path = os.path.join(scene_dir, f"{idx}.png")

            if not os.path.exists(extr_path) or not os.path.exists(depth_path):
                continue

            T_d2w_target = load_extrinsics(extr_path)
            T_w2d_target = np.linalg.inv(T_d2w_target)

            projected_points = []
            visible_count = 0

            for p_3d in points_3d:
                C_w_aligned_h = np.append(p_3d, 1.0)
                C_w_unaligned_h = inv_axisAlignment @ C_w_aligned_h
                C_d_target_h = T_w2d_target @ C_w_unaligned_h
                x_d, y_d, z_d = C_d_target_h[:3]

                C_c_target_h = T_d2c @ C_d_target_h
                x_c, y_c, z_predict = C_c_target_h[:3]

                if z_predict <= 0.1 or z_d <= 0.1:
                    continue

                u_proj = int(round((x_c * fx_c / z_predict) + cx_c))
                v_proj = int(round((y_c * fy_c / z_predict) + cy_c))

                if 0 <= u_proj < color_w and 0 <= v_proj < color_h:
                    p_world = inv_axisAlignment[:3, :3] @ p_3d + inv_axisAlignment[:3, 3]
                    if tsdf_manager.is_visible(p_world, T_d2w_target):
                        projected_points.append([u_proj, v_proj])
                        visible_count += 1

            if len(projected_points) >= 3 and visible_count >= 3:
                center_point = np.mean(projected_points, axis=0).astype(int)
                retrieved_frames.append({
                    "frame_id": idx,
                    "rgb_path": rgb_path,
                    "point_prompt": [center_point.tolist()],
                    "all_projected_points": projected_points,
                    "z_predict": z_predict
                })

        print(f"\nTSDF-based retrieval complete!")
        print(f"   Found {len(retrieved_frames)} valid frames out of {len(rgb_files)}")

        print(f"\n--- 4. Visualizing Centroid on Retrieved Frames ---")
        output_dir = os.path.join(self.context.cache_dir, 'centroid_vis')
        os.makedirs(output_dir, exist_ok=True)
        for i, frame_data in enumerate(retrieved_frames[:20]):
            rgb_path = frame_data['rgb_path']
            frame_id = frame_data['frame_id']
            u_proj, v_proj = frame_data['point_prompt'][0]
            img = cv2.imread(rgb_path)

            if 'all_projected_points' in frame_data:
                for (pu, pv) in frame_data['all_projected_points']:
                    cv2.circle(img, (pu, pv), radius=5, color=(255, 0, 0), thickness=-1)

            cv2.circle(img, (u_proj, v_proj), radius=15, color=(255, 255, 255), thickness=4)
            cv2.circle(img, (u_proj, v_proj), radius=12, color=(0, 0, 255), thickness=-1)
            cv2.drawMarker(img, (u_proj, v_proj), color=(0, 0, 0),
                          markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
            save_path = os.path.join(output_dir, f"centroid_vis_{frame_id}.jpg")
            cv2.imwrite(save_path, img)
        print(f"\nVisualizations saved to: {os.path.abspath(output_dir)}")

        print(f"\n--- 5. Segmenting and Filtering Masks using Centroid ---")
        centroid_final_images = []
        centroid_final_masks = []
        mask_output_dir = os.path.join(self.context.cache_dir, 'centroid_vis_masks')
        os.makedirs(mask_output_dir, exist_ok=True)

        for i, frame_data in enumerate(retrieved_frames):
            rgb_path = frame_data['rgb_path']
            frame_id = frame_data['frame_id']
            u_proj, v_proj = frame_data['point_prompt'][0]

            target_image = Image.open(rgb_path).convert("RGB")
            image_np = np.array(target_image)

            _, results = segment_mask(self.context.sam_processor, self.context.sam_model,
                                      target_image, parsed_query['target_class'],
                                      threshold=threshold)

            valid_mask_found = False
            final_mask = None
            target_id = 0
            min_distance = float('inf')
            fallback_mask = None
            fallback_target_id = -1

            if results is not None and 'masks' in results and len(results['masks']) > 0:
                for j in range(results['masks'].shape[0]):
                    mask_np_current = results['masks'][j].cpu().numpy()
                    bool_mask = (mask_np_current > 0.5)

                    if v_proj < bool_mask.shape[0] and u_proj < bool_mask.shape[1]:
                        if results['masks'].shape[0] == 1:
                            final_mask = bool_mask
                            valid_mask_found = True
                            target_id = j
                            break
                        else:
                            if bool_mask[v_proj, u_proj]:
                                final_mask = bool_mask
                                valid_mask_found = True
                                target_id = j
                                break
                            else:
                                y_idx, x_idx = np.where(bool_mask)
                                if len(y_idx) > 0:
                                    distances = np.sqrt(
                                        (x_idx - u_proj) ** 2 + (y_idx - v_proj) ** 2)
                                    current_min_dist = np.min(distances)
                                    if current_min_dist < min_distance:
                                        min_distance = current_min_dist
                                        fallback_mask = bool_mask
                                        fallback_target_id = j

                if not valid_mask_found and fallback_target_id != -1:
                    final_mask = fallback_mask
                    valid_mask_found = True
                    target_id = fallback_target_id

            if valid_mask_found:
                overlay_image_np = image_np.copy()
                overlay_color = np.array([0, 255, 0])
                alpha = 0.5
                original_pixels = image_np[final_mask]
                blended_pixels = (1 - alpha) * original_pixels.astype(np.float32) + \
                                 alpha * overlay_color.astype(np.float32)
                overlay_image_np[final_mask] = np.clip(blended_pixels, 0, 255).astype(np.uint8)
                cv2.circle(overlay_image_np, (u_proj, v_proj), radius=6,
                          color=(255, 0, 0), thickness=-1)
                final_image = Image.fromarray(overlay_image_np)
                save_path = os.path.join(mask_output_dir, f"filtered_mask_{frame_id}.jpg")
                final_image.save(save_path)
                # print(f"[{frame_id}] Target mask validated & saved.")
                centroid_final_images.append(rgb_path)
                centroid_final_masks.append(results['masks'][target_id])
            else:
                print(f"[{frame_id}] No mask contains the centroid point ({u_proj}, {v_proj}).")

        centroid_final_images_file = os.path.join(self.context.cache_dir, 'centroid_final_images.json')
        centroid_final_masks_file = os.path.join(self.context.cache_dir, 'centroid_final_masks.pth')
        save_json(centroid_final_images, centroid_final_images_file)
        torch.save(centroid_final_masks, centroid_final_masks_file)

        print(f"\nCentroid complete (TSDF version) finished!")
        return centroid_final_images_file, centroid_final_masks_file
