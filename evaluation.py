"""
Evaluation logic for RoboSpatial-Home benchmark.
Contains functions for evaluating model responses against ground truth data.
"""

import os
import re
import ast
from functools import lru_cache
from tqdm import tqdm
from PIL import Image
import numpy as np

_HF_DATASET_NAME = "chanhee-luke/RoboSpatial-Home"
DEFAULT_NUM_POINTS_TO_MATCH = 2
_POINT_RE = re.compile(
    r'[\(\[]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[\)\]]'
)

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _infer_split_and_index_from_mask_path(mask_path):
    """
    Best-effort parser for mask file names produced by this repo's downloader:
      images/mask_context_42.png  -> ("context", 42)
    Returns (split, idx) or (None, None).
    """
    if not mask_path:
        return None, None
    base = os.path.basename(mask_path)
    m = re.match(r"mask_(?P<split>[a-zA-Z]+)_(?P<idx>\d+)\.(png|jpg|jpeg|webp)$", base)
    if not m:
        return None, None
    return m.group("split").lower(), int(m.group("idx"))


def _point_in_mask_pil(x, y, pil_mask):
    """
    Returns True if normalized point (x,y) falls on a nonzero mask pixel.
    """
    im = pil_mask.convert("L")
    w, h = im.size
    px, py = _normalized_xy_to_mask_indices(x, y, w, h)
    arr = np.array(im, dtype=np.uint8)
    return int(arr[py, px]) > 0


_HF_SPLIT_DATASET_CACHE = {}

def _get_hf_split_dataset(split, hf_dataset_name=_HF_DATASET_NAME):
    """
    Cache HF dataset split objects to avoid reloading per entry.
    """
    key = (hf_dataset_name, split)
    if key not in _HF_SPLIT_DATASET_CACHE:
        from datasets import load_dataset
        _HF_SPLIT_DATASET_CACHE[key] = load_dataset(hf_dataset_name, split=split)
    return _HF_SPLIT_DATASET_CACHE[key]


@lru_cache(maxsize=256)
def _load_mask_pil_from_hf(split, idx, hf_dataset_name=_HF_DATASET_NAME):
    """
    Load mask as a PIL image directly from HF (no local saving).
    """
    ds = _get_hf_split_dataset(split, hf_dataset_name=hf_dataset_name)
    sample = ds[int(idx)]
    mask_field = sample.get("mask", None)
    if mask_field is None:
        raise FileNotFoundError(f"HF sample has no mask (split={split} idx={idx})")

    # Convert to PIL
    if isinstance(mask_field, Image.Image):
        pil_mask = mask_field
    elif isinstance(mask_field, dict):
        if "pil" in mask_field and isinstance(mask_field["pil"], Image.Image):
            pil_mask = mask_field["pil"]
        elif "path" in mask_field and mask_field["path"]:
            pil_mask = Image.open(mask_field["path"])
        elif "bytes" in mask_field and mask_field["bytes"] is not None:
            import io
            pil_mask = Image.open(io.BytesIO(mask_field["bytes"]))
        else:
            raise ValueError(f"Unsupported HF mask dict keys: {list(mask_field.keys())}")
    else:
        raise ValueError(f"Unsupported HF mask type: {type(mask_field)}")

    # Detach from any underlying file handle and normalize mode once.
    return pil_mask.convert("L")


def _load_mask_pil(mask_abs_path, mask_path=None, category=None, hf_dataset_name=_HF_DATASET_NAME):
    """
    Load a mask image either from local disk (if present) or directly from HF.
    """
    if mask_abs_path and os.path.exists(mask_abs_path):
        with Image.open(mask_abs_path) as im:
            return im.convert("L")

    split, idx = _infer_split_and_index_from_mask_path(mask_path)
    # Prefer explicit category if provided (mirrors previous behavior)
    if category:
        split = str(category).lower()
    if split is None or idx is None:
        raise FileNotFoundError(
            f"Mask missing and cannot infer HF row from path: {mask_abs_path or mask_path}"
        )

    return _load_mask_pil_from_hf(split, idx, hf_dataset_name=hf_dataset_name)

def _normalized_xy_to_mask_indices(x, y, width, height):
    """
    Convert normalized (x,y) in [0,1]x[0,1] to integer pixel indices.
    Uses nearest-neighbor rounding and clamps to image bounds.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Invalid mask size")
    px = int(round(x * (width - 1)))
    py = int(round(y * (height - 1)))
    px = _clamp(px, 0, width - 1)
    py = _clamp(py, 0, height - 1)
    return px, py


def _normalize_num_points_to_match(num_points_to_match):
    try:
        value = int(num_points_to_match)
    except (TypeError, ValueError):
        value = DEFAULT_NUM_POINTS_TO_MATCH
    return max(1, value)


def _coerce_point(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return (float(x), float(y))


def _collect_points(value, points, limit):
    if len(points) >= limit:
        return

    point = _coerce_point(value)
    if point is not None:
        points.append(point)
        return

    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_points(item, points, limit)
            if len(points) >= limit:
                return


def _extract_points(generated_answer, num_points_to_match=DEFAULT_NUM_POINTS_TO_MATCH):
    """
    Extract up to num_points_to_match (x,y) points from the model output string.
    Returns (parsed_points_list_or_None, is_parsable_bool).
    """
    limit = _normalize_num_points_to_match(num_points_to_match)
    answer_text = str(generated_answer)
    points = []

    # Common model outputs mix explanatory text with tuple/list coordinates.
    for match in _POINT_RE.finditer(answer_text):
        try:
            points.append((float(match.group(1)), float(match.group(2))))
        except (ValueError, TypeError):
            continue
        if len(points) >= limit:
            return points, True

    if points:
        return points, True

    # Fall back to Python-literal parsing for structured outputs that do not
    # contain bracketed coordinate pairs in the usual textual form.
    try:
        gen_val = ast.literal_eval(answer_text.strip())
    except (SyntaxError, ValueError):
        return None, False

    _collect_points(gen_val, points, limit)
    if not points:
        return None, False
    return points, True


def _extract_first_point(generated_answer):
    """
    Extract the first (x,y) point from the model output string.
    Returns (parsed_answer_tuple_or_None, is_parsable_bool).
    """
    points, is_parsable = _extract_points(generated_answer, num_points_to_match=1)
    if not is_parsable or not points:
        return None, False
    return points[0], True

def evaluate_answer(
    ground_truth,
    generated_answer,
    mask_path=None,
    data_dir=None,
    category=None,
    num_points_to_match=DEFAULT_NUM_POINTS_TO_MATCH,
):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
    """
    gen_answer = generated_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()
    
    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_gt_yes = (gt_lower == "yes")
        # Binary answers are always considered parsable if they contain text
        is_parsable = len(gen_answer) > 0
        if is_gt_yes:
            correct = gen_answer.startswith("yes")
        else:
            correct = gen_answer.startswith("no")
        return correct, is_binary, gen_answer, is_parsable
    else:
        # Numeric evaluation: use mask-based point-in-region only
        is_binary = False
        parsed_answer = None
        is_parsable = False

        parsed_points, is_parsable = _extract_points(
            generated_answer,
            num_points_to_match=num_points_to_match,
        )
        if not is_parsable or parsed_points is None:
            return False, is_binary, None, False
        parsed_answer = parsed_points[0] if len(parsed_points) == 1 else parsed_points

        # Require mask for non-binary questions from now on
        if not mask_path:
            return False, is_binary, parsed_answer, is_parsable

        try:
            mask_abs_path = os.path.join(data_dir, mask_path) if data_dir else mask_path
            pil_mask = _load_mask_pil(
                mask_abs_path,
                mask_path=mask_path,
                category=category,
            )
            correct = any(_point_in_mask_pil(x, y, pil_mask) for x, y in parsed_points)
            return correct, is_binary, parsed_answer, is_parsable
        except Exception as e:
            print(f"Error evaluating mask-based answer: {e}")
            return False, is_binary, parsed_answer, is_parsable

def eval_robospatial_home(
    json_data,
    model_name,
    model_kwargs,
    data_dir,
    run_model_fn,
    pbar=None,
    split_name=None,
    num_points_to_match=DEFAULT_NUM_POINTS_TO_MATCH,
):
    """
    Evaluate RoboSpatial-Home data by running the model on each example.
    
    Args:
        json_data: List of data entries to evaluate
        model_name: Name of the model being evaluated
        model_kwargs: Model-specific arguments (tokenizer, model object, etc.)
        data_dir: Root directory containing dataset files and images
        run_model_fn: Function to run the model on a single example
        
    Returns:
        Dictionary containing evaluation statistics and results
    """
    results = []
    num_correct = 0
    num_total = len(json_data)
    illformed_questions = 0
    illformed_responses = 0

    # Dictionary to keep per-category statistics
    category_stats = {}

    iterator = json_data if pbar is not None else tqdm(json_data, desc="Evaluating RoboSpatial-Home")
    for entry in iterator:
        # Extract question and ground-truth answer directly from the entry
        question = entry.get("question", "")
        ground_truth = entry.get("answer", "")
        mask_rel_path = entry.get("mask", None)
        
        if not question or not ground_truth:
            illformed_questions += 1
            continue

        category = entry.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"num_correct": 0, "num_total": 0}
        category_stats[category]["num_total"] += 1

        # Build absolute image path using the img field
        image_rel_path = entry.get("img", "")
        image_path = os.path.join(data_dir, image_rel_path)

        depth_rel_path = entry.get("depth_image", None)
        depth_path = os.path.join(data_dir, depth_rel_path) if depth_rel_path else None

        # Run the model
        generated_answer = run_model_fn(question, image_path, depth_path, model_name, model_kwargs)

        # Evaluate the answer
        correct, is_binary, parsed_answer, is_parsable = evaluate_answer(
            ground_truth,
            generated_answer,
            mask_path=mask_rel_path,
            data_dir=data_dir,
            category=category,
            num_points_to_match=num_points_to_match,
        )
        
        # Count illformed responses - now tracks any answer that couldn't be parsed correctly
        if not is_parsable:
            illformed_responses += 1

        if correct:
            num_correct += 1
            category_stats[category]["num_correct"] += 1

        results.append({
            "question": question,
            "expected_answer": ground_truth,
            "generated_answer": generated_answer,
            "parsed_answer": str(parsed_answer) if parsed_answer is not None else None,
            "correct": correct,
            "is_parsable": is_parsable,
            "category": category,
            "image": image_path,
            "depth_image": depth_path,
        })
        if pbar is not None:
            if split_name is not None:
                pbar.set_postfix(split=split_name, category=category, refresh=False)
            pbar.update(1)

    accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "illformed_questions": illformed_questions,
        "illformed_responses": illformed_responses,
        "num_points_to_match": _normalize_num_points_to_match(num_points_to_match),
        "category_stats": category_stats,
        "results": results
    }

def eval_pregenerated_results(
    gt_data,
    results_data,
    data_dir,
    pbar=None,
    split_name=None,
    num_points_to_match=DEFAULT_NUM_POINTS_TO_MATCH,
):
    """
    Evaluate pre-generated results against ground truth.
    
    Args:
        gt_data: List of ground truth data (from the benchmark)
        results_data: List of pre-generated model responses
        data_dir: Root directory containing dataset files and images
        
    Returns:
        Dictionary of evaluation statistics
    """
    results = []
    num_correct = 0
    num_total = 0  # Will count only entries that can be evaluated
    illformed_questions = 0
    illformed_responses = 0
    unmatched_entries = 0  # New counter for entries without matching results
    
    # Dictionary to keep per-category statistics
    category_stats = {}

    # Pre-process results_data for more efficient matching
    # Build lookup dictionaries for faster matching
    results_by_question_and_image = {}
    
    for result_entry in results_data:
        question = result_entry.get("question", "")
        img_path = result_entry.get("img", "")
        
        if question and img_path:
            key = (question, img_path)
            results_by_question_and_image[key] = result_entry

    # Process ground truth entries
    iterator = gt_data if pbar is not None else tqdm(gt_data, desc="Evaluating Pre-generated Results")
    for gt_entry in iterator:
        # Extract data from ground truth entry
        question = gt_entry.get("question", "")
        ground_truth = gt_entry.get("answer", "")
        image_rel_path = gt_entry.get("img", "")
        mask_rel_path = gt_entry.get("mask", None)
        
        if not question or not ground_truth:
            illformed_questions += 1
            continue

        # Increment category stats
        category = gt_entry.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"num_correct": 0, "num_total": 0}
        
        # Try to find a match in the pre-processed results
        key = (question, image_rel_path)
        matched_result = results_by_question_and_image.get(key)
        
        # If no match found, check if this is from a known source file
        if matched_result is None:
            # Count as unmatched rather than illformed
            unmatched_entries += 1
            continue
        
        # Only now do we count this entry toward the total and category
        num_total += 1
        category_stats[category]["num_total"] += 1
        
        # Extract generated answer
        generated_answer = matched_result.get("answer", "")
        
        if not generated_answer:
            illformed_responses += 1
            continue

        # Build absolute image path
        image_path = os.path.join(data_dir, image_rel_path)

        # Evaluate the answer
        correct, is_binary, parsed_answer, is_parsable = evaluate_answer(
            ground_truth,
            generated_answer,
            mask_path=mask_rel_path,
            data_dir=data_dir,
            category=category,
            num_points_to_match=num_points_to_match,
        )
        
        # Count illformed responses - now tracks any answer that couldn't be parsed correctly
        if not is_parsable:
            illformed_responses += 1

        if correct:
            num_correct += 1
            category_stats[category]["num_correct"] += 1

        results.append({
            "question": question,
            "expected_answer": ground_truth,
            "generated_answer": generated_answer,
            "parsed_answer": str(parsed_answer) if parsed_answer is not None else None,
            "correct": correct,
            "is_parsable": is_parsable,
            "category": category,
            "image": image_path
        })
        if pbar is not None:
            if split_name is not None:
                pbar.set_postfix(split=split_name, category=category, refresh=False)
            pbar.update(1)

    # Calculate accuracy
    accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "illformed_questions": illformed_questions,
        "illformed_responses": illformed_responses,
        "num_points_to_match": _normalize_num_points_to_match(num_points_to_match),
        "unmatched_entries": unmatched_entries,  # New field to track unmatched entries
        "category_stats": category_stats,
        "results": results
    } 
