"""GPT/Gemini vision-language model reward scorer."""

import json
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from fastvideo.reward.base import RewardScorer
from fastvideo.utils.logging_ import main_print


class GPTRewardScorer(RewardScorer):
    """Binary (0/1) reward from a vision-language model (Gemini / GPT).

    Sends a 2x2 grid of 4 sampled video frames to the API and asks for a
    structured pass/fail judgement.  Any single failure criterion -> score 0.
    """

    def __init__(
        self,
        api_base: str = "http://35.220.164.252:3888/v1/",
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        import os
        import cv2 as _cv2  # noqa - lazy import
        self._cv2 = _cv2
        from openai import OpenAI
        if api_key is None:
            api_key = os.environ.get("GPT_API_KEY", "")
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

    # -- video -> base64 grid -------------------------------------------
    def _video_to_grid_base64(self, video_path: str):
        """Returns (base64_str, grid_numpy_bgr)."""
        import base64
        import numpy as np
        cv2 = self._cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick frames at 0, 1/3, 2/3, and the last frame
        pick_indices = sorted(set([
            0,
            int(total * 1 / 3),
            int(total * 2 / 3),
            total - 1,
        ]))
        pick_indices = [max(0, min(i, total - 1)) for i in pick_indices]
        pick_set = set(pick_indices)

        frames_map = {}
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in pick_set:
                # Crop to main view (head camera) only - top 2/3
                h = frame.shape[0]
                main_h = h * 2 // 3
                frame = frame[:main_h, :, :]
                # Crop left/right 1/10 each
                w = frame.shape[1]
                margin_lr = w // 10
                frame = frame[:, margin_lr:w - margin_lr, :]
                frames_map[idx] = frame
            idx += 1
        cap.release()

        grid_frames = [frames_map[i] for i in pick_indices]

        # 2x2 grid
        rows, cols = 2, 2
        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.concatenate(grid_frames[r * cols:(r + 1) * cols], axis=1))
        grid = np.concatenate(row_imgs, axis=0)

        _, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8"), grid

    # -- prompt ---------------------------------------------------------
    @staticmethod
    def _build_prompt(task_description: str) -> str:
        return f"""You are evaluating an AI-generated robot manipulation video.

The image shows a 2x2 grid of 4 frames sampled from the video in chronological order (read row by row, left-to-right then top-to-bottom).
Each frame is from a fixed rear camera showing the full workspace of a dual-arm "aloha" robot.

**Task description:** {task_description}

This task requires **two arms to collaborate**:
- The **right arm** (on the right bottom side of each image) should **open the drawer** by reaching for and pulling the handle.
- The **left arm** (on the left bottom side of each image) should **pick up the object from the table and place it inside the opened drawer**.

Evaluate the video for the following **failure criteria**. If ANY of them is true, the task FAILS (score = 0). Only if NONE of them is true, the task PASSES (score = 1).

### Failure Criteria
1. **Right arm frozen, drawer opens by itself**: The right arm does not move towards or contact the drawer, yet the drawer opens on its own.
2. **Right arm frozen, left arm does everything**: The right arm stays still while the left arm picks up the object AND also attempts to open the drawer by itself.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"pass": true/false, "reason": "one-sentence explanation", "failures": ["list of triggered failure criterion numbers, e.g. 1,3"]}}
"""

    # -- save debug image: grid + GPT response text below ---------------
    @staticmethod
    def _save_debug_image(grid_bgr, response_text: str, save_path: str):
        """Save the frame grid with GPT response rendered below it."""
        import numpy as np
        cv2_mod = __import__("cv2")

        grid_h, grid_w = grid_bgr.shape[:2]

        # -- render text into a strip below the grid ------
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        color = (255, 255, 255)   # white text
        line_height = 36
        margin = 16

        # Word-wrap the response to fit the grid width using actual text size
        usable_w = grid_w - margin * 2
        words = response_text.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            candidate = f"{cur} {w}" if cur else w
            (tw, _), _ = cv2_mod.getTextSize(candidate, font, font_scale, thickness)
            if tw > usable_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = candidate
        if cur:
            lines.append(cur)

        text_h = margin * 2 + line_height * len(lines)
        text_strip = np.zeros((text_h, grid_w, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            y = margin + line_height * (i + 1)
            cv2_mod.putText(text_strip, line, (margin, y),
                            font, font_scale, color, thickness, cv2_mod.LINE_AA)

        combined = np.concatenate([grid_bgr, text_strip], axis=0)
        cv2_mod.imwrite(save_path, combined)

    def score(self, prompt: str, first_frame: Image.Image,
              video_path: Optional[str] = None,
              debug_save_path: Optional[str] = None) -> Dict[str, float]:
        if video_path is None:
            raise ValueError("video_path is required for GPTRewardScorer")

        grid_b64, grid_bgr = self._video_to_grid_base64(str(Path(video_path).resolve()))
        gpt_prompt = self._build_prompt(prompt)

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": gpt_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{grid_b64}"}},
                        ],
                    }],
                    temperature=self._temperature,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                result = json.loads(raw)
                passed = bool(result.get("pass", False))
                reason = result.get("reason", "")
                failures = result.get("failures", [])
                reward = 1.0 if passed else 0.0

                label = "PASS" if passed else "FAIL"
                response_text = f"[{label}] {reason}  failures={failures}"

                if debug_save_path:
                    try:
                        self._save_debug_image(grid_bgr, response_text, debug_save_path)
                    except Exception as e:
                        main_print(f"  [GPT reward] debug image save failed: {e}")

                return {
                    "reward": reward, "pass": passed,
                    "reason": reason, "failures": failures,
                    "_grid_bgr": grid_bgr, "_response_text": response_text,
                }
            except Exception as exc:
                if attempt < self._max_retries:
                    main_print(f"  [GPT reward] attempt {attempt+1} failed: {exc}, retrying...")
                    continue
                main_print(f"  [GPT reward] all attempts failed: {exc}")
                err_text = f"[API ERROR] {exc}"
                if debug_save_path:
                    try:
                        self._save_debug_image(grid_bgr, err_text, debug_save_path)
                    except Exception:
                        pass
                return {
                    "reward": 0.0, "pass": False,
                    "reason": f"API error: {exc}", "failures": [],
                    "_grid_bgr": grid_bgr, "_response_text": err_text,
                }
