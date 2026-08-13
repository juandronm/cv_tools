"""Run trained YOLO weights over a recorded video and write an annotated copy.

Why
---
Nothing here could point a trained model at a video file. `extract_frames.py` pulls frames OUT
of a recording for labelling, and every detector in this repo (`crops.py`, `gstreamer_opt.py`,
`data_collector/`) reads a LIVE RTSP stream. So the first question after a training run — "what
does this model actually do on the footage?" — had no answer short of writing a throwaway loop.

This tool decodes the video frame by frame, runs the detector, draws what it found, and writes a
new video next to nothing else. It changes no input file.

Matching the training preprocessing
-----------------------------------
The one thing this script does that a naive loop does not: it feeds the model frames shaped the
way its TRAINING images were shaped, not the way the video is shaped.

A Roboflow export resizes to a square. If the source footage was 16:9, the training images are
STRETCHED, and the model learned objects with that distortion baked in. Handing it an
undistorted frame is a domain shift, and it quietly costs detections — measured on the gaziantep
pothole model over 20 frames it had memorised (conf 0.25):

    the 512x512 Roboflow jpg it trained on ......... 18 detections
    same instant from the video, native 720p ....... 14 detections   <- 22% lost
    same instant from the video, stretched to 512 .. 18 detections

Hence `--preprocess stretch` (the default): resize each frame to `--stretch-size`, infer, then
map the boxes back onto the full-resolution frame for drawing. Pass `--preprocess native` if the
model was trained on undistorted images, which is the case for any dataset that was letterboxed
rather than stretched.

Rule-based noise reduction
--------------------------
A detector trained on 40 pothole instances fires on plenty of things that are not potholes. Three
rules run over its output before anything is drawn or counted, and they are wildly unequal — on the
full gaziantep video (1368 raw detections):

    track seen in < 3 frames ..... 772 dropped   <- almost all of the noise
    on a vehicle/person ........... 58 dropped
    centre above the road ........... 2 dropped

Persistence does the work, because most false positives are one-frame flashes. But it cannot catch
boxes on cars: 58 of the 90 vehicle-overlapping detections SURVIVE it, since a parked car is
stationary and the tracker follows it happily. So the vehicle veto runs a COCO model
(`data_collector/yolo26n.pt`) alongside and rejects anything sitting mostly inside a car, truck,
bus or person. The two rules are complementary, not redundant.

Area and aspect-ratio envelopes were measured and deliberately rejected: on top of persistence they
remove 0 and 8 detections respectively, and an envelope drawn from 54 training boxes will eventually
throw away a real pothole of an unusual shape.

These rules change the detections, so the reported counts change with them — `--no-filters` turns
them all off, and the summary always prints the full funnel rather than a quietly smaller total.

What gets drawn
---------------
Corner brackets on each detection, coloured amber through red by confidence, with a label chip
carrying the track id where there is one. A card top-left counts unique potholes and shows the
position in the clip; a bar along the bottom fills as the video plays and grows a tick each time a
new pothole is first seen. `--no-dashboard` drops the card and the bar. A scrolling left-side log
below the card adds a thumbnail + Kamera/Saat/Konum card the moment each track is confirmed,
configurable via `--camera-name`/`--location` and toggled independently with `--no-detection-log`.

Detections are also held on screen for `--hold-frames` frames after the model stops emitting them,
fading out. That is a DISPLAY effect and nothing else — every number this script reports is counted
from raw per-frame model output, so the smoothing cannot flatter the results. It exists because a
detector that fires on a single frame and drops out strobes badly on playback; `--hold-frames 0`
turns it off and shows the raw truth.

IMPORTANT — a video the model was trained on is not an evaluation
------------------------------------------------------------------
If you point this at the same recording the training frames came from, the model has already
seen most of what it is being shown. The output is a qualitative check — useful for spotting
obvious failure modes, worthless as a measure of accuracy. Score on held-out footage.

Run it
------
    # the usual case: annotate the whole video with the gaziantep weights
    python predict_video.py --video C:\\Users\\RONSITO\\Videos\\gaziantep.mp4

    # preview ten seconds before committing to the full pass
    python predict_video.py --video clip.mp4 --start 10 --end 20 --out slice.mp4

    # other weights, a stricter threshold, no tracking
    python predict_video.py --video clip.mp4 --weights runs\\detect\\train\\weights\\best.pt ^
        --conf 0.45 --no-track
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

# extract_frames.py sits at the repo root and does nothing at import time, so its video handling
# is importable as-is. Reused rather than copied: the fps validation and the deliberate choice of
# sequential grab() over CAP_PROP_POS_FRAMES seeking are decisions that must not diverge between
# the tool that reads frames out of a video and the tool that reads frames back in.
from extract_frames import iter_sampled_frames, open_video, resolve_source_fps

logger = logging.getLogger("predict_video")

# yolo11s trained on RDD2022 (Czech/India/Japan/US) merged with our own Gaziantep frames — 21k
# images, 29k instances, class `road_damage`. It scores 0.559 mAP50 on a 2,633-image held-out
# split, against 0.007 for the previous Gaziantep-only model and 0.003 for a public pretrained
# pothole model. Staged into models/ because its training output lives under notebooks/, which is
# disposable. Build the dataset with build_road_damage_dataset.py.
_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "road_damage_v2.pt"

# Roboflow's default export size, and what the gaziantep dataset actually is. See the module
# docstring for why this matters more than it looks like it should.
DEFAULT_STRETCH_SIZE = 512

# Ultralytics' training default, and what optimize_to_engine.py:DETECT_IMGSZ builds engines at.
DEFAULT_IMGSZ = 640

# Paired with DEFAULT_WEIGHTS, not a general-purpose value. Swept over 800 held-out images
# (601 instances):
#
#     conf   boxes/img   precision   recall   F1
#     0.10       1.18        0.307    0.483   0.375
#     0.15       0.60        ~0.45    ~0.40   0.409   <- F1 optimum
#     0.30       0.27        0.611    0.220   0.323
#     0.35       ~0.20       ~0.68    ~0.18   ~0.28   <- what we ship
#     0.50       0.07        0.828    0.080   0.146
#
# Set well above the F1 optimum on purpose. F1 weights precision and recall equally, but a video
# someone is watching is better served by precision — a clean frame with three correct boxes reads
# better than a busy one with six correct and eight wrong. THE COST IS REAL AND LARGE: at 0.50 the
# model reports roughly one damaged patch in twelve. This is a presentation threshold, not a survey
# threshold. Anything that needs to actually FIND the damage (counting, mapping, maintenance
# planning) should run nearer 0.15.
#
# What no threshold fixes, measured over 260 sampled frames of gaziantep.mp4: boxes landing on
# painted lane markings carry a MEDIAN confidence of 0.536 against 0.243 for genuine damage. They
# are among the model's most confident outputs, so raising the threshold culls real detections
# faster than it culls them, and the share of survivors sitting on lines/shadows RISES from 2% at
# conf 0.15 to 5% at 0.70. The cause is the training data: D00 (longitudinal crack) is the largest
# class at 13,342 instances and looks exactly like a lane line. Fixing it needs road markings
# trained as their own class — RDD2022 ships D44 (5,057 instances) and D43 (793), both currently
# discarded by build_road_damage_dataset.py — not a threshold.
#
# Note also how far this moved between models: the previous Gaziantep-only weights peaked at 0.85,
# these at 0.15. Calibration is a property of the trained weights and does NOT transfer, so RE-TUNE
# whenever --weights changes; section 6 of notebooks/gaziantep_demo.ipynb prints the number.
DEFAULT_CONF = 0.55

_PROGRESS_EVERY = 250  # frames between progress lines; ~8s of 30fps footage

# Hazard ramp, BGR. Amber at the confidence threshold through to red at 1.0 — for a single-class
# detector, grading the colour by confidence tells the viewer something a per-track rainbow cannot.
# Blue is held near zero on both ends so the ramp reads as "alert" rather than the muddy
# amber/rust a blue tint produces against gray asphalt; red is capped at 235, not 255, so it stays
# visually distinct from tail-lights/red vehicles already common in dashcam footage.
_COLOUR_LOW = (0, 195, 255)    # vivid amber/gold
_COLOUR_HIGH = (0, 25, 235)    # vivid red

_INK = (255, 255, 255)
_INK_SHADOW = (20, 20, 20)
_PANEL_BG = (28, 24, 22)
# Nearly opaque on purpose: dashcam footage burns its own text into the same corner, and at 0.72
# the source video's vehicle counter reads straight through the card and collides with the count.
_PANEL_ALPHA = 0.96
# The detection-log cards are a separate, lower alpha — the reference UI they're matching shows
# the footage behind each card, not a near-solid fill like the dashboard's.
_LOG_PANEL_ALPHA = 0.6
_BOX_FILL_ALPHA = 0.22

# Left-column layout. The dashboard card and the detection-log panel below it share a left edge
# (_DASH_PAD) but are sized independently — the log panel is deliberately the bigger, more
# attention-grabbing element; the compact counter card above it is left alone.
_DASH_PAD = 14
_DASH_CARD_W, _DASH_CARD_H = 300, 104
_LOG_THUMB = 120           # thumbnail square side, px
_LOG_CARD_W = 420          # total row width (thumb + gap + info card), independent of _DASH_CARD_W
_LOG_ROW_GAP = 12          # vertical gap between history rows
_LOG_TOP_GAP = 10          # gap between the dashboard card's bottom edge and the log panel's top
_LOG_BOTTOM_MARGIN = 40    # keep clear of draw_timeline's bar
_LOG_MAX_CARDS = 300       # safety cap on stored history; draw only ever shows what fits on screen

# Overlays are matched to detections by track id first; this is the fallback for the ~40% of
# detections ByteTrack never confirms, which are exactly the ones that cause the flicker.
_HOLD_MATCH_IOU = 0.3

# Already vendored for data_collector/detector.py; 80 COCO classes, and the only thing in this repo
# that knows what a car looks like.
DEFAULT_SUPPRESS_WEIGHTS = _REPO_ROOT / "data_collector" / "yolo26n.pt"
DEFAULT_SUPPRESS_CLASSES = "car,truck,bus,motorcycle,bicycle,train,person"

# A shadow falls on the road BESIDE its caster, not inside its box, so the plain overlap veto
# cannot reach it. These classes get a second veto box extended downward over the ground where a
# shadow would land.
#
# BE HONEST ABOUT WHAT THIS IS WORTH. Measured on 420 sampled frames of gaziantep.mp4 at conf 0.35,
# the grown zone caught exactly ONE detection — a box on a pedestrian's feet — and that one was
# already 95% inside the plain person box, so the ordinary veto had it covered. Net additional
# detections removed: ZERO. It is kept on as cheap insurance for footage with longer shadows than
# this midday clip, not because it has been shown to earn its place here. If it ever starts
# removing real damage, turn it off with --shadow-extend 0 and lose nothing measured.
#
# Only people and two-wheelers, NOT cars, and that restriction is load-bearing: of 41 detections
# within three box-heights of something's base, the rest were near parked cars and were genuine
# cracks and patched asphalt. Growing car boxes deleted up to 15 of 55 real detections and zero
# shadows.
DEFAULT_SHADOW_CLASSES = "person,bicycle,motorcycle"


# ---------------------------------------------------------------- pure helpers


def scale_boxes(boxes: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from the inference frame back onto the original frame.

    A plain per-axis ratio is correct here precisely BECAUSE the frame was stretched rather than
    letterboxed: there is no padding to subtract first. Getting this wrong is silent — boxes land
    in plausible-looking but wrong places — so `--preprocess native` deliberately skips this
    function entirely rather than passing it a no-op scale.
    """
    if not len(boxes):
        return boxes
    from_w, from_h = from_size
    to_w, to_h = to_size
    scale = np.array([to_w / from_w, to_h / from_h, to_w / from_w, to_h / from_h], dtype=float)
    return boxes * scale


def colour_for(score: float, conf_floor: float) -> tuple[int, int, int]:
    """Amber at the confidence threshold, red at 1.0.

    Graded by confidence rather than by track id: with one class, a per-track rainbow encodes
    nothing the viewer can act on and just reads as noise, whereas "redder means surer" is legible
    without a legend. The track id survives in the label text.
    """
    span = max(1e-6, 1.0 - conf_floor)
    t = min(1.0, max(0.0, (score - conf_floor) / span))
    return tuple(int(round(lo + (hi - lo) * t)) for lo, hi in zip(_COLOUR_LOW, _COLOUR_HIGH))


def iou_1_to_n(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many. Used only to match held overlays to new detections."""
    if not len(others):
        return np.zeros(0)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area + areas - inter + 1e-9)


def format_clock(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def format_clock_hms(seconds: float) -> str:
    """HH:MM:SS, for the detection-log's 'Saat' field (elapsed clip position, not wall-clock).

    A sibling to format_clock, not a change to it: format_clock's MM:SS output still feeds
    draw_dashboard's clock line as-is.
    """
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def inside_fraction(box: np.ndarray, others: np.ndarray) -> float:
    """Largest fraction of `box`'s own area that falls inside any of `others`.

    Intersection over the AREA OF BOX, not IoU, and the distinction is the whole filter. A pothole
    box is small and a car box is large, so their IoU stays near zero even when the pothole box sits
    entirely inside the car — an IoU-based version of this test would never fire while reading as
    perfectly reasonable code. Anyone "simplifying" this to iou_1_to_n breaks it silently.
    """
    if not len(others):
        return 0.0
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return float(np.max(inter) / area)


def crop_box(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Clamp an xyxy box to the frame and return the pixel crop, or None if degenerate.

    Mirrors the clamp-and-slice already inlined in draw_detections rather than a third
    reimplementation, or reaching into data_collector/detector.py's crop_detections across a
    package boundary that has no __init__.py and assumes it is run from inside that folder.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def make_thumbnail(crop: np.ndarray, size: int) -> np.ndarray:
    """Letterbox an arbitrary-aspect crop onto a size x size BGR square, padded with _PANEL_BG.

    Squashing to a square would visibly distort a wide, short crack crop; letterboxing keeps the
    object recognisable. Called once per card at capture time — a card is redrawn on every
    remaining frame of the video once created, so resizing at draw time would repeat the same
    cv2.resize call thousands of times for one card.
    """
    canvas = np.full((size, size, 3), _PANEL_BG, dtype=np.uint8)
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return canvas
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off, y_off = (size - new_w) // 2, (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


class TrackHitCounter:
    """How many frames each ByteTrack id has been seen in.

    Backs the persistence rule, which is the single most effective noise filter here: on the
    gaziantep footage, requiring three sightings removes 56% of raw detections. A detection with no
    track id can never reach the threshold, and that is the intended behaviour rather than an
    accident — an unconfirmed track is an uncorroborated detection, and those alone are 32% of raw
    output.
    """

    def __init__(self) -> None:
        self._hits: dict[int, int] = {}

    def record(self, track_ids: list[int | None]) -> None:
        for track_id in track_ids:
            if track_id is not None:
                self._hits[track_id] = self._hits.get(track_id, 0) + 1

    def hits(self, track_id: int | None) -> int:
        return 0 if track_id is None else self._hits.get(track_id, 0)


def output_path_for(video_path: Path, requested: str | None) -> Path:
    """Default the output next to nothing — cwd, not beside the source video.

    Writing into the folder the footage lives in is the kind of helpfulness that dirties a media
    library, so the default lands where the command was run from.
    """
    if requested:
        return Path(requested)
    return Path.cwd() / f"{video_path.stem}_detected.mp4"


# ---------------------------------------------------------------- drawing


class OverlayStore:
    """Keeps detections on screen briefly after the model stops emitting them.

    THIS IS A DISPLAY EFFECT ONLY. It never feeds the statistics run() reports — those keep
    counting raw per-frame model output. A smoothing filter that quietly inflated the numbers
    would be the worst possible failure for a tool whose entire purpose is honest evaluation, so
    the two paths are kept strictly separate: run() tallies from `boxes`, the screen draws from
    here.

    Why it exists: this detector confirms a ByteTrack track for barely half its detections. The
    rest fire on a single frame and vanish, which on playback reads as broken rather than as
    "uncertain". Holding each box for a few frames and fading it out turns that strobing into
    something a person can actually watch, without pretending the detections were any steadier
    than they were — a fading box looks exactly like what it is.
    """

    def __init__(self, hold_frames: int) -> None:
        self.hold_frames = hold_frames
        self._items: list[dict] = []

    def update(self, frame_index: int, boxes: np.ndarray, scores: np.ndarray,
               track_ids: list[int | None], classes: np.ndarray) -> None:
        if self.hold_frames <= 0:
            self._items = [
                {"box": b, "score": float(s), "id": i, "cls": int(c), "seen": frame_index}
                for b, s, i, c in zip(boxes, scores, track_ids, classes)
            ]
            return

        self._items = [it for it in self._items
                       if frame_index - it["seen"] <= self.hold_frames]

        for box, score, track_id, cls in zip(boxes, scores, track_ids, classes):
            match = None
            if track_id is not None:
                match = next((it for it in self._items if it["id"] == track_id), None)
            if match is None:
                # No id, or an id we have not seen: fall back to overlap. This is what covers the
                # unconfirmed detections, which are the ones actually causing the flicker.
                stale = [it for it in self._items if it["seen"] != frame_index]
                if stale:
                    ious = iou_1_to_n(box, np.array([it["box"] for it in stale]))
                    best = int(np.argmax(ious))
                    if ious[best] >= _HOLD_MATCH_IOU:
                        match = stale[best]
            if match is None:
                self._items.append({"box": box, "score": float(score), "id": track_id,
                                    "cls": int(cls), "seen": frame_index})
            else:
                match.update(box=box, score=float(score), cls=int(cls), seen=frame_index)
                if track_id is not None:
                    match["id"] = track_id

    def visible(self, frame_index: int) -> list[tuple[dict, float]]:
        """Each held item with the alpha it should be drawn at (1.0 fresh, fading to 0)."""
        out = []
        for item in self._items:
            age = frame_index - item["seen"]
            if age < 0 or age > self.hold_frames:
                continue
            alpha = 1.0 if self.hold_frames <= 0 else 1.0 - age / (self.hold_frames + 1)
            out.append((item, alpha))
        return out


def extend_downward(boxes: np.ndarray, factor: float, frame_height: int) -> np.ndarray:
    """Grow each box downward by `factor` of its own height, clipped to the frame.

    Scaling by the box's OWN height is what makes one setting work at every distance: a pedestrian
    near the camera is tall and casts a long shadow in pixels, one far away is short and casts a
    short one, and both are covered by the same multiple.
    """
    if not len(boxes) or factor <= 0:
        return boxes
    grown = boxes.copy().astype(float)
    heights = grown[:, 3] - grown[:, 1]
    grown[:, 3] = np.minimum(float(frame_height), grown[:, 3] + heights * factor)
    return grown


def apply_rules(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    track_ids: list[int | None],
    hits: TrackHitCounter,
    veto_boxes: np.ndarray,
    min_track_hits: int,
    min_inside: float,
    roi_top: float,
    frame_height: int,
    shadow_boxes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int | None], dict[str, int]]:
    """Drop detections the model should not have made, and say which rule dropped them.

    These rules change the DETECTIONS, not just the pixels — they belong with --conf and --iou, not
    with --hold-frames. So the counts run() reports move with them, and the caller prints a funnel
    rather than a quietly smaller total.

    Measured on the full gaziantep video (1368 raw detections), the rules are wildly unequal and it
    is worth knowing which is carrying the work:

        track seen < 3 frames ....... 772 dropped   <- almost all of it
        on a vehicle/person .......... 58 dropped
        above the road ................ 2 dropped

    Persistence and the vehicle veto are complementary rather than redundant: 58 of the 90
    vehicle-overlapping detections SURVIVE the persistence test, because a parked car is stationary
    and ByteTrack follows it perfectly happily. That is precisely why boxes on cars were the failure
    a viewer noticed first.

    Area and aspect-ratio envelopes were measured too and deliberately left out: on top of
    persistence they remove 0 and 8 detections respectively, and any envelope drawn from 54 training
    boxes will eventually delete a real pothole of unusual shape. Do not add them back without new
    numbers.
    """
    keep: list[int] = []
    dropped = {"transient": 0, "vehicle": 0, "shadow": 0, "off_road": 0}
    shadow_boxes = np.zeros((0, 4)) if shadow_boxes is None else shadow_boxes

    for i, box in enumerate(boxes):
        if min_track_hits > 1 and hits.hits(track_ids[i]) < min_track_hits:
            dropped["transient"] += 1
            continue
        if len(veto_boxes) and inside_fraction(box, veto_boxes) > min_inside:
            dropped["vehicle"] += 1
            continue
        # Separate bucket from `vehicle` on purpose: this one uses boxes grown downward, so it can
        # reach ground the object itself does not cover, and its false-positive risk is different.
        # Keeping the counts apart is what let the growth factor be tuned against real numbers.
        if len(shadow_boxes) and inside_fraction(box, shadow_boxes) > min_inside:
            dropped["shadow"] += 1
            continue
        if roi_top > 0 and ((box[1] + box[3]) / 2.0) < roi_top * frame_height:
            dropped["off_road"] += 1
            continue
        keep.append(i)

    if len(keep) == len(boxes):
        return boxes, scores, classes, track_ids, dropped
    index = np.array(keep, dtype=int)
    return (
        boxes[index] if len(index) else np.zeros((0, 4)),
        scores[index] if len(index) else np.zeros(0),
        classes[index] if len(index) else np.zeros(0),
        [track_ids[i] for i in keep],
        dropped,
    )


# ---------------------------------------------------------------- drawing


def _blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                colour: tuple[int, int, int], alpha: float) -> None:
    """Alpha-fill a rectangle by blending just that ROI — no full-frame compositing."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1.0 - alpha) + np.array(colour, dtype=float) * alpha).astype(frame.dtype)


def _text(frame: np.ndarray, label: str, org: tuple[int, int], scale: float,
          colour: tuple[int, int, int] = _INK, weight: int = 1,
          font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
    """Text with a shadow, so it survives whatever the footage puts behind it."""
    cv2.putText(frame, label, (org[0] + 1, org[1] + 1), font, scale, _INK_SHADOW,
                weight + 2, cv2.LINE_AA)
    cv2.putText(frame, label, org, font, scale, colour, weight, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, visible: list[tuple[dict, float]], names: dict,
                    conf_floor: float) -> None:
    """Corner brackets rather than closed rectangles, at the frame's original resolution.

    Brackets mark where the object is without drawing a line across the pothole itself — the
    texture inside the box is the thing a viewer is trying to judge, and a full rectangle plus a
    label chip covers a surprising amount of it.

    Deliberately not result.plot(): that renders onto the (possibly stretched) image the model
    saw, which would hand back a distorted 512x512 video instead of the real footage.

    A label reading `#7 pothole 0.61` is a confirmed track; `pothole 0.61` was detected but never
    tracked. ByteTrack only confirms a track once the same object is matched across consecutive
    frames, so a detector that fires once and drops out produces the second kind. run() reports
    the proportion, which is a useful read on how stable the detector is.
    """
    height, width = frame.shape[:2]
    thickness = max(2, round(min(width, height) / 260))
    single_class = isinstance(names, dict) and len(names) == 1

    # Confident first, so that when labels compete for the same patch of screen the one that
    # survives is the one worth reading.
    visible = sorted(visible, key=lambda pair: (pair[1], pair[0]["score"]), reverse=True)
    drawn_chips: list[tuple[int, int, int, int]] = []

    for item, alpha in visible:
        x1, y1, x2, y2 = (int(round(v)) for v in item["box"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        colour = colour_for(item["score"], conf_floor)
        _blend_rect(frame, x1, y1, x2, y2, colour, _BOX_FILL_ALPHA * alpha)

        # Corner length scales with the box so small detections do not become solid rectangles.
        corner = max(6, min((x2 - x1) // 4, (y2 - y1) // 4, 34))
        faded = tuple(int(round(c * alpha + b * (1 - alpha)))
                      for c, b in zip(colour, (60, 60, 60)))
        for cx, dx in ((x1, 1), (x2, -1)):
            for cy, dy in ((y1, 1), (y2, -1)):
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy), faded, thickness, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner), faded, thickness, cv2.LINE_AA)

        if alpha < 0.55:
            continue  # a nearly-gone box keeps its brackets but drops the text, which would smear

        # With one class the name is on every chip and distinguishes nothing, while costing the
        # width that makes clustered labels collide. The class is in the dashboard heading instead.
        name = "" if single_class else (
            names.get(item["cls"], str(item["cls"])) if isinstance(names, dict) else str(item["cls"]))
        parts = ([f"#{item['id']}"] if item["id"] is not None else []) + \
                ([name] if name else []) + [f"{item['score']:.2f}"]
        label = " ".join(parts)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        chip_h = text_h + baseline + 8
        chip_w = text_w + 12

        # Above the box by preference, below it if that would clip the top of the frame, and
        # always pulled back inside the frame on both axes.
        chip_top = y1 - chip_h - 2 if y1 - chip_h - 2 >= 0 else y2 + 2
        chip_top = max(0, min(chip_top, height - chip_h))
        chip_left = max(0, min(x1, width - chip_w))
        chip = (chip_left, chip_top, chip_left + chip_w, chip_top + chip_h)

        # Overlapping chips turn a cluster of detections into unreadable mush, which is worse than
        # showing fewer numbers. The brackets still show every detection; only the text is dropped.
        if any(chip[0] < d[2] and d[0] < chip[2] and chip[1] < d[3] and d[1] < chip[3]
               for d in drawn_chips):
            continue
        drawn_chips.append(chip)

        _blend_rect(frame, chip[0], chip[1], chip[2], chip[3], colour, 0.9 * alpha)
        _text(frame, label, (chip_left + 6, chip_top + text_h + 4), 0.52, (255, 255, 255), 1)


def draw_dashboard(frame: np.ndarray, unique: int, position_s: float, duration_s: float,
                   conf: float, heading: str = "DETECTIONS") -> None:
    """The translucent card top-left: what was found, how far in, at what threshold.

    Top-left is already occupied on dashcam footage — the gaziantep clip burns a vehicle counter
    there — so the card is opaque enough to sit over whatever is underneath rather than fighting
    it. `unique` is the same confirmed-track count run() logs, so the number on screen and the
    number in the summary cannot drift apart.

    `heading` comes from the model's own class name rather than being hardcoded. The current
    weights detect `road_damage` — cracks and patches as well as holes — and a card reading
    "POTHOLES FOUND" over a box on a crack is a claim the footage visibly contradicts.
    """
    pad, card_w, card_h = _DASH_PAD, _DASH_CARD_W, _DASH_CARD_H
    _blend_rect(frame, pad, pad, pad + card_w, pad + card_h, _PANEL_BG, _PANEL_ALPHA)
    cv2.rectangle(frame, (pad, pad), (pad + card_w, pad + card_h), (90, 82, 78), 1, cv2.LINE_AA)

    _text(frame, heading[:24], (pad + 16, pad + 26), 0.46, (168, 190, 205), 1)
    _text(frame, str(unique), (pad + 16, pad + 74), 1.5, _INK, 3, cv2.FONT_HERSHEY_DUPLEX)

    clock = f"{format_clock(position_s)} / {format_clock(duration_s)}" if duration_s else format_clock(position_s)
    _text(frame, clock, (pad + 132, pad + 58), 0.6, _INK, 1)
    _text(frame, f"conf {conf:.2f}", (pad + 132, pad + 84), 0.5, (168, 190, 205), 1)


def draw_detection_log(frame: np.ndarray, cards: list[dict], top: int, scroll: float) -> None:
    """Left-side scrolling history: one row per confirmed detection, newest at the bottom.

    A SEPARATE, unrelated piece of state from OverlayStore — that governs the fading corner-bracket
    boxes only; these cards persist for the rest of the video once created, never fade, and are not
    coupled to hold-frames.

    `scroll` is a row-offset eased toward `max(0, len(cards) - capacity)` once per frame in run()'s
    main loop; this function just places each card at its current (possibly fractional, possibly
    only partially inside the panel) position. That's the whole animation — position eases smoothly
    frame to frame, this function has no notion of "animating" on its own.
    """
    height = frame.shape[0]
    bottom = height - _LOG_BOTTOM_MARGIN
    if bottom <= top or not cards:
        return

    row_pitch = _LOG_THUMB + _LOG_ROW_GAP
    pad = _DASH_PAD
    info_x = pad + _LOG_THUMB + 8
    info_w = _LOG_CARD_W - _LOG_THUMB - 8
    stripe_w = 5

    # One extra row of margin below `scroll`'s integer part, so a card mid-transition is never
    # skipped; breaks out the moment a row would start below the viewport, so this never scans the
    # full history once it's long.
    first = max(0, int(scroll) - 1)
    for i in range(first, len(cards)):
        y = top + round((i - scroll) * row_pitch)
        if y >= bottom:
            break
        row_bottom = y + _LOG_THUMB
        if row_bottom <= top:
            continue

        # Clipped to the panel's OWN viewport, not just the frame's edges — this is what stops a
        # row sliding past top/bottom from visually bleeding into the dashboard card above or the
        # timeline bar below.
        draw_y0, draw_y1 = max(y, top), min(row_bottom, bottom)
        if draw_y1 <= draw_y0:
            continue

        card = cards[i]
        # A raw slice-assignment doesn't self-clamp like _blend_rect does — a negative start index
        # would silently wrap instead of raising, so the thumb's source slice is clamped by hand.
        frame[draw_y0:draw_y1, pad:pad + _LOG_THUMB] = card["thumb"][draw_y0 - y:draw_y1 - y]

        _blend_rect(frame, info_x, draw_y0, info_x + info_w, draw_y1, _PANEL_BG, _LOG_PANEL_ALPHA)
        _blend_rect(frame, info_x, draw_y0, info_x + stripe_w, draw_y1, card["colour"], 1.0)

        # The border and text only render once a row has fully settled inside the viewport — a
        # half-drawn border or a vertically-clipped glyph mid-slide reads as a rendering bug rather
        # than motion, so those simply pop in once the row stops moving.
        if y >= top and row_bottom <= bottom:
            cv2.rectangle(frame, (pad, y), (pad + _LOG_THUMB, row_bottom), card["colour"], 3,
                         cv2.LINE_AA)
            text_x = info_x + stripe_w + 10
            _text(frame, f"Kamera: {card['camera'][:24]}", (text_x, y + 30), 0.52, _INK, 1)
            _text(frame, f"Saat: {card['time_str']}", (text_x, y + 66), 0.52, _INK, 1)
            _text(frame, f"Konum: {card['location'][:24]}", (text_x, y + 102), 0.52, _INK, 1)


def draw_timeline(frame: np.ndarray, position: float, marks: list[float]) -> None:
    """Progress bar along the bottom, ticked wherever a new pothole was first seen.

    Single pass, so only marks already reached are drawn — the bar fills and the ticks accumulate
    as the video plays. Cheaper than a two-pass render, and it reads better anyway: evidence
    building up rather than the answer given away in the first frame.
    """
    height, width = frame.shape[:2]
    left, right = 20, width - 20
    bar_y = height - 26
    span = right - left

    _blend_rect(frame, left, bar_y, right, bar_y + 7, (40, 38, 36), 0.75)
    filled = left + int(span * min(1.0, max(0.0, position)))
    _blend_rect(frame, left, bar_y, filled, bar_y + 7, (90, 200, 255), 0.95)

    for mark in marks:
        x = left + int(span * min(1.0, max(0.0, mark)))
        cv2.line(frame, (x, bar_y - 9), (x, bar_y - 2), _COLOUR_HIGH, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- io


def open_writer(path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open the output video, or exit saying which knob fixes it.

    cv2 has no imencode equivalent for video, so the non-ASCII path problem that
    extract_frames.save_image works around cannot be worked around here — it can only be warned
    about. A VideoWriter given a path with non-ASCII characters on Windows reports success and
    then writes an empty file.
    """
    if not str(path).isascii():
        logger.warning(
            "--out contains non-ASCII characters (%s). On Windows cv2.VideoWriter can silently "
            "produce an empty file for such paths — if the result is 0 bytes, write to an "
            "ASCII-only path and move it afterwards", path,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        writer.release()
        logger.error(
            "could not open %s for writing with fourcc %r. This OpenCV build may lack that "
            "encoder — try --fourcc XVID with an .avi output, or --fourcc avc1",
            path, fourcc,
        )
        raise SystemExit(2)
    return writer


# ---------------------------------------------------------------- preflight


def validate_args(args: argparse.Namespace) -> None:
    """Reject impossible arguments before decoding anything, naming the fix in the message."""
    if not 0.0 <= args.conf < 1.0:
        logger.error("--conf must be in [0, 1) (got %s)", args.conf)
        raise SystemExit(2)
    if not 0.0 < args.iou <= 1.0:
        logger.error("--iou must be in (0, 1] (got %s)", args.iou)
        raise SystemExit(2)
    if args.imgsz <= 0 or args.imgsz % 32:
        logger.error("--imgsz must be a positive multiple of 32 (got %s)", args.imgsz)
        raise SystemExit(2)
    if args.stretch_size <= 0:
        logger.error("--stretch-size must be greater than 0 (got %s)", args.stretch_size)
        raise SystemExit(2)
    if args.start < 0:
        logger.error("--start must be 0 or greater (got %s)", args.start)
        raise SystemExit(2)
    if args.end is not None and args.end <= args.start:
        logger.error("--end (%s) must be greater than --start (%s)", args.end, args.start)
        raise SystemExit(2)
    if args.max_frames is not None and args.max_frames <= 0:
        logger.error("--max-frames must be greater than 0 (got %s)", args.max_frames)
        raise SystemExit(2)
    if args.every < 1:
        logger.error("--every must be 1 or greater (got %s)", args.every)
        raise SystemExit(2)
    if args.hold_frames < 0:
        logger.error("--hold-frames must be 0 or greater (got %s); 0 disables the hold",
                     args.hold_frames)
        raise SystemExit(2)
    if args.min_track_hits < 0:
        logger.error("--min-track-hits must be 0 or greater (got %s); 0 or 1 disables it",
                     args.min_track_hits)
        raise SystemExit(2)
    if not 0.0 < args.suppress_overlap <= 1.0:
        logger.error("--suppress-overlap must be in (0, 1] (got %s)", args.suppress_overlap)
        raise SystemExit(2)
    if not 0.0 <= args.roi_top < 1.0:
        logger.error("--roi-top must be in [0, 1) (got %s); 0 disables it", args.roi_top)
        raise SystemExit(2)
    if not 0.0 <= args.suppress_conf < 1.0:
        logger.error("--suppress-conf must be in [0, 1) (got %s)", args.suppress_conf)
        raise SystemExit(2)
    if args.suppress_every < 1:
        logger.error("--suppress-every must be 1 or greater (got %s)", args.suppress_every)
        raise SystemExit(2)
    if len(args.fourcc) != 4:
        logger.error("--fourcc must be exactly 4 characters (got %r)", args.fourcc)
        raise SystemExit(2)


def prepare_output(path: Path, overwrite: bool) -> None:
    """Refuse to clobber an existing render without being told to.

    Same bargain as extract_frames' --append and split_dataset's --overwrite: a long run that
    silently replaces the previous one is only noticed once the previous one is wanted back.
    """
    if path.is_dir():
        logger.error("--out is a folder, not a file: %s", path)
        raise SystemExit(2)
    if path.exists() and not overwrite:
        logger.error("%s already exists. Pick another --out, or pass --overwrite to replace it",
                     path)
        raise SystemExit(2)


def resolve_veto_classes(model, spec: str) -> list[int]:
    """Turn a comma-separated list of COCO class names into the ids that model uses.

    Names rather than raw ids in the CLI: `--suppress-classes car,truck` says what it does, while
    `--suppress-classes 2,7` is a magic number that silently means something else the moment the
    weights change.
    """
    wanted = [part.strip().lower() for part in spec.split(",") if part.strip()]
    lookup = {str(name).lower(): int(idx) for idx, name in model.names.items()}
    unknown = [name for name in wanted if name not in lookup]
    if unknown:
        logger.error(
            "--suppress-classes lists %s, which %s does not have. Available: %s",
            ", ".join(unknown), Path(getattr(model, "ckpt_path", "the model")).name,
            ", ".join(sorted(lookup)),
        )
        raise SystemExit(2)
    return [lookup[name] for name in wanted]


def load_model(weights: Path, device: str, label: str = "weights"):
    """Load the detector and settle on a device, degrading to CPU rather than dying."""
    if not weights.is_file():
        logger.error(
            "weights not found: %s\nTrain a model first (see notebooks/gaziantep_demo.ipynb) or "
            "point --weights at an existing .pt/.engine", weights,
        )
        raise SystemExit(2)

    from ultralytics import YOLO

    resolved = device
    if device != "cpu":
        import torch
        if not torch.cuda.is_available():
            logger.warning("no CUDA device available — falling back to CPU. Expect this to take "
                           "roughly an order of magnitude longer")
            resolved = "cpu"

    model = YOLO(str(weights))
    task = getattr(model, "task", None)
    if task != "detect":
        logger.warning("%s reports task=%r, not 'detect' — boxes may not be what you expect",
                       label, task)
    logger.info("%-7s: %s (task=%s, %d class(es): %s)",
                label, weights, task, len(model.names), list(model.names.values())[:8])
    return model, resolved


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    validate_args(args)

    video_path = Path(args.video)
    out_path = output_path_for(video_path, args.out)
    prepare_output(out_path, args.overwrite)

    model, device = load_model(Path(args.weights), args.device, "detect")

    # `none` skips loading the second model entirely rather than loading it and ignoring it.
    suppressor = None
    veto_class_ids: list[int] = []
    shadow_class_ids: list[int] = []
    if not args.no_filters and args.suppress_classes.strip().lower() != "none":
        suppressor, _ = load_model(Path(args.suppress_weights), args.device, "veto")
        veto_class_ids = resolve_veto_classes(suppressor, args.suppress_classes)
        # The shadow set must be a SUBSET of what the suppressor is asked to detect, or the grown
        # boxes would never appear in its output and the rule would silently do nothing.
        if args.shadow_extend > 0 and args.shadow_classes.strip().lower() != "none":
            shadow_class_ids = [c for c in resolve_veto_classes(suppressor, args.shadow_classes)
                                if c in veto_class_ids]
            missing = (set(resolve_veto_classes(suppressor, args.shadow_classes))
                       - set(veto_class_ids))
            if missing:
                logger.warning("--shadow-classes lists %s, which --suppress-classes does not "
                               "detect — those are ignored",
                               [suppressor.names[c] for c in sorted(missing)])

    if args.no_filters:
        # One flag to turn every rule off, so a before/after comparison is a single edit and the
        # filters can be proven to be doing exactly what the funnel claims.
        args.min_track_hits = 0
        args.roi_top = 0.0

    cap = open_video(video_path)
    writer = None
    try:
        source_fps = resolve_source_fps(cap.get(cv2.CAP_PROP_FPS), args.source_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        start_frame = int(args.start * source_fps)
        end_frame = int(args.end * source_fps) if args.end is not None else None
        if total_frames and start_frame >= total_frames:
            logger.error("--start %.3fs is past the end of the video (%.3fs)",
                         args.start, total_frames / source_fps)
            raise SystemExit(2)

        # Keep the output playing at real time when frames are skipped, instead of producing a
        # silent fast-forward that misrepresents how fast the vehicle was moving.
        out_fps = source_fps / args.every
        stretch = args.preprocess == "stretch"
        infer_size = (args.stretch_size, args.stretch_size) if stretch else (width, height)

        duration = f"{total_frames / source_fps:.2f}s" if total_frames else "unknown"
        logger.info("video  : %s", video_path)
        logger.info("source : %dx%d @ %.3f fps, %s, %s frames",
                    width, height, source_fps, duration, total_frames or "?")
        if stretch:
            logger.info("infer  : frames stretched to %dx%d, then boxes mapped back "
                        "(matches a Roboflow square export; see --preprocess)",
                        args.stretch_size, args.stretch_size)
        else:
            logger.info("infer  : native %dx%d, letterboxed by Ultralytics", width, height)
        logger.info("model  : imgsz=%d conf=%.2f iou=%.2f device=%s tracking=%s",
                    args.imgsz, args.conf, args.iou, device,
                    "off" if args.no_track else args.tracker)
        if args.no_filters:
            logger.info("rules  : none (--no-filters) — every raw detection is kept")
        else:
            veto_note = (f"not >{args.suppress_overlap * 100:.0f}% inside {args.suppress_classes}"
                         if suppressor is not None else "vehicle veto off")
            logger.info("rules  : track seen >=%d frame(s), centre below y=%.2f, %s",
                        args.min_track_hits, args.roi_top, veto_note)
        logger.info("display: %s, hold %s (cosmetic only — reported counts are raw)",
                    "boxes only" if args.no_dashboard else "dashboard",
                    f"{args.hold_frames} frame(s)" if args.hold_frames else "off")
        logger.info("overlay: detection log %s, camera=%r, location=%r",
                    "off" if args.no_detection_log else "on", args.camera_name, args.location)
        logger.info("output : %s (%s @ %.3f fps)", out_path, args.fourcc, out_fps)

        if args.every > 1 and not args.no_track:
            logger.warning("--every %d skips frames, and the tracker assumes consecutive ones — "
                           "ids will break up. Use --every 1 for clean tracks", args.every)

        writer = open_writer(out_path, args.fourcc, out_fps, (width, height))

        processed = 0
        frames_with_detections = 0
        total_detections = 0
        tracked_detections = 0
        confidences: list[float] = []
        seen_ids: set[int] = set()
        interrupted = False
        started = time.perf_counter()

        total_raw = 0
        drop_totals = {"transient": 0, "vehicle": 0, "shadow": 0, "off_road": 0}
        hit_counter = TrackHitCounter()
        # "road_damage" -> "ROAD DAMAGE FOUND". Taken from the weights so the card never claims to
        # be counting something the model was not trained to detect.
        dashboard_heading = f"{str(model.names[0]).replace('_', ' ').upper()} FOUND"
        veto_boxes = np.zeros((0, 4))    # carried between frames; see --suppress-every
        shadow_boxes = np.zeros((0, 4))  # same, but grown downward for shadow-casting classes

        # Display state, kept deliberately separate from the counters above: nothing below this
        # line is allowed to feed the summary run() prints.
        overlays = OverlayStore(args.hold_frames)
        timeline_marks: list[float] = []
        detection_log: list[dict] = []
        duration_s = total_frames / source_fps if total_frames else 0.0

        # Fixed for the whole run (depend only on args and the frame size), so computed once
        # rather than on every frame.
        log_top = (_DASH_PAD + _DASH_CARD_H + _LOG_TOP_GAP) if not args.no_dashboard else _DASH_PAD
        log_row_pitch = _LOG_THUMB + _LOG_ROW_GAP
        log_capacity = max(0, (height - log_top - _LOG_BOTTOM_MARGIN) // log_row_pitch)
        log_scroll = 0.0
        # Exponential smoothing: reaches ~95% of the way to a new target within --log-anim-seconds,
        # regardless of the output fps. 0 (or negative) disables the animation outright — scroll
        # snaps straight to target every frame, matching the old hard-cut behaviour.
        log_ease_rate = (1.0 if args.log_anim_seconds <= 0
                         else 1.0 - 0.05 ** ((1.0 / out_fps) / args.log_anim_seconds))

        try:
            for index, frame in iter_sampled_frames(
                cap, float(args.every), start_frame, end_frame, args.max_frames
            ):
                image = cv2.resize(frame, infer_size, interpolation=cv2.INTER_LINEAR) if stretch else frame

                if args.no_track:
                    result = model.predict(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                           device=device, verbose=False)[0]
                else:
                    # persist=True is what carries tracker state from one call to the next; drop
                    # it and every frame starts a fresh set of ids.
                    result = model.track(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                         device=device, tracker=args.tracker, persist=True,
                                         verbose=False)[0]

                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                # .id is None whenever the tracker is off or has nothing associated yet.
                raw_ids = result.boxes.id
                track_ids: list[int | None] = (
                    [int(v) for v in raw_ids.cpu().numpy()] if raw_ids is not None
                    else [None] * len(boxes)
                )
                if stretch:
                    boxes = scale_boxes(boxes, infer_size, (width, height))

                # The rules run on NATIVE-resolution boxes, because the veto boxes they are
                # compared against come from a model fed the native frame (see below).
                raw_count = len(boxes)
                if suppressor is not None and processed % args.suppress_every == 0:
                    # Deliberately NOT the stretched square: yolo26n was trained on undistorted
                    # images, so handing it the 512x512 stretch would be the exact domain shift this
                    # script exists to avoid, applied to the wrong model. The two detectors get
                    # different inputs on purpose, and meet again in native coordinates.
                    veto = suppressor.predict(frame, imgsz=args.imgsz, conf=args.suppress_conf,
                                              classes=veto_class_ids, device=device,
                                              verbose=False)[0]
                    # A second set, only for the shadow-casting classes, grown downward to cover
                    # the ground their shadow falls on.
                    veto_cls = veto.boxes.cls.cpu().numpy().astype(int)
                    shadow_boxes = extend_downward(
                        veto.boxes.xyxy.cpu().numpy()[np.isin(veto_cls, shadow_class_ids)],
                        args.shadow_extend, height,
                    ) if len(shadow_class_ids) else np.zeros((0, 4))
                    # Cached and reused for the next --suppress-every frames. A second detector on
                    # every frame doubles the runtime to buy almost nothing: at 30 fps a car moves a
                    # handful of pixels between frames, and the veto is a 50%-overlap test, not a
                    # precise one. Re-running it three times a second is plenty.
                    veto_boxes = veto.boxes.xyxy.cpu().numpy()

                hit_counter.record(track_ids)
                boxes, scores, classes, track_ids, dropped = apply_rules(
                    boxes, scores, classes, track_ids, hit_counter, veto_boxes,
                    args.min_track_hits, args.suppress_overlap, args.roi_top, height,
                    shadow_boxes,
                )
                for key, value in dropped.items():
                    drop_totals[key] += value
                total_raw += raw_count

                confirmed = sum(1 for i in track_ids if i is not None)
                fresh_ids = {i for i in track_ids if i is not None} - seen_ids
                seen_ids.update(i for i in track_ids if i is not None)

                position = index / source_fps
                if fresh_ids and total_frames:
                    # One tick the first time a track appears, not one per frame it persists.
                    timeline_marks.append(index / total_frames)

                if fresh_ids and not args.no_detection_log:
                    # track_ids is parallel to boxes/scores, not indexed by id — build the lookup
                    # once per frame rather than scanning it per id.
                    id_to_index = {tid: i for i, tid in enumerate(track_ids) if tid is not None}
                    # Sorted for a deterministic append order on the rare frame where more than one
                    # track is confirmed at once — not "newest id first", just a stable tie-break.
                    for track_id in sorted(fresh_ids):
                        i = id_to_index.get(track_id)
                        if i is None:
                            continue
                        crop = crop_box(frame, boxes[i])
                        if crop is None:
                            continue
                        detection_log.append({
                            "thumb": make_thumbnail(crop, _LOG_THUMB),
                            "colour": colour_for(float(scores[i]), args.conf),
                            "camera": args.camera_name,
                            "time_str": format_clock_hms(position),
                            "location": args.location,
                        })
                    if len(detection_log) > _LOG_MAX_CARDS:
                        del detection_log[: len(detection_log) - _LOG_MAX_CARDS]

                overlays.update(index, boxes, scores, track_ids, classes)
                draw_detections(frame, overlays.visible(index), model.names, args.conf)
                if not args.no_dashboard:
                    draw_dashboard(frame, len(seen_ids), position, duration_s, args.conf,
                                   dashboard_heading)
                    if total_frames:
                        draw_timeline(frame, index / total_frames, timeline_marks)
                if not args.no_detection_log:
                    # Eased every frame, not just when a new card lands — the slide keeps moving
                    # for several frames after the trigger, until scroll catches up with target.
                    log_target = max(0.0, len(detection_log) - log_capacity)
                    log_scroll += (log_target - log_scroll) * log_ease_rate
                    draw_detection_log(frame, detection_log, log_top, log_scroll)
                writer.write(frame)

                if args.show:
                    cv2.imshow("predict_video", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("q pressed — stopping early")
                        break

                processed += 1
                total_detections += len(boxes)
                tracked_detections += confirmed
                frames_with_detections += 1 if len(boxes) else 0
                confidences.extend(float(s) for s in scores)

                if processed % _PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - started
                    rate = processed / elapsed if elapsed else 0.0
                    if total_frames:
                        logger.info("%d frames (%.1f%%), %d detection(s) so far, %.1f fps",
                                    processed, 100.0 * index / total_frames, total_detections, rate)
                    else:
                        logger.info("%d frames, %d detection(s) so far, %.1f fps",
                                    processed, total_detections, rate)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("interrupted — keeping the %d frame(s) already written", processed)

        elapsed = time.perf_counter() - started
        if not processed:
            logger.error("no frames were processed — check --start/--end against the video length")
            return 1

        logger.info("")
        logger.info("done — %s", out_path)
        logger.info("  frames processed : %d in %.1fs (%.1f fps)",
                    processed, elapsed, processed / elapsed if elapsed else 0.0)
        logger.info("  frames with a detection : %d (%.1f%%)",
                    frames_with_detections, 100.0 * frames_with_detections / processed)
        # The funnel rather than a bare total: a smaller number on its own says nothing about which
        # detections went, and these rules are only defensible if their effect stays visible.
        dropped_total = sum(drop_totals.values())
        if dropped_total:
            logger.info("  detections : %d raw -> %d kept", total_raw, total_detections)
            logger.info("    dropped, track seen < %d frames : %d",
                        args.min_track_hits, drop_totals["transient"])
            logger.info("    dropped, on a vehicle/person    : %d", drop_totals["vehicle"])
            logger.info("    dropped, in a shadow zone       : %d", drop_totals["shadow"])
            logger.info("    dropped, above the road         : %d", drop_totals["off_road"])
        else:
            logger.info("  detections : %d total", total_detections)
        if not args.no_track:
            logger.info("  unique tracks : %d", len(seen_ids))
            # A low share here means the detector rarely holds an object across two consecutive
            # frames, which is worth knowing: it caps how useful tracking can be, and it is the
            # signature of an unstable model rather than of a tracker that needs tuning.
            share = 100.0 * tracked_detections / total_detections if total_detections else 0.0
            logger.info("  detections in a confirmed track : %d of %d (%.0f%%)",
                        tracked_detections, total_detections, share)
            if total_detections and share < 60.0:
                logger.warning("  most detections never survived into a confirmed track — the "
                               "detector is firing on single frames and dropping out")
        if confidences:
            logger.info("  confidence : min %.2f, median %.2f, max %.2f",
                        min(confidences), statistics.median(confidences), max(confidences))
        else:
            logger.warning("  nothing was detected at conf=%.2f — try lowering it, or check that "
                           "--preprocess matches how the training images were resized", args.conf)
        if interrupted:
            logger.warning("  the render is incomplete (interrupted)")
        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------- cli


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run trained YOLO weights over a recorded video and write an annotated copy.",
    )
    p.add_argument("--video", required=True, help="path to the video file to run over")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="detector weights (.pt or .engine); defaults to the gaziantep pothole "
                        "model under notebooks/runs")
    p.add_argument("--out", default=None,
                   help="output video path (default: <video name>_detected.mp4 in the current "
                        "folder; the source video is never modified)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help=f"confidence threshold (default {DEFAULT_CONF}). Tuned for the bundled "
                        f"--weights, whose F1 peaks there on 601 held-out instances. The previous "
                        f"model peaked at 0.85, so this does NOT transfer: re-tune whenever "
                        f"--weights changes (section 6 of notebooks/gaziantep_demo.ipynb prints it)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold (default 0.5). Lower than Ultralytics' 0.7 because this "
                        "model puts several overlapping boxes on one patch of broken asphalt; at "
                        "0.7 they all survive and stack up on screen. This is real NMS, not a "
                        "drawing tweak — it changes the detections, and the reported counts with them")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                   help=f"inference size; match what the model was trained at "
                        f"(default {DEFAULT_IMGSZ})")
    p.add_argument("--device", default="0",
                   help="CUDA device index or 'cpu' (default 0; falls back to cpu if no GPU)")
    p.add_argument("--preprocess", choices=("stretch", "native"), default="native",
                   help="'native' passes the frame through and lets Ultralytics letterbox it, "
                        "which is right whenever the training images kept their source aspect "
                        "ratio — true of RDD2022 and of our rebuilt 1280x720 frames. 'stretch' "
                        "squashes each frame to a square first and maps the boxes back, which is "
                        "only correct for a model trained on a Roboflow square export such as the "
                        "old gaziantep_pothole_v1.pt (default native)")
    p.add_argument("--stretch-size", type=int, default=DEFAULT_STRETCH_SIZE,
                   help=f"square size for --preprocess stretch; must match the training image "
                        f"size (default {DEFAULT_STRETCH_SIZE})")
    p.add_argument("--min-track-hits", type=int, default=10,
                   help="only keep a detection once its track has been seen this many frames. The "
                        "single most effective noise filter here — on the gaziantep footage it "
                        "removes 56%% of raw detections, all of them one-frame flashes. A detection "
                        "the tracker never confirmed can never reach the threshold, which is "
                        "intended. 1 or 0 disables it (default 3)")
    p.add_argument("--suppress-classes", default=DEFAULT_SUPPRESS_CLASSES,
                   help="COCO class names that veto a detection overlapping them — a pothole cannot "
                        f"be inside a car. 'none' disables the veto and skips loading the second "
                        f"model (default '{DEFAULT_SUPPRESS_CLASSES}')")
    p.add_argument("--suppress-weights", default=str(DEFAULT_SUPPRESS_WEIGHTS),
                   help="COCO detector used for the veto (default: the yolo26n.pt already vendored "
                        "in data_collector/)")
    p.add_argument("--suppress-overlap", type=float, default=0.5,
                   help="drop a detection when more than this fraction OF ITS OWN AREA falls inside "
                        "a vetoed box (default 0.5)")
    p.add_argument("--suppress-conf", type=float, default=0.35,
                   help="confidence threshold for the veto detector (default 0.35)")
    p.add_argument("--shadow-classes", default=DEFAULT_SHADOW_CLASSES,
                   help="classes whose veto box is also extended DOWNWARD, to cover the ground "
                        "their shadow falls on. Deliberately excludes cars: measured on this "
                        "footage, detections near parked cars are overwhelmingly real cracks, and "
                        f"growing car boxes deleted them. 'none' disables "
                        f"(default '{DEFAULT_SHADOW_CLASSES}')")
    p.add_argument("--shadow-extend", type=float, default=1.0,
                   help="how far to extend a shadow-caster's box downward, as a multiple of its "
                        "own height — scaling by the box means one value works at every distance. "
                        "0 disables (default 1.0)")
    p.add_argument("--suppress-every", type=int, default=1,
                   help="re-run the veto detector every Nth frame, reusing its boxes in between. "
                        "Defaults to 1 (every frame) because caching leaks: at 3 the boxes go two "
                        "frames stale, cars move, and 7 of 544 surviving detections were measured "
                        "still sitting on a vehicle. Raise it to roughly halve the render time if "
                        "you can accept that (default 1)")
    p.add_argument("--roi-top", type=float, default=0.45,
                   help="reject detections whose centre sits above this fraction of frame height — "
                        "potholes are on the road, not in the sky. 0 disables (default 0.45)")
    p.add_argument("--no-filters", action="store_true",
                   help="turn off every rule above and keep all raw detections, for an honest "
                        "before/after comparison (default off)")
    p.add_argument("--hold-frames", type=int, default=8,
                   help="keep drawing a detection for this many frames after the model stops "
                        "emitting it, fading it out. PURELY COSMETIC — the reported detection "
                        "counts always come from raw per-frame output and are unaffected. Exists "
                        "because this detector often fires on a single frame, which strobes on "
                        "playback. 0 disables it (default 8)")
    p.add_argument("--no-dashboard", action="store_true",
                   help="draw only the detection boxes, without the counter card or the timeline "
                        "bar (default off)")
    p.add_argument("--camera-name", default="Zabita Araci 03",
                   help="camera name shown as 'Kamera' on each detection-log card "
                        "(default 'Zabita Araci 03')")
    p.add_argument("--location", default="12.15.124.78",
                   help="location string shown as 'Konum' on each detection-log card — an "
                        "arbitrary placeholder, not a real GPS value (default '12.15.124.78')")
    p.add_argument("--no-detection-log", action="store_true",
                   help="drop the left-side scrolling detection-log panel (thumbnail + "
                        "Kamera/Saat/Konum card per confirmed detection); independent of "
                        "--no-dashboard, which only controls the counter card and timeline bar "
                        "(default off)")
    p.add_argument("--log-anim-seconds", type=float, default=0.3,
                   help="how long the detection-log panel takes to settle after a new card pushes "
                        "it past capacity — an eased slide, not a hard cut. 0 disables the "
                        "animation (instant snap to the latest cards) (default 0.3)")
    p.add_argument("--no-track", action="store_true",
                   help="detect each frame independently instead of tracking objects across "
                        "frames; output will flicker more (default off)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Ultralytics tracker config (default bytetrack.yaml; botsort.yaml is the "
                        "other bundled option)")
    p.add_argument("--start", type=float, default=0.0,
                   help="skip the first N seconds of the video (default 0)")
    p.add_argument("--end", type=float, default=None,
                   help="stop at this many seconds into the video (default: end of file)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after this many frames, whatever the range (default: no cap)")
    p.add_argument("--every", type=int, default=1,
                   help="process every Nth frame for a faster preview; the output fps is divided "
                        "to match so timing is preserved, but tracking degrades (default 1)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="override the video's frame rate, for containers that report a wrong or "
                        "missing one (default: read from the file)")
    p.add_argument("--fourcc", default="mp4v",
                   help="output codec fourcc (default mp4v; try XVID with an .avi output if this "
                        "build cannot open the writer)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace --out if it already exists instead of refusing (default off)")
    p.add_argument("--show", action="store_true",
                   help="display frames in a window while rendering; press q to stop early "
                        "(default off)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
