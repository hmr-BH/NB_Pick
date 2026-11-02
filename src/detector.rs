use crate::image_ops::{PersonRect, pose_crop_person, pose_crop_rect, valid_person};
use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage, imageops};
use ndarray::Array4;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};

const INPUT_SIZE: u32 = 640;
const OUTPUT_VALUES: usize = 85;
const SCORE_THRESHOLD: f32 = 0.25;
const POSE_WIDTH: u32 = 192;
const POSE_HEIGHT: u32 = 256;
const POSE_KEYPOINTS: usize = 133;
const POSE_X_BINS: usize = 384;
const POSE_Y_BINS: usize = 512;

#[derive(Clone, Copy)]
struct Anchor {
    x: f32,
    y: f32,
    stride: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Clone, Copy, Debug)]
struct PoseEvidence {
    valid: bool,
    head: Option<(f32, f32)>,
}

pub struct Detector {
    session: Session,
    pose_session: Session,
    anchors: Vec<Anchor>,
}

impl Detector {
    pub fn load(path: &Path, pose_path: &Path) -> Result<Self> {
        let env = ort::init()
            .with_name("nb-pick")
            .with_telemetry(false)
            .build()
            .context("初始化离线推理引擎失败")?;
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(path)
            .with_context(|| format!("加载人体模型失败：{}", path.display()))?;
        let pose_session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(pose_path)
            .with_context(|| format!("加载人体关键点模型失败：{}", pose_path.display()))?;
        Ok(Self {
            session,
            pose_session,
            anchors: anchors(),
        })
    }

    pub fn detect(&mut self, frame: &RgbImage) -> Result<Vec<PersonRect>> {
        let started = Instant::now();
        let mut candidates = Vec::new();
        for tile in detection_tiles(frame.width(), frame.height()) {
            let crop = imageops::crop_imm(
                frame,
                tile.x as u32,
                tile.y as u32,
                tile.w as u32,
                tile.h as u32,
            )
            .to_image();
            let (input, scale, pad_x, pad_y) = preprocess(&crop);
            let outputs = self
                .session
                .run(ort::inputs![TensorRef::from_array_view(&input)?])?;
            let predictions = tensor(outputs.get("output").context("人体模型缺少 output 输出")?)?;
            decode_people(
                predictions,
                &self.anchors,
                scale,
                pad_x,
                pad_y,
                tile,
                frame.width(),
                frame.height(),
                &mut candidates,
            )?;
        }

        let mut verified = Vec::new();
        for person in nms(candidates, 0.45) {
            let evidence = self.inspect_person(frame, person)?;
            if evidence.valid {
                verified.push((person, evidence.head));
            }
        }
        let mut people = deduplicate_verified(verified);
        people.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
        people.truncate(96);

        // Leave CPU time for camera decoding and UI after tiled inference.
        let elapsed = started.elapsed();
        let delay = if elapsed.as_millis() > 900 {
            Duration::from_millis(80)
        } else if elapsed.as_millis() > 400 {
            Duration::from_millis(40)
        } else {
            Duration::from_millis(20)
        };
        std::thread::sleep(delay);
        Ok(people)
    }

    pub fn verify_person(&mut self, frame: &RgbImage, person: PersonRect) -> Result<bool> {
        Ok(self.inspect_person(frame, person)?.valid)
    }

    fn inspect_person(&mut self, frame: &RgbImage, person: PersonRect) -> Result<PoseEvidence> {
        inspect_pose(frame, person, &mut self.pose_session)
    }
}

fn inspect_pose(
    frame: &RgbImage,
    person: PersonRect,
    pose_session: &mut Session,
) -> Result<PoseEvidence> {
    let crop_rect = pose_crop_rect(frame, person);
    let crop = pose_crop_person(frame, person);
    let resized = imageops::resize(
        &crop,
        POSE_WIDTH,
        POSE_HEIGHT,
        imageops::FilterType::Triangle,
    );
    let mean = [123.675_f32, 116.28, 103.53];
    let std = [58.395_f32, 57.12, 57.375];
    let mut input = Array4::<f32>::zeros((1, 3, POSE_HEIGHT as usize, POSE_WIDTH as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        for channel in 0..3 {
            input[[0, channel, y as usize, x as usize]] =
                (pixel[channel] as f32 - mean[channel]) / std[channel];
        }
    }
    let outputs = pose_session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
    let simcc_x = tensor(
        outputs
            .get("simcc_x")
            .context("人体关键点模型缺少 simcc_x 输出")?,
    )?;
    let simcc_y = tensor(
        outputs
            .get("simcc_y")
            .context("人体关键点模型缺少 simcc_y 输出")?,
    )?;
    if simcc_x.len() != POSE_KEYPOINTS * POSE_X_BINS
        || simcc_y.len() != POSE_KEYPOINTS * POSE_Y_BINS
    {
        bail!("人体关键点模型输出形状不兼容");
    }

    let response = |index: usize| {
        let max_x = max_axis(simcc_x, index, POSE_X_BINS);
        let max_y = max_axis(simcc_y, index, POSE_Y_BINS);
        max_x.min(max_y)
    };
    let mut best_head = (0.0_f32, 0_usize, 0_usize);
    for index in 0..5 {
        let (max_x, x_bin) = axis_max(simcc_x, index, POSE_X_BINS);
        let (max_y, y_bin) = axis_max(simcc_y, index, POSE_Y_BINS);
        let score = max_x.min(max_y);
        if score > best_head.0 {
            best_head = (score, x_bin, y_bin);
        }
    }
    let head = best_head.0;
    let shoulders = (5..7).map(response).fold(0.0_f32, f32::max);
    let hips = (11..13).map(response).fold(0.0_f32, f32::max);
    let knees = (13..15).map(response).fold(0.0_f32, f32::max);
    let valid = pose_has_person(head, shoulders, hips, knees, person.score);
    let head = (head >= 0.26).then(|| {
        let x = crop_rect.x as f32
            + best_head.1 as f32 / 2.0 / POSE_WIDTH as f32 * crop_rect.size as f32;
        let y = crop_rect.y as f32
            + best_head.2 as f32 / 2.0 / POSE_HEIGHT as f32 * crop_rect.size as f32;
        (
            x.clamp(0.0, frame.width().saturating_sub(1) as f32),
            y.clamp(0.0, frame.height().saturating_sub(1) as f32),
        )
    });
    Ok(PoseEvidence { valid, head })
}

fn max_axis(values: &[f32], row: usize, bins: usize) -> f32 {
    axis_max(values, row, bins).0
}

fn axis_max(values: &[f32], row: usize, bins: usize) -> (f32, usize) {
    values[row * bins..(row + 1) * bins]
        .iter()
        .copied()
        .enumerate()
        .fold((0.0_f32, 0), |best, (index, value)| {
            if value > best.0 { (value, index) } else { best }
        })
}

fn pose_has_person(head: f32, shoulders: f32, hips: f32, knees: f32, detector: f32) -> bool {
    head >= 0.26 || (detector >= 0.85 && shoulders >= 0.45 && (hips >= 0.35 || knees >= 0.35))
}

fn deduplicate_verified(mut candidates: Vec<(PersonRect, Option<(f32, f32)>)>) -> Vec<PersonRect> {
    candidates.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    let mut kept: Vec<(PersonRect, (f32, f32))> = Vec::new();
    for (person, head) in candidates {
        let anchor = head.unwrap_or((person.x + person.w / 2.0, person.y + person.h / 2.0));
        let duplicate = kept.iter().any(|(other, other_anchor)| {
            let min_side = person.w.min(person.h).min(other.w.min(other.h)).max(1.0);
            let tolerance = (min_side * 0.20).clamp(8.0, 24.0);
            ((anchor.0 - other_anchor.0).abs() <= tolerance
                && (anchor.1 - other_anchor.1).abs() <= tolerance)
                || looks_like_head_of_body(person, *other)
        });
        if !duplicate {
            kept.push((person, anchor));
        }
    }
    kept.into_iter().map(|(person, _)| person).collect()
}

fn looks_like_head_of_body(a: PersonRect, b: PersonRect) -> bool {
    let (small, large) = if a.w * a.h <= b.w * b.h {
        (a, b)
    } else {
        (b, a)
    };
    let area_ratio = large.w * large.h / (small.w * small.h).max(1.0);
    let center_x = small.x + small.w / 2.0;
    let large_center_x = large.x + large.w / 2.0;
    area_ratio >= 2.0
        && small.h <= large.h * 0.55
        && (center_x - large_center_x).abs() <= large.w * 0.30
        && small.y >= large.y - large.h * 0.18
        && small.y <= large.y + large.h * 0.35
}

fn anchors() -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(8400);
    for stride in [8u32, 16, 32] {
        let side = INPUT_SIZE / stride;
        for row in 0..side {
            for col in 0..side {
                anchors.push(Anchor {
                    x: col as f32,
                    y: row as f32,
                    stride: stride as f32,
                });
            }
        }
    }
    anchors
}

fn preprocess(frame: &RgbImage) -> (Array4<f32>, f32, f32, f32) {
    let scale =
        (INPUT_SIZE as f32 / frame.width() as f32).min(INPUT_SIZE as f32 / frame.height() as f32);
    let w = ((frame.width() as f32 * scale).round() as u32).max(1);
    let h = ((frame.height() as f32 * scale).round() as u32).max(1);
    let resized = imageops::resize(frame, w, h, imageops::FilterType::Triangle);
    let pad_x = (INPUT_SIZE - w) / 2;
    let pad_y = (INPUT_SIZE - h) / 2;
    let mut canvas = RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb([114, 114, 114]));
    imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (x, y, pixel) in canvas.enumerate_pixels() {
        // YOLOX's exported COCO model expects BGR, unnormalised 0..255 input.
        for channel in 0..3 {
            input[[0, channel, y as usize, x as usize]] = pixel[2 - channel] as f32;
        }
    }
    (input, scale, pad_x as f32, pad_y as f32)
}

#[allow(clippy::too_many_arguments)]
fn decode_people(
    predictions: &[f32],
    anchors: &[Anchor],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    tile: Tile,
    frame_width: u32,
    frame_height: u32,
    output: &mut Vec<PersonRect>,
) -> Result<()> {
    if predictions.len() != anchors.len() * OUTPUT_VALUES {
        bail!(
            "人体模型输出形状不兼容：期望 {} 个值，实际 {}",
            anchors.len() * OUTPUT_VALUES,
            predictions.len()
        );
    }
    for (index, anchor) in anchors.iter().enumerate() {
        let base = index * OUTPUT_VALUES;
        // COCO class zero is person; all other classes are intentionally ignored.
        let score = predictions[base + 4] * predictions[base + 5];
        if !score.is_finite() || score < SCORE_THRESHOLD {
            continue;
        }
        let cx = (predictions[base] + anchor.x) * anchor.stride;
        let cy = (predictions[base + 1] + anchor.y) * anchor.stride;
        let w = predictions[base + 2].exp() * anchor.stride;
        let h = predictions[base + 3].exp() * anchor.stride;
        let x1 = (cx - w / 2.0 - pad_x) / scale + tile.x;
        let y1 = (cy - h / 2.0 - pad_y) / scale + tile.y;
        let x2 = (cx + w / 2.0 - pad_x) / scale + tile.x;
        let y2 = (cy + h / 2.0 - pad_y) / scale + tile.y;

        let left = x1.max(tile.x).max(0.0);
        let top = y1.max(tile.y).max(0.0);
        let right = x2.min(tile.x + tile.w).min(frame_width as f32);
        let bottom = y2.min(tile.y + tile.h).min(frame_height as f32);
        let person = PersonRect {
            x: left,
            y: top,
            w: right - left,
            h: bottom - top,
            score,
        };
        if valid_person(person, frame_width, frame_height) {
            output.push(person);
        }
    }
    Ok(())
}

fn detection_tiles(width: u32, height: u32) -> Vec<Tile> {
    let full = Tile {
        x: 0.0,
        y: 0.0,
        w: width as f32,
        h: height as f32,
    };
    if width < 960 || height < 540 {
        return vec![full];
    }

    let columns = if width >= 1600 { 3 } else { 2 };
    let tile_width = (width as f32 * if columns == 3 { 0.40 } else { 0.60 }).ceil() as u32;
    let tile_height = (height as f32 * 0.60).ceil() as u32;
    let xs = tile_positions(width, tile_width, columns);
    let ys = tile_positions(height, tile_height, 2);
    let mut tiles = vec![full];
    for y in ys {
        for x in &xs {
            tiles.push(Tile {
                x: *x as f32,
                y: y as f32,
                w: tile_width as f32,
                h: tile_height as f32,
            });
        }
    }
    tiles
}

fn tile_positions(full: u32, tile: u32, count: usize) -> Vec<u32> {
    let last = full.saturating_sub(tile);
    match count {
        1 => vec![0],
        2 => vec![0, last],
        3 => vec![0, last / 2, last],
        _ => unreachable!(),
    }
}

fn tensor(value: &ort::value::DynValue) -> Result<&[f32]> {
    Ok(value.try_extract_tensor::<f32>()?.1)
}

fn intersection(a: PersonRect, b: PersonRect) -> f32 {
    let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
    let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
    if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h }
}

fn iou(a: PersonRect, b: PersonRect) -> f32 {
    let overlap = intersection(a, b);
    if overlap <= 0.0 {
        return 0.0;
    }
    overlap / (a.w * a.h + b.w * b.h - overlap)
}

fn is_duplicate(a: PersonRect, b: PersonRect, threshold: f32) -> bool {
    let overlap = intersection(a, b);
    let smaller_area = (a.w * a.h).min(b.w * b.h);
    iou(a, b) > threshold || (smaller_area > 0.0 && overlap / smaller_area > 0.75)
}

fn nms(mut candidates: Vec<PersonRect>, threshold: f32) -> Vec<PersonRect> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<PersonRect> = Vec::new();
    for person in candidates {
        if kept
            .iter()
            .all(|other| !is_duplicate(person, *other, threshold))
        {
            kept.push(person);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(x: f32, y: f32, w: f32, h: f32, score: f32) -> PersonRect {
        PersonRect { x, y, w, h, score }
    }

    #[test]
    fn nms_keeps_distinct_people_and_removes_duplicates() {
        let a = person(10.0, 10.0, 50.0, 100.0, 0.9);
        let b = person(12.0, 12.0, 50.0, 100.0, 0.8);
        let c = person(150.0, 10.0, 50.0, 100.0, 0.85);
        assert_eq!(nms(vec![a, b, c], 0.45).len(), 2);
    }

    #[test]
    fn nms_removes_partial_tile_duplicates() {
        let complete = person(100.0, 100.0, 80.0, 180.0, 0.8);
        let partial = person(100.0, 100.0, 80.0, 150.0, 0.6);
        assert_eq!(nms(vec![complete, partial], 0.45).len(), 1);
    }

    #[test]
    fn nms_keeps_nearby_but_distinct_people() {
        let left = person(100.0, 100.0, 40.0, 76.0, 0.6);
        let right = person(112.0, 100.0, 45.0, 55.0, 0.7);
        assert_eq!(nms(vec![left, right], 0.45).len(), 2);
    }

    #[test]
    fn hdpi_1080p_uses_full_frame_and_six_overlapping_tiles() {
        let tiles = detection_tiles(1920, 1080);
        assert_eq!(tiles.len(), 7);
        assert_eq!(tiles[0].w, 1920.0);
        assert!(tiles.iter().skip(1).all(|tile| tile.x + tile.w <= 1920.0));
        assert!(tiles.iter().skip(1).all(|tile| tile.y + tile.h <= 1080.0));
        assert!(tiles.iter().any(|tile| tile.x == 0.0));
        assert!(tiles.iter().any(|tile| tile.x + tile.w == 1920.0));
    }

    #[test]
    fn small_frames_use_one_inference() {
        assert_eq!(detection_tiles(640, 480).len(), 1);
        assert_eq!(detection_tiles(1280, 720).len(), 5);
    }

    #[test]
    fn decoder_ignores_non_person_classes() {
        let anchors = anchors();
        let mut predictions = vec![0.0; anchors.len() * OUTPUT_VALUES];
        predictions[2] = 4.0_f32.ln();
        predictions[3] = 8.0_f32.ln();
        predictions[4] = 0.9;
        predictions[6] = 0.9;
        let tile = Tile {
            x: 0.0,
            y: 0.0,
            w: 640.0,
            h: 640.0,
        };
        let mut people = Vec::new();
        decode_people(
            &predictions,
            &anchors,
            1.0,
            0.0,
            0.0,
            tile,
            640,
            640,
            &mut people,
        )
        .unwrap();
        assert!(people.is_empty());
    }

    #[test]
    fn pose_verifier_rejects_clothing_shape_without_a_head() {
        assert!(!pose_has_person(0.201, 0.359, 0.271, 0.0, 0.95));
        assert!(!pose_has_person(0.203, 0.223, 0.092, 0.0, 0.62));
    }

    #[test]
    fn pose_verifier_accepts_seated_and_headless_real_people() {
        assert!(pose_has_person(0.275, 0.15, 0.0, 0.0, 0.82));
        assert!(pose_has_person(0.0, 0.55, 0.42, 0.0, 0.90));
    }

    #[test]
    fn verified_duplicate_boxes_on_one_head_are_counted_once() {
        let candidates = vec![
            (person(100.0, 80.0, 60.0, 130.0, 0.9), Some((130.0, 105.0))),
            (person(96.0, 76.0, 68.0, 150.0, 0.8), Some((132.0, 107.0))),
            (person(90.0, 70.0, 80.0, 180.0, 0.7), Some((129.0, 104.0))),
        ];
        assert_eq!(deduplicate_verified(candidates).len(), 1);
    }

    #[test]
    fn deduplication_keeps_adjacent_people_separate() {
        let candidates = vec![
            (person(100.0, 80.0, 20.0, 70.0, 0.9), Some((110.0, 95.0))),
            (person(122.0, 80.0, 20.0, 70.0, 0.8), Some((132.0, 95.0))),
        ];
        assert_eq!(deduplicate_verified(candidates).len(), 2);
    }

    #[test]
    fn deduplication_merges_head_box_with_same_full_body_box() {
        let candidates = vec![
            (
                person(1324.0, 640.0, 155.0, 241.0, 0.85),
                Some((1379.0, 697.0)),
            ),
            (
                person(1395.0, 623.0, 35.0, 38.0, 0.48),
                Some((1422.0, 645.0)),
            ),
        ];
        assert_eq!(deduplicate_verified(candidates).len(), 1);
    }
}
