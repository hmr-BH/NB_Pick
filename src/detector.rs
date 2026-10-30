use crate::image_ops::{FaceRect, valid_face};
use anyhow::{Context, Result};
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

pub struct Detector {
    session: Session,
    delay: Duration,
}

impl Detector {
    pub fn load(path: &Path) -> Result<Self> {
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
            .with_context(|| format!("加载模型失败：{}", path.display()))?;
        Ok(Self {
            session,
            delay: Duration::from_millis(80),
        })
    }

    pub fn detect(&mut self, frame: &RgbImage) -> Result<Vec<FaceRect>> {
        let started = Instant::now();
        let scale = 640.0 / frame.width().max(frame.height()) as f32;
        let w = ((frame.width() as f32 * scale).round() as u32).max(1);
        let h = ((frame.height() as f32 * scale).round() as u32).max(1);
        let resized = imageops::resize(frame, w, h, imageops::FilterType::Triangle);
        let mut canvas = RgbImage::from_pixel(640, 640, Rgb([114, 114, 114]));
        imageops::replace(&mut canvas, &resized, 0, 0);
        let mut input = Array4::<f32>::zeros((1, 3, 640, 640));
        for (x, y, pixel) in canvas.enumerate_pixels() {
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] = pixel[2 - c] as f32;
            }
        }
        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let mut faces = Vec::new();
        for &stride in &[8usize, 16, 32] {
            let cls = tensor(
                outputs
                    .get(format!("cls_{stride}"))
                    .context("模型缺少分类输出")?,
            )?;
            let obj = tensor(
                outputs
                    .get(format!("obj_{stride}"))
                    .context("模型缺少目标输出")?,
            )?;
            let bbox = tensor(
                outputs
                    .get(format!("bbox_{stride}"))
                    .context("模型缺少边框输出")?,
            )?;
            for i in 0..cls.len() {
                let score = (cls[i] * obj[i]).sqrt();
                if score < 0.80 {
                    continue;
                }
                let col = i % (640 / stride);
                let row = i / (640 / stride);
                let cx = (col as f32 + bbox[i * 4]) * stride as f32;
                let cy = (row as f32 + bbox[i * 4 + 1]) * stride as f32;
                let bw = bbox[i * 4 + 2].exp() * stride as f32;
                let bh = bbox[i * 4 + 3].exp() * stride as f32;
                let x1 = ((cx - bw / 2.0).max(0.0) / scale).min(frame.width() as f32);
                let y1 = ((cy - bh / 2.0).max(0.0) / scale).min(frame.height() as f32);
                let x2 = ((cx + bw / 2.0).min(w as f32) / scale).min(frame.width() as f32);
                let y2 = ((cy + bh / 2.0).min(h as f32) / scale).min(frame.height() as f32);
                let face = FaceRect {
                    x: x1,
                    y: y1,
                    w: x2 - x1,
                    h: y2 - y1,
                    score,
                };
                if valid_face(face, frame.width(), frame.height()) {
                    faces.push(face);
                }
            }
        }
        let mut faces = nms(faces, 0.3);
        faces.sort_by(|a, b| a.x.total_cmp(&b.x));
        faces.truncate(64);
        let elapsed = started.elapsed();
        self.delay = if elapsed.as_millis() > 350 {
            Duration::from_millis(400)
        } else if elapsed.as_millis() > 180 {
            Duration::from_millis(200)
        } else {
            Duration::from_millis(60)
        };
        std::thread::sleep(self.delay);
        Ok(faces)
    }
}

fn tensor(value: &ort::value::DynValue) -> Result<&[f32]> {
    Ok(value.try_extract_tensor::<f32>()?.1)
}

fn iou(a: FaceRect, b: FaceRect) -> f32 {
    let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
    let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
    if w <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    w * h / (a.w * a.h + b.w * b.h - w * h)
}

fn nms(mut candidates: Vec<FaceRect>, threshold: f32) -> Vec<FaceRect> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<FaceRect> = Vec::new();
    for face in candidates {
        if kept.iter().all(|other| iou(face, *other) <= threshold) {
            kept.push(face);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nms_keeps_distinct_faces_and_removes_duplicates() {
        let a = FaceRect {
            x: 10.0,
            y: 10.0,
            w: 50.0,
            h: 50.0,
            score: 0.9,
        };
        let b = FaceRect {
            x: 12.0,
            y: 12.0,
            w: 50.0,
            h: 50.0,
            score: 0.8,
        };
        let c = FaceRect {
            x: 150.0,
            y: 10.0,
            w: 50.0,
            h: 50.0,
            score: 0.85,
        };
        assert_eq!(nms(vec![a, b, c], 0.3).len(), 2);
    }
}
