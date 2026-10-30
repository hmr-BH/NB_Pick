use image::{Rgb, RgbImage, imageops};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

pub fn valid_face(face: FaceRect, width: u32, height: u32) -> bool {
    face.score.is_finite()
        && face.score >= 0.80
        && [face.x, face.y, face.w, face.h]
            .iter()
            .all(|v| v.is_finite())
        && face.w >= width as f32 * 0.035
        && face.h >= height as f32 * 0.035
        && face.x >= 0.0
        && face.y >= 0.0
        && face.x + face.w <= width as f32
        && face.y + face.h <= height as f32
        && (0.55..=1.8).contains(&(face.w / face.h))
}

pub fn crop_face(frame: &RgbImage, face: FaceRect) -> RgbImage {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let side = face.w.max(face.h) * 1.65;
    let side = side.ceil().max(1.0);
    let cx = face.x + face.w / 2.0;
    let cy = face.y + face.h / 2.0 - side * 0.04;
    let source = RgbImage::from_pixel(side as u32, side as u32, Rgb([21, 27, 35]));
    let mut canvas = source;
    let sx = (cx - side / 2.0).round() as i32;
    let sy = (cy - side / 2.0).round() as i32;
    let left = sx.max(0) as u32;
    let top = sy.max(0) as u32;
    let right = (sx as f32 + side).ceil().clamp(0.0, fw) as u32;
    let bottom = (sy as f32 + side).ceil().clamp(0.0, fh) as u32;
    if right > left && bottom > top {
        let crop = imageops::crop_imm(frame, left, top, right - left, bottom - top).to_image();
        imageops::replace(
            &mut canvas,
            &crop,
            (left as i32 - sx) as i64,
            (top as i32 - sy) as i64,
        );
    }
    imageops::resize(&canvas, 384, 384, imageops::FilterType::Triangle)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_small_and_out_of_bounds_faces() {
        assert!(!valid_face(
            FaceRect {
                x: -2.0,
                y: 0.0,
                w: 40.0,
                h: 40.0,
                score: 0.9
            },
            320,
            240
        ));
        assert!(!valid_face(
            FaceRect {
                x: 0.0,
                y: 0.0,
                w: 2.0,
                h: 40.0,
                score: 0.9
            },
            320,
            240
        ));
        assert!(!valid_face(
            FaceRect {
                x: 0.0,
                y: 0.0,
                w: 40.0,
                h: 40.0,
                score: f32::NAN
            },
            320,
            240
        ));
        assert!(!valid_face(
            FaceRect {
                x: 20.0,
                y: 20.0,
                w: 60.0,
                h: 60.0,
                score: 0.79
            },
            320,
            240
        ));
    }
    #[test]
    fn edge_crop_is_square_and_does_not_panic() {
        let f = RgbImage::new(320, 240);
        let face = FaceRect {
            x: 0.0,
            y: 0.0,
            w: 60.0,
            h: 60.0,
            score: 0.9,
        };
        assert_eq!(crop_face(&f, face).dimensions(), (384, 384));
    }
    #[test]
    fn edge_crop_keeps_source_pixels_in_correct_place() {
        let mut frame = RgbImage::new(100, 100);
        for pixel in frame.pixels_mut() {
            *pixel = Rgb([255, 0, 0]);
        }
        let face = FaceRect {
            x: 0.0,
            y: 0.0,
            w: 30.0,
            h: 30.0,
            score: 0.9,
        };
        let crop = crop_face(&frame, face);
        assert_eq!(crop.get_pixel(0, 0), &Rgb([21, 27, 35]));
        assert!(crop.pixels().any(|p| *p == Rgb([255, 0, 0])));
    }
}
