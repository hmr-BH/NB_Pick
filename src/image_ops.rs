use image::{RgbImage, imageops};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PersonRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PersonCrop {
    pub x: u32,
    pub y: u32,
    pub size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseCrop {
    pub x: i32,
    pub y: i32,
    pub size: u32,
}

pub fn valid_person(person: PersonRect, width: u32, height: u32) -> bool {
    person.score.is_finite()
        && person.score >= 0.25
        && [person.x, person.y, person.w, person.h]
            .iter()
            .all(|v| v.is_finite())
        && person.w >= 6.0
        && person.h >= 12.0
        && person.x >= 0.0
        && person.y >= 0.0
        && person.x + person.w <= width as f32 + 0.01
        && person.y + person.h <= height as f32 + 0.01
        && (0.08..=2.5).contains(&(person.w / person.h))
}

pub fn person_crop_rect(frame: &RgbImage, person: PersonRect) -> PersonCrop {
    // Enlarge the head and torso while keeping a full square inside the source
    // frame. Translating the square inward prevents dark padding in the result.
    let side = person.w.max(person.h * 0.52) * 1.55;
    let cx = person.x + person.w / 2.0;
    let cy = person.y + person.h * 0.26;
    inside_square(frame, cx, cy, side)
}

pub fn pose_crop_rect(_frame: &RgbImage, person: PersonRect) -> PoseCrop {
    // Keep more body context for the pose model. A tight head crop can turn
    // clothing folds into false facial keypoints.
    let side = (person.w.max(person.h * 0.58) * 1.45).ceil().max(1.0);
    let cx = person.x + person.w / 2.0;
    let cy = person.y + person.h * 0.30;
    PoseCrop {
        x: (cx - side / 2.0).round() as i32,
        y: (cy - side / 2.0).round() as i32,
        size: side as u32,
    }
}

fn inside_square(frame: &RgbImage, cx: f32, cy: f32, desired_side: f32) -> PersonCrop {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let min_dimension = frame.width().min(frame.height()).max(1) as f32;
    let side = desired_side.ceil().clamp(1.0, min_dimension);
    let size = side.round() as u32;
    let max_x = (fw as u32).saturating_sub(size);
    let max_y = (fh as u32).saturating_sub(size);
    PersonCrop {
        x: (cx - side / 2.0).round().clamp(0.0, max_x as f32) as u32,
        y: (cy - side / 2.0).round().clamp(0.0, max_y as f32) as u32,
        size,
    }
}

pub fn crop_person(frame: &RgbImage, person: PersonRect) -> RgbImage {
    let rect = person_crop_rect(frame, person);
    let crop = imageops::crop_imm(frame, rect.x, rect.y, rect.size, rect.size).to_image();
    imageops::resize(&crop, 384, 384, imageops::FilterType::Triangle)
}

pub fn pose_crop_person(frame: &RgbImage, person: PersonRect) -> RgbImage {
    let rect = pose_crop_rect(frame, person);
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let mut canvas = RgbImage::from_pixel(rect.size, rect.size, image::Rgb([17, 24, 39]));
    let left = rect.x.max(0) as u32;
    let top = rect.y.max(0) as u32;
    let right = (rect.x as f32 + rect.size as f32).ceil().clamp(0.0, fw) as u32;
    let bottom = (rect.y as f32 + rect.size as f32).ceil().clamp(0.0, fh) as u32;
    if right > left && bottom > top {
        let crop = imageops::crop_imm(frame, left, top, right - left, bottom - top).to_image();
        imageops::replace(
            &mut canvas,
            &crop,
            (left as i32 - rect.x) as i64,
            (top as i32 - rect.y) as i64,
        );
    }
    imageops::resize(&canvas, 384, 384, imageops::FilterType::Triangle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn rejects_invalid_and_low_confidence_people() {
        assert!(!valid_person(
            PersonRect {
                x: -2.0,
                y: 0.0,
                w: 40.0,
                h: 80.0,
                score: 0.9
            },
            320,
            240
        ));
        assert!(!valid_person(
            PersonRect {
                x: 0.0,
                y: 0.0,
                w: 5.0,
                h: 40.0,
                score: 0.9
            },
            320,
            240
        ));
        assert!(!valid_person(
            PersonRect {
                x: 0.0,
                y: 0.0,
                w: 40.0,
                h: 80.0,
                score: f32::NAN
            },
            320,
            240
        ));
        assert!(!valid_person(
            PersonRect {
                x: 20.0,
                y: 20.0,
                w: 30.0,
                h: 60.0,
                score: 0.24
            },
            320,
            240
        ));
    }

    #[test]
    fn accepts_distant_and_bent_over_people() {
        assert!(valid_person(
            PersonRect {
                x: 1000.0,
                y: 300.0,
                w: 8.0,
                h: 24.0,
                score: 0.3
            },
            1920,
            1080
        ));
        assert!(valid_person(
            PersonRect {
                x: 100.0,
                y: 300.0,
                w: 180.0,
                h: 90.0,
                score: 0.7
            },
            1920,
            1080
        ));
    }

    #[test]
    fn edge_crop_is_square_and_does_not_panic() {
        let frame = RgbImage::new(320, 240);
        let person = PersonRect {
            x: 0.0,
            y: 0.0,
            w: 60.0,
            h: 150.0,
            score: 0.9,
        };
        assert_eq!(crop_person(&frame, person).dimensions(), (384, 384));
    }

    #[test]
    fn edge_crop_is_shifted_inside_frame_without_padding() {
        let frame = RgbImage::from_pixel(320, 240, Rgb([255, 0, 0]));
        let person = PersonRect {
            x: 0.0,
            y: 0.0,
            w: 60.0,
            h: 150.0,
            score: 0.9,
        };
        let rect = person_crop_rect(&frame, person);
        assert_eq!(rect.x, 0);
        assert_eq!(rect.y, 0);
        assert!(rect.x + rect.size <= frame.width());
        assert!(rect.y + rect.size <= frame.height());
        assert!(
            crop_person(&frame, person)
                .pixels()
                .all(|pixel| *pixel == Rgb([255, 0, 0]))
        );
    }

    #[test]
    fn oversized_person_crop_uses_largest_inside_square() {
        let frame = RgbImage::new(320, 240);
        let person = PersonRect {
            x: 20.0,
            y: 10.0,
            w: 280.0,
            h: 230.0,
            score: 0.9,
        };
        let rect = person_crop_rect(&frame, person);
        assert_eq!(rect.size, 240);
        assert!(rect.x + rect.size <= 320);
        assert!(rect.y + rect.size <= 240);
    }
}
