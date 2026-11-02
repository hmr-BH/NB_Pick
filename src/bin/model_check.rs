use nb_pick::detector::Detector;
use nb_pick::image_ops::PersonRect;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = Detector::load(
        &PathBuf::from("assets/models/person_detection_yolox_s.onnx"),
        &PathBuf::from("assets/models/person_keypoints_rtmpose_t.onnx"),
    )?;
    let args = std::env::args().collect::<Vec<_>>();
    if let Some(path) = args.get(1) {
        let image = image::open(path)?.to_rgb8();
        if let (Some(x), Some(y), Some(w), Some(h)) =
            (args.get(2), args.get(3), args.get(4), args.get(5))
        {
            let person = PersonRect {
                x: x.parse()?,
                y: y.parse()?,
                w: w.parse()?,
                h: h.parse()?,
                score: 1.0,
            };
            println!(
                "verified person: {}",
                detector.verify_person(&image, person)?
            );
        } else {
            println!("people: {:?}", detector.detect(&image)?);
        }
    } else {
        println!(
            "empty image people: {:?}",
            detector.detect(&image::RgbImage::new(640, 480))?
        );
    }
    Ok(())
}
