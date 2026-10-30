use nb_pick::detector::Detector;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = Detector::load(&PathBuf::from(
        "assets/models/face_detection_yunet_2023mar.onnx",
    ))?;
    if let Some(path) = std::env::args().nth(1) {
        let image = image::open(path)?.to_rgb8();
        println!("faces: {:?}", detector.detect(&image)?);
    } else {
        println!(
            "empty image faces: {:?}",
            detector.detect(&image::RgbImage::new(640, 480))?
        );
    }
    Ok(())
}
