use nb_pick::camera;
use nokhwa::{
    pixel_format::RgbAFormat,
    utils::{CameraFormat, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = camera::devices();
    let device = devices.first().ok_or("no camera")?;
    let (mut camera, format) = camera::open(&device.id)?;
    if std::env::args().nth(1).as_deref() == Some("small") {
        camera.set_camera_requset(RequestedFormat::new::<RgbAFormat>(
            RequestedFormatType::Exact(CameraFormat::new(
                Resolution::new(640, 480),
                FrameFormat::NV12,
                30,
            )),
        ))?;
    }
    for _ in 0..3 {
        let _ = camera::read(&mut camera)?;
    }
    let start = Instant::now();
    let mut frames = 0usize;
    while start.elapsed().as_secs_f32() < 3.0 {
        let _ = camera::read(&mut camera)?;
        frames += 1;
    }
    println!(
        "requested={format} actual={}x{} {:?} rate={} frames={frames} fps={:.1}",
        camera.resolution().width(),
        camera.resolution().height(),
        camera.frame_format(),
        camera.frame_rate(),
        frames as f32 / start.elapsed().as_secs_f32()
    );
    let _ = camera.stop_stream();
    Ok(())
}
