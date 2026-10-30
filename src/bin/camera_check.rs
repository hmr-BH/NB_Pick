use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};

fn main() {
    for device in nokhwa::query(ApiBackend::MediaFoundation).unwrap_or_default() {
        println!("DEVICE name={} id={}", device.human_name(), device.misc());
        if let Ok(mut camera) = nokhwa::Camera::with_backend(
            CameraIndex::String(device.misc().to_string()),
            RequestedFormat::new::<nokhwa::pixel_format::RgbAFormat>(RequestedFormatType::None),
            ApiBackend::MediaFoundation,
        ) {
            match camera.compatible_camera_formats() {
                Ok(formats) => println!("FORMATS {formats:?}"),
                Err(error) => println!("FORMATS ERR {error}"),
            }
        }
        for format in [
            RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(RequestedFormatType::Closest(
                CameraFormat::new(Resolution::new(1280, 720), FrameFormat::MJPEG, 30),
            )),
            RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
                RequestedFormatType::AbsoluteHighestResolution,
            ),
            RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(RequestedFormatType::None),
        ] {
            match nokhwa::Camera::with_backend(
                CameraIndex::String(device.misc().to_string()),
                format,
                ApiBackend::MediaFoundation,
            ) {
                Ok(mut camera) => match camera.open_stream() {
                    Ok(()) => match camera.frame() {
                        Ok(frame) => {
                            println!(
                                "OK {:?} {}x{}",
                                camera.frame_format(),
                                frame.resolution().width(),
                                frame.resolution().height()
                            );
                            if let Ok(rgb) = frame.decode_image::<nokhwa::pixel_format::RgbFormat>()
                            {
                                let _ = rgb.save("target/camera-frame.png");
                            }
                            let _ = camera.stop_stream();
                            return;
                        }
                        Err(e) => println!("READ ERR {e}"),
                    },
                    Err(e) => println!("OPEN ERR {e}"),
                },
                Err(e) => println!("NEW ERR {e}"),
            }
        }
    }
}
