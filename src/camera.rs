use anyhow::{Context, Result};
use image::RgbImage;
use nokhwa::{
    Camera,
    pixel_format::{RgbAFormat, RgbFormat},
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    },
};

#[derive(Clone, Debug)]
pub struct Device {
    pub id: String,
    pub name: String,
}

pub fn devices() -> Vec<Device> {
    nokhwa::query(ApiBackend::MediaFoundation)
        .unwrap_or_default()
        .into_iter()
        .map(|info| Device {
            id: info.misc().to_string(),
            name: info.human_name().to_string(),
        })
        .collect()
}

pub fn open(id: &str) -> Result<(Camera, String)> {
    let index = CameraIndex::String(id.to_string());
    let mut camera = Camera::with_backend(
        index,
        RequestedFormat::new::<RgbAFormat>(RequestedFormatType::None),
        ApiBackend::MediaFoundation,
    )
    .with_context(|| "无法打开摄像头，请检查系统隐私权限或设备是否被占用")?;
    if let Ok(formats) = camera.compatible_camera_formats()
        && let Some(format) = choose_format(&formats)
    {
        // Preserve distant people for tiled inference; the UI remains independent of capture size.
        if camera
            .set_camera_requset(RequestedFormat::new::<RgbAFormat>(
                RequestedFormatType::Exact(format),
            ))
            .is_err()
        {
            anyhow::bail!("摄像头不支持可用画面格式，请尝试其他设备");
        }
    }
    camera
        .open_stream()
        .context("摄像头启动失败，请尝试其他设备")?;
    let format = format!(
        "{}×{} · {}",
        camera.resolution().width(),
        camera.resolution().height(),
        camera.frame_format()
    );
    Ok((camera, format))
}

pub fn read(camera: &mut Camera) -> Result<RgbImage> {
    let frame = camera.frame().context("摄像头读取失败")?;
    frame
        .decode_image::<RgbFormat>()
        .context("摄像头画面解码失败")
}

fn choose_format(formats: &[CameraFormat]) -> Option<CameraFormat> {
    const TARGET_AREA: f32 = 1920.0 * 1080.0;
    formats.iter().copied().min_by(|left, right| {
        format_score(*left, TARGET_AREA).total_cmp(&format_score(*right, TARGET_AREA))
    })
}

fn format_score(format: CameraFormat, target_area: f32) -> f32 {
    let width = format.width() as f32;
    let height = format.height() as f32;
    let area = width * height;
    let mut score = (area / target_area).ln().abs();
    let fps = format.frame_rate() as f32;
    score += (fps / 30.0).ln().abs() * 0.35;
    if fps < 15.0 {
        score += 1.0;
    }
    score
        + match format.format() {
            FrameFormat::NV12 => 0.0,
            FrameFormat::MJPEG => 0.08,
            FrameFormat::YUYV => 0.2,
            FrameFormat::RAWRGB | FrameFormat::RAWBGR => 0.3,
            FrameFormat::GRAY => 1.0,
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_prefers_1080p_for_distant_people() {
        let formats = [
            CameraFormat::new(
                nokhwa::utils::Resolution::new(1920, 1080),
                FrameFormat::NV12,
                30,
            ),
            CameraFormat::new(
                nokhwa::utils::Resolution::new(640, 480),
                FrameFormat::YUYV,
                30,
            ),
            CameraFormat::new(
                nokhwa::utils::Resolution::new(1280, 720),
                FrameFormat::NV12,
                30,
            ),
        ];
        assert_eq!(
            choose_format(&formats).map(|format| format.resolution()),
            Some(nokhwa::utils::Resolution::new(1920, 1080))
        );
    }
}
