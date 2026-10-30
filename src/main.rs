#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;

use audio::{Audio, Sound};
use image::RgbImage;
use nb_pick::camera;
use nb_pick::camera::Device;
use nb_pick::config::{Config, choose_camera};
use nb_pick::detector::Detector;
use nb_pick::image_ops::{FaceRect, crop_face};
use rand::{rng, seq::IndexedRandom};
use slint::{
    CloseRequestResponse, ComponentHandle, Image, ModelRc, Rgb8Pixel, Rgba8Pixel,
    SharedPixelBuffer, SharedString, VecModel, Weak,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

slint::include_modules!();

struct LatestFrame {
    frame: Arc<RgbImage>,
    faces: Vec<FaceRect>,
    detected_at: Instant,
    captured_at: Instant,
}

struct UiFrame {
    generation: u64,
    frame: Arc<RgbImage>,
    format: String,
}

struct Runtime {
    latest: Mutex<Option<LatestFrame>>,
    generation: Mutex<u64>,
    stop: Mutex<bool>,
    active: AtomicBool,
    last_face_at: Mutex<Option<Instant>>,
    config: Mutex<Config>,
    devices: Mutex<Vec<Device>>,
    started: Mutex<Instant>,
    audio: Mutex<Option<Audio>>,
    loading: Mutex<bool>,
    camera_error: Mutex<Option<String>>,
    inference: Mutex<Option<Arc<RgbImage>>>,
    ui_frame: Mutex<Option<UiFrame>>,
}

static RUNTIME: OnceLock<Arc<Runtime>> = OnceLock::new();

enum Event {
    Ready(Detector),
    Fatal(String),
    Devices(Vec<Device>),
    CameraError {
        generation: u64,
        message: String,
    },
    InferenceError {
        generation: u64,
    },
    Detected {
        generation: u64,
        frame: Arc<RgbImage>,
        faces: Vec<FaceRect>,
    },
}

fn runtime() -> Arc<Runtime> {
    RUNTIME.get().expect("runtime initialized").clone()
}

fn send(app: &Weak<AppWindow>, event: Event) {
    let app = app.clone();
    let _ = slint::invoke_from_event_loop(move || process_event(&app, event));
}

fn present_latest_frame(app: &AppWindow, rt: &Runtime) {
    let Some(ui_frame) = rt.ui_frame.lock().unwrap().take() else {
        return;
    };
    if ui_frame.generation != *rt.generation.lock().unwrap() || !rt.active.load(Ordering::SeqCst) {
        return;
    }
    *rt.camera_error.lock().unwrap() = None;

    let first_frame = {
        let mut latest = rt.latest.lock().unwrap();
        if latest.is_none() {
            *latest = Some(LatestFrame {
                frame: ui_frame.frame.clone(),
                faces: Vec::new(),
                detected_at: Instant::now(),
                captured_at: Instant::now(),
            });
            true
        } else {
            if let Some(latest) = latest.as_mut() {
                latest.frame = ui_frame.frame.clone();
                latest.captured_at = Instant::now();
            }
            false
        }
    };

    if first_frame {
        app.set_page(1);
        app.set_camera_info(ui_frame.format.into());
        app.set_status("正在识别画面中的人脸".into());
        show_scan_window(app.window());
        *rt.started.lock().unwrap() = Instant::now();
        if !app.get_muted()
            && let Some(audio) = rt.audio.lock().unwrap().as_ref()
        {
            audio.play(Sound::Draw);
        }
    }
    app.set_frame(to_slint_image(ui_frame.frame.as_ref()));
}

fn process_event(app: &Weak<AppWindow>, event: Event) {
    let Some(app) = app.upgrade() else { return };
    let rt = runtime();
    match event {
        Event::Ready(detector) => {
            app.set_page(0);
            app.set_progress(0.62);
            app.set_status("模型就绪，正在连接摄像头".into());
            show_scan_window(app.window());
            *rt.camera_error.lock().unwrap() = None;
            rt.active.store(true, Ordering::SeqCst);
            *rt.last_face_at.lock().unwrap() = None;
            *rt.started.lock().unwrap() = Instant::now();
            let weak = app.as_weak();
            let detector_rt = rt.clone();
            let detector_app = weak.clone();
            thread::spawn(move || detection_loop(detector_app, detector_rt, detector));
            thread::spawn(move || capture_loop(weak, rt));
        }
        Event::Fatal(message) => {
            rt.active.store(false, Ordering::SeqCst);
            app.set_page(0);
            app.set_progress(-1.0);
            app.set_status(message.into());
        }
        Event::Devices(devices) => {
            *rt.devices.lock().unwrap() = devices.clone();
            app.set_cameras(ModelRc::new(VecModel::from(
                devices
                    .iter()
                    .map(|device| SharedString::from(&device.name))
                    .collect::<Vec<_>>(),
            )));
            let config = rt.config.lock().unwrap().clone();
            let remembered = config.camera_id.as_deref().unwrap_or("");
            if let Some(index) = choose_camera(&devices, remembered, |d| &d.id, |d| &d.name) {
                app.set_active_camera_index(index as i32);
                app.set_camera_name(devices[index].name.clone().into());
            } else {
                app.set_active_camera_index(-1);
            }
        }
        Event::CameraError {
            generation,
            message,
        } => {
            if generation != *rt.generation.lock().unwrap() {
                return;
            }
            let friendly = friendly_camera_error(&message);
            *rt.camera_error.lock().unwrap() = Some(friendly.clone());
            app.set_page(1);
            app.set_status(friendly.into());
            app.set_face_count(0);
            app.set_faces(ModelRc::new(VecModel::from(Vec::<FaceBox>::new())));
            app.set_progress(0.0);
            app.set_status("正在尝试恢复摄像头".into());
            show_scan_window(app.window());
        }
        Event::InferenceError { generation, .. } => {
            if generation == *rt.generation.lock().unwrap() && rt.active.load(Ordering::SeqCst) {
                app.set_status("识别暂时中断，正在自动恢复".into());
            }
        }
        Event::Detected {
            generation,
            frame,
            faces,
        } => {
            if generation != *rt.generation.lock().unwrap() || !rt.active.load(Ordering::SeqCst) {
                return;
            }
            let mut lock = rt.latest.lock().unwrap();
            let Some(current) = lock.as_mut() else { return };
            current.faces = faces.clone();
            current.detected_at = Instant::now();
            current.frame = frame.clone();
            drop(lock);

            if !faces.is_empty() {
                *rt.last_face_at.lock().unwrap() = Some(Instant::now());
            }
            app.set_face_count(faces.len() as i32);
            let (width, height) = logical_window_size(&app);
            app.set_faces(face_boxes(&faces, &frame, width, height));
            app.set_status(if faces.is_empty() {
                "请让同学们面向摄像头".into()
            } else {
                format!("已识别 {} 人，正在抽取", faces.len()).into()
            });
        }
    }
}

fn detection_loop(app: Weak<AppWindow>, rt: Arc<Runtime>, mut detector: Detector) {
    lower_detection_priority();
    loop {
        if *rt.stop.lock().unwrap() {
            break;
        }
        if !rt.active.load(Ordering::SeqCst) {
            *rt.inference.lock().unwrap() = None;
            *rt.ui_frame.lock().unwrap() = None;
            thread::sleep(Duration::from_millis(120));
            continue;
        }
        let Some(frame) = rt.inference.lock().unwrap().take() else {
            thread::sleep(Duration::from_millis(25));
            continue;
        };
        let generation = *rt.generation.lock().unwrap();
        match detector.detect(frame.as_ref()) {
            Ok(faces) => {
                if rt.active.load(Ordering::SeqCst) {
                    send(
                        &app,
                        Event::Detected {
                            generation,
                            frame,
                            faces,
                        },
                    );
                }
            }
            Err(_) => {
                if rt.active.load(Ordering::SeqCst) {
                    send(&app, Event::InferenceError { generation });
                }
            }
        }
    }
}

#[cfg(windows)]
fn lower_detection_priority() {
    use windows::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
    };
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

#[cfg(not(windows))]
fn lower_detection_priority() {}

fn capture_loop(app: Weak<AppWindow>, rt: Arc<Runtime>) {
    loop {
        if *rt.stop.lock().unwrap() {
            break;
        }
        if !rt.active.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(120));
            continue;
        }

        let devices = camera::devices();
        send(&app, Event::Devices(devices.clone()));
        if devices.is_empty() {
            send(
                &app,
                Event::CameraError {
                    generation: *rt.generation.lock().unwrap(),
                    message: "没有找到可用的摄像头，请连接设备或检查系统隐私权限。".into(),
                },
            );
            thread::sleep(Duration::from_secs(2));
            continue;
        }

        let config = rt.config.lock().unwrap().clone();
        let remembered = config.camera_id.as_deref().unwrap_or("");
        let chosen = choose_camera(&devices, remembered, |d| &d.id, |d| &d.name).unwrap_or(0);
        let device = devices[chosen].clone();
        if let Ok(mut saved) = rt.config.lock()
            && saved.camera_id.as_deref() != Some(&device.id)
        {
            saved.camera_id = Some(device.id.clone());
            let _ = saved.save();
        }

        let generation = *rt.generation.lock().unwrap() + 1;
        *rt.generation.lock().unwrap() = generation;
        send(&app, Event::Devices(devices.clone()));

        match camera::open(&device.id) {
            Ok((mut cam, format)) => {
                let mut consecutive_errors = 0;
                loop {
                    if *rt.stop.lock().unwrap()
                        || !rt.active.load(Ordering::SeqCst)
                        || generation != *rt.generation.lock().unwrap()
                    {
                        break;
                    }
                    match camera::read(&mut cam) {
                        Ok(frame) => {
                            consecutive_errors = 0;
                            if !rt.active.load(Ordering::SeqCst)
                                || generation != *rt.generation.lock().unwrap()
                            {
                                break;
                            }
                            let frame = Arc::new(frame);
                            *rt.inference.lock().unwrap() = Some(frame.clone());
                            *rt.ui_frame.lock().unwrap() = Some(UiFrame {
                                generation,
                                frame,
                                format: format.clone(),
                            });
                        }
                        Err(error) => {
                            consecutive_errors += 1;
                            if consecutive_errors >= 3 {
                                send(
                                    &app,
                                    Event::CameraError {
                                        generation,
                                        message: format!("{error:#}"),
                                    },
                                );
                                break;
                            }
                            thread::sleep(Duration::from_millis(100));
                        }
                    }
                }
                if cam.stop_stream().is_err() {
                    // Nokhwa's Camera::drop unwraps stop_stream; leaking a wedged
                    // device is safer than aborting the whole classroom UI.
                    std::mem::forget(cam);
                }
            }
            Err(error) => send(
                &app,
                Event::CameraError {
                    generation,
                    message: format!("{error:#}"),
                },
            ),
        }
        thread::sleep(Duration::from_millis(800));
    }
}

fn logical_window_size(app: &AppWindow) -> (f32, f32) {
    let size = app.window().size();
    let scale = app.window().scale_factor().max(0.5);
    (size.width as f32 / scale, size.height as f32 / scale)
}

fn friendly_camera_error(detail: &str) -> String {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("no device") {
        "未找到摄像头。请连接设备，或检查系统的摄像头隐私权限。".into()
    } else if lower.contains("busy")
        || lower.contains("occupied")
        || lower.contains("占用")
        || lower.contains("access")
    {
        "摄像头可能正被其他程序占用。关闭占用程序后重试，或切换其他设备。".into()
    } else {
        "摄像头连接中断。请检查连接后重试，或切换其他设备。".into()
    }
}

fn no_face_timed_out(scan_started: Instant, last_face_at: Option<Instant>, now: Instant) -> bool {
    now.duration_since(last_face_at.unwrap_or(scan_started)) >= Duration::from_secs(5)
}

fn face_boxes(faces: &[FaceRect], frame: &RgbImage, width: f32, height: f32) -> ModelRc<FaceBox> {
    ModelRc::new(VecModel::from(map_face_boxes(faces, frame, width, height)))
}

fn map_face_boxes(faces: &[FaceRect], frame: &RgbImage, width: f32, height: f32) -> Vec<FaceBox> {
    let scale_x = width / frame.width() as f32;
    let scale_y = height / frame.height() as f32;
    faces
        .iter()
        .map(|face| FaceBox {
            x: face.x * scale_x,
            y: face.y * scale_y,
            w: face.w * scale_x,
            h: face.h * scale_y,
        })
        .collect()
}

fn to_slint_image(frame: &RgbImage) -> Image {
    Image::from_rgb8(SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(
        frame.as_raw(),
        frame.width(),
        frame.height(),
    ))
}

fn to_slint_circle_image(frame: &RgbImage) -> Image {
    let width = frame.width();
    let height = frame.height();
    let radius = width.min(height) as f32 / 2.0;
    let center = (width as f32 / 2.0, height as f32 / 2.0);
    let pixels = frame
        .pixels()
        .enumerate()
        .flat_map(|(index, pixel)| {
            let x = (index as u32 % width) as f32 + 0.5;
            let y = (index as u32 / width) as f32 + 0.5;
            let distance = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
            let alpha = ((radius - distance).clamp(0.0, 1.2) / 1.2 * 255.0) as u8;
            [pixel[0], pixel[1], pixel[2], alpha]
        })
        .collect::<Vec<_>>();
    Image::from_rgba8(SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
        &pixels, width, height,
    ))
}

fn model_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.parent()
                .map(|parent| parent.join("assets/models/face_detection_yunet_2023mar.onnx"))
        })
        .filter(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("assets/models/face_detection_yunet_2023mar.onnx"))
}

fn load_model(app: Weak<AppWindow>) {
    let rt = runtime();
    if *rt.loading.lock().unwrap() {
        return;
    }
    *rt.loading.lock().unwrap() = true;
    thread::spawn(move || {
        let result = Detector::load(&model_path());
        *runtime().loading.lock().unwrap() = false;
        match result {
            Ok(detector) => send(&app, Event::Ready(detector)),
            Err(error) => send(
                &app,
                Event::Fatal(format!(
                    "离线模型加载失败：{error:#}\n请确认 assets/models 文件夹完整。"
                )),
            ),
        }
    });
}

fn shutdown() {
    let rt = runtime();
    rt.active.store(false, Ordering::SeqCst);
    *rt.stop.lock().unwrap() = true;
    *rt.inference.lock().unwrap() = None;
    *rt.ui_frame.lock().unwrap() = None;
    if let Some(audio) = rt.audio.lock().unwrap().as_ref() {
        audio.stop();
    }
}

fn show_no_face_failure(app: &AppWindow, rt: &Runtime) {
    rt.active.store(false, Ordering::SeqCst);
    *rt.inference.lock().unwrap() = None;
    *rt.ui_frame.lock().unwrap() = None;
    *rt.generation.lock().unwrap() += 1;
    if let Some(audio) = rt.audio.lock().unwrap().as_ref() {
        audio.stop();
    }
    app.set_camera_menu(false);
    app.set_settings(false);
    app.set_page(3);
    resize_result_window(app.window());
    app.window().request_redraw();
}

fn main() -> Result<(), slint::PlatformError> {
    let config = Config::load();
    let rt = Arc::new(Runtime {
        latest: Mutex::new(None),
        generation: Mutex::new(0),
        stop: Mutex::new(false),
        active: AtomicBool::new(false),
        last_face_at: Mutex::new(None),
        config: Mutex::new(config.clone()),
        devices: Mutex::new(Vec::new()),
        started: Mutex::new(Instant::now()),
        audio: Mutex::new(None),
        loading: Mutex::new(false),
        camera_error: Mutex::new(None),
        inference: Mutex::new(None),
        ui_frame: Mutex::new(None),
    });
    let _ = RUNTIME.set(rt);

    let app = AppWindow::new()?;
    app.set_scan_seconds(config.scan_seconds);
    app.set_muted(config.muted);
    app.set_page(0);
    app.set_progress(0.12);
    load_model(app.as_weak());
    thread::spawn(|| {
        let audio = Audio::new();
        *runtime().audio.lock().unwrap() = audio;
    });

    app.on_retry({
        let weak = app.as_weak();
        move || {
            if let Some(app) = weak.upgrade() {
                app.set_progress(0.12);
                app.set_status("正在重新加载离线模型".into());
                load_model(app.as_weak());
            }
        }
    });
    app.on_toggle_cameras({
        let weak = app.as_weak();
        move || {
            if let Some(app) = weak.upgrade() {
                app.set_settings(false);
                app.set_camera_menu(!app.get_camera_menu());
            }
        }
    });
    app.on_toggle_settings({
        let weak = app.as_weak();
        move || {
            if let Some(app) = weak.upgrade() {
                app.set_camera_menu(false);
                app.set_settings(!app.get_settings());
            }
        }
    });
    app.on_close_menus({
        let weak = app.as_weak();
        move || {
            if let Some(app) = weak.upgrade() {
                app.set_camera_menu(false);
                app.set_settings(false);
            }
        }
    });
    app.on_select_camera({
        let weak = app.as_weak();
        move |index| {
            let rt = runtime();
            let device = rt.devices.lock().unwrap().get(index as usize).cloned();
            let Some(device) = device else { return };
            let mut config = rt.config.lock().unwrap().clone();
            config.camera_id = Some(device.id);
            let _ = config.save();
            *rt.config.lock().unwrap() = config;
            if let Some(app) = weak.upgrade() {
                app.set_camera_menu(false);
                app.set_camera_name(device.name.into());
                app.set_active_camera_index(index);
                app.set_status("正在切换摄像头".into());
            }
        }
    });
    app.on_duration_changed({
        let weak = app.as_weak();
        move |seconds| {
            let rt = runtime();
            let mut config = rt.config.lock().unwrap().clone();
            config.scan_seconds = seconds.clamp(2.0, 6.0);
            let _ = config.save();
            *rt.config.lock().unwrap() = config.clone();
            if let Some(app) = weak.upgrade() {
                app.set_scan_seconds(config.scan_seconds);
            }
        }
    });
    app.on_mute_changed({
        let weak = app.as_weak();
        move |muted| {
            let rt = runtime();
            let mut config = rt.config.lock().unwrap().clone();
            config.muted = muted;
            let _ = config.save();
            *rt.config.lock().unwrap() = config;
            if muted && let Some(audio) = rt.audio.lock().unwrap().as_ref() {
                audio.stop();
            }
            if let Some(app) = weak.upgrade() {
                app.set_muted(muted);
            }
        }
    });
    app.on_quit(|| {
        shutdown();
        let _ = slint::quit_event_loop();
    });
    app.window().on_close_requested(|| {
        shutdown();
        let _ = slint::quit_event_loop();
        CloseRequestResponse::HideWindow
    });
    app.on_redraw({
        let weak = app.as_weak();
        move || {
            let Some(app) = weak.upgrade() else { return };
            let rt = runtime();
            *rt.latest.lock().unwrap() = None;
            *rt.inference.lock().unwrap() = None;
            *rt.ui_frame.lock().unwrap() = None;
            *rt.camera_error.lock().unwrap() = None;
            *rt.generation.lock().unwrap() += 1;
            *rt.started.lock().unwrap() = Instant::now();
            rt.active.store(true, Ordering::SeqCst);
            *rt.last_face_at.lock().unwrap() = None;
            app.set_page(1);
            app.set_progress(0.0);
            app.set_status("正在唤醒摄像头".into());
            show_scan_window(app.window());
            if let Some(audio) = rt.audio.lock().unwrap().as_ref() {
                audio.stop();
            }
        }
    });

    let weak = app.as_weak();
    let timer = slint::Timer::default();
    let mut box_tick = 0u8;
    timer.start(
        slint::TimerMode::Repeated,
        Duration::from_millis(33),
        move || {
            let Some(app) = weak.upgrade() else { return };
            let rt = runtime();
            present_latest_frame(&app, &rt);
            if app.get_page() != 1 {
                return;
            }
            let elapsed = rt.started.lock().unwrap().elapsed().as_secs_f32();
            let duration = rt.config.lock().unwrap().scan_seconds;
            app.set_progress((elapsed / duration).min(1.0));

            let scan_started = *rt.started.lock().unwrap();
            if no_face_timed_out(
                scan_started,
                *rt.last_face_at.lock().unwrap(),
                Instant::now(),
            ) {
                let detail = if rt.camera_error.lock().unwrap().is_some() {
                    "摄像头没有提供可用画面\n请检查连接或切换设备"
                } else {
                    "5 秒内没有检测到人脸\n请调整摄像头后重试"
                };
                app.set_failure_detail(detail.into());
                show_no_face_failure(&app, &rt);
                return;
            }

            box_tick = box_tick.wrapping_add(1);
            if box_tick.is_multiple_of(3) {
                // Slint uses logical coordinates; recomputing also keeps boxes correct after a resize.
                let (width, height) = logical_window_size(&app);
                if let Some(latest) = rt.latest.lock().unwrap().as_ref() {
                    app.set_faces(face_boxes(&latest.faces, &latest.frame, width, height));
                }
            }

            if elapsed < duration || rt.camera_error.lock().unwrap().is_some() {
                return;
            }
            let selection = rt.latest.lock().unwrap().as_ref().and_then(|latest| {
                if latest.faces.is_empty()
                    || latest.detected_at.elapsed() > Duration::from_secs(2)
                    || latest.captured_at.elapsed() > Duration::from_secs(2)
                {
                    None
                } else {
                    latest
                        .faces
                        .choose(&mut rng())
                        .map(|face| (crop_face(&latest.frame, *face), latest.faces.len()))
                }
            });
            let Some((portrait, count)) = selection else {
                app.set_status("正在等待稳定的人脸画面".into());
                return;
            };

            if let Ok(audio) = rt.audio.lock()
                && let Some(audio) = audio.as_ref()
            {
                audio.stop();
                if !app.get_muted() {
                    audio.play(Sound::Result);
                }
            }
            rt.active.store(false, Ordering::SeqCst);
            *rt.inference.lock().unwrap() = None;
            *rt.ui_frame.lock().unwrap() = None;
            *rt.generation.lock().unwrap() += 1;
            app.set_drawn_count(count as i32);
            app.set_portrait(to_slint_circle_image(&portrait));
            resize_result_window(app.window());
            app.set_page(2);
            app.window().request_redraw();
        },
    );
    app.run()
}

#[cfg(windows)]
fn show_scan_window(window: &slint::Window) {
    use raw_window_handle::HasWindowHandle;
    use windows::Win32::{
        Foundation::HWND,
        Graphics::Gdi::{
            GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow,
        },
        UI::WindowsAndMessaging::{HWND_TOP, SWP_SHOWWINDOW, SetForegroundWindow, SetWindowPos},
    };
    let handle = window.window_handle();
    let Ok(handle) = handle.window_handle() else {
        return;
    };
    let raw_window_handle::RawWindowHandle::Win32(handle) = handle.as_raw() else {
        return;
    };
    let hwnd = HWND(handle.hwnd.get() as _);
    let mut monitor = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    if unsafe {
        GetMonitorInfoW(
            MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST),
            &mut monitor,
        )
    }
    .as_bool()
    {
        let rect = monitor.rcMonitor;
        unsafe {
            let _ = SetWindowPos(
                hwnd,
                Some(HWND_TOP),
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
                SWP_SHOWWINDOW,
            );
            let _ = SetForegroundWindow(hwnd);
        }
    }
}

#[cfg(not(windows))]
fn show_scan_window(window: &slint::Window) {
    window.set_size(slint::LogicalSize::new(1280.0, 720.0));
}

#[cfg(windows)]
fn resize_result_window(window: &slint::Window) {
    use raw_window_handle::HasWindowHandle;
    use windows::Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Dwm::{DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND, DwmSetWindowAttribute},
            Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow},
        },
        UI::WindowsAndMessaging::SetForegroundWindow,
    };
    let handle = window.window_handle();
    let Ok(handle) = handle.window_handle() else {
        return;
    };
    let raw_window_handle::RawWindowHandle::Win32(handle) = handle.as_raw() else {
        return;
    };
    let hwnd_value = handle.hwnd.get();
    let hwnd = HWND(hwnd_value as _);
    let mut monitor = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    if !unsafe {
        GetMonitorInfoW(
            MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST),
            &mut monitor,
        )
    }
    .as_bool()
    {
        return;
    }

    let RECT {
        left,
        top,
        right,
        bottom,
    } = monitor.rcWork;
    let width = ((right - left) / 5).max(280);
    let height = ((bottom - top) / 3).max(330);
    let target_x = left + (right - left - width) / 2;
    let target_y = top + (bottom - top - height) / 2;

    let scale = window.scale_factor().max(0.5);
    window.set_size(slint::LogicalSize::new(
        width as f32 / scale,
        height as f32 / scale,
    ));
    window.set_position(slint::LogicalPosition::new(
        target_x as f32 / scale,
        target_y as f32 / scale,
    ));
    unsafe {
        let _ = SetForegroundWindow(hwnd);
    }
    let preference = DWMWCP_ROUND.0;
    unsafe {
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            (&preference as *const i32).cast(),
            size_of::<i32>() as u32,
        );
    }
}

#[cfg(not(windows))]
fn resize_result_window(window: &slint::Window) {
    window.set_size(slint::LogicalSize::new(360.0, 420.0));
}

#[cfg(test)]
mod ui_snapshots {
    use super::*;

    fn save_snapshot(app: &AppWindow, name: &str) {
        let snapshot = app.window().take_snapshot().expect("snapshot");
        let bytes: Vec<u8> = snapshot
            .as_slice()
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        image::save_buffer(
            PathBuf::from("target").join(name),
            &bytes,
            snapshot.width(),
            snapshot.height(),
            image::ColorType::Rgba8,
        )
        .expect("write snapshot");
    }

    #[test]
    fn render_core_pages_at_target_sizes() {
        slint::platform::set_platform(Box::new(i_slint_backend_testing::TestingBackend::new(
            i_slint_backend_testing::TestingBackendOptions {
                mock_time: true,
                threading: false,
                renderer_name: Some("software".into()),
            },
        )))
        .expect("testing platform");
        let app = AppWindow::new().expect("window");
        app.window()
            .set_size(slint::LogicalSize::new(1000.0, 650.0));
        app.set_page(0);
        app.set_progress(0.58);
        app.set_status("模型就绪，正在连接摄像头".into());
        app.show().expect("show");
        save_snapshot(&app, "ui-loading.png");

        let frame = RgbImage::from_pixel(1280, 720, image::Rgb([190, 80, 70]));
        app.set_page(1);
        app.set_frame(to_slint_image(&frame));
        app.set_status("已识别 12 人，正在抽取".into());
        app.set_face_count(12);
        app.set_camera_name("教室摄像头".into());
        app.set_faces(face_boxes(
            &[
                FaceRect {
                    x: 210.0,
                    y: 120.0,
                    w: 180.0,
                    h: 220.0,
                    score: 0.9,
                },
                FaceRect {
                    x: 780.0,
                    y: 180.0,
                    w: 150.0,
                    h: 190.0,
                    score: 0.9,
                },
            ],
            &frame,
            1000.0,
            650.0,
        ));
        save_snapshot(&app, "ui-scan.png");

        let portrait = RgbImage::from_fn(420, 420, |x, y| {
            image::Rgb([
                (80 + x / 3) as u8,
                (95 + y / 4) as u8,
                (70 + (x + y) / 8) as u8,
            ])
        });
        app.window().set_size(slint::LogicalSize::new(320.0, 360.0));
        app.set_page(2);
        app.set_drawn_count(12);
        app.set_portrait(to_slint_circle_image(&portrait));
        save_snapshot(&app, "ui-result.png");

        app.set_cameras(ModelRc::new(VecModel::from(vec![
            SharedString::from("HD Webcam"),
            SharedString::from("USB Classroom Camera"),
        ])));
        app.set_active_camera_index(0);
        app.set_camera_menu(true);
        save_snapshot(&app, "ui-cameras.png");
        app.set_camera_menu(false);
        app.set_settings(true);
        save_snapshot(&app, "ui-settings.png");
        app.set_settings(false);

        app.set_page(3);
        save_snapshot(&app, "ui-failure.png");
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn face_boxes_follow_viewport_scaling() {
        let frame = RgbImage::new(1280, 720);
        let faces = [FaceRect {
            x: 640.0,
            y: 360.0,
            w: 128.0,
            h: 72.0,
            score: 0.9,
        }];
        let mapped = map_face_boxes(&faces, &frame, 1000.0, 650.0);
        assert!((mapped[0].x - 500.0).abs() < 0.01);
        assert!((mapped[0].y - 325.0).abs() < 0.01);
        assert!((mapped[0].w - 100.0).abs() < 0.01);
        assert!((mapped[0].h - 65.0).abs() < 0.01);
    }

    #[test]
    fn five_second_no_face_timeout_tracks_last_face() {
        let start = Instant::now();
        assert!(!no_face_timed_out(
            start,
            None,
            start + Duration::from_millis(4999)
        ));
        assert!(no_face_timed_out(
            start,
            None,
            start + Duration::from_secs(5)
        ));
        assert!(!no_face_timed_out(
            start,
            Some(start + Duration::from_secs(1)),
            start + Duration::from_secs(5)
        ));
        assert!(no_face_timed_out(
            start,
            Some(start + Duration::from_secs(1)),
            start + Duration::from_secs(6)
        ));
    }
}
