use rodio::{OutputStreamBuilder, Sink, buffer::SamplesBuffer};
use std::{
    f32::consts::TAU,
    sync::{Arc, Mutex},
};

const SAMPLE_RATE: u32 = 48_000;

#[derive(Clone)]
pub struct Audio {
    sink: Arc<Mutex<Option<Sink>>>,
    stream: Arc<rodio::OutputStream>,
}

impl Audio {
    pub fn new() -> Option<Self> {
        let stream = OutputStreamBuilder::open_default_stream().ok()?;
        Some(Self {
            sink: Arc::new(Mutex::new(None)),
            stream: Arc::new(stream),
        })
    }

    pub fn play(&self, sound: Sound) {
        self.stop();
        let samples = match sound {
            Sound::Draw => draw_sweep(),
            Sound::Result => result_chime(),
        };
        let sink = Sink::connect_new(self.stream.mixer());
        sink.set_volume(0.72);
        sink.append(SamplesBuffer::new(1, SAMPLE_RATE, samples));
        if let Ok(mut current) = self.sink.lock() {
            *current = Some(sink);
        }
    }

    pub fn stop(&self) {
        if let Ok(mut current) = self.sink.lock()
            && let Some(sink) = current.take()
        {
            sink.stop();
        }
    }
}

#[derive(Clone, Copy)]
pub enum Sound {
    Draw,
    Result,
}

fn draw_sweep() -> Vec<f32> {
    let seconds = 0.42;
    let count = (SAMPLE_RATE as f32 * seconds) as usize;
    let mut samples = Vec::with_capacity(count);
    let mut phase = 0.0f32;
    let mut noise_state = 0x8f3a_62d1u32;
    let mut filtered_noise = 0.0f32;
    for i in 0..count {
        let t = i as f32 / SAMPLE_RATE as f32;
        let progress = t / seconds;
        let frequency = 280.0 + 620.0 * progress.powf(1.7);
        phase = (phase + TAU * frequency / SAMPLE_RATE as f32) % TAU;
        noise_state ^= noise_state << 13;
        noise_state ^= noise_state >> 17;
        noise_state ^= noise_state << 5;
        let noise = (noise_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        filtered_noise = filtered_noise * 0.82 + noise * 0.18;
        let attack = (t / 0.035).min(1.0);
        let release = ((seconds - t) / 0.17).clamp(0.0, 1.0);
        let envelope = attack * release * release;
        let tone = phase.sin() * 0.14 + (phase * 2.0).sin() * 0.035;
        samples.push((tone + filtered_noise * 0.035) * envelope);
    }
    samples
}

fn result_chime() -> Vec<f32> {
    let seconds = 1.15;
    let count = (SAMPLE_RATE as f32 * seconds) as usize;
    let mut samples = vec![0.0f32; count];
    let notes = [
        (0.00f32, 523.25f32, 0.14f32),
        (0.13f32, 659.25f32, 0.13f32),
        (0.25f32, 783.99f32, 0.16f32),
    ];
    for (start, frequency, gain) in notes {
        let first = (start * SAMPLE_RATE as f32) as usize;
        for (i, sample) in samples.iter_mut().enumerate().skip(first) {
            let t = (i - first) as f32 / SAMPLE_RATE as f32;
            let attack = (t / 0.012).min(1.0);
            let envelope = attack * (-3.8 * t).exp();
            let fundamental = (TAU * frequency * t).sin();
            let overtone = (TAU * frequency * 2.0 * t).sin() * 0.18;
            let shimmer = (TAU * frequency * 3.01 * t).sin() * 0.055;
            *sample += (fundamental + overtone + shimmer) * envelope * gain;
        }
    }
    let delay = (0.105 * SAMPLE_RATE as f32) as usize;
    for i in delay..count {
        samples[i] += samples[i - delay] * 0.16;
    }
    for sample in &mut samples {
        *sample = sample.clamp(-0.28, 0.28);
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_cues_are_finite_short_and_audible() {
        for samples in [draw_sweep(), result_chime()] {
            assert!(!samples.is_empty());
            assert!(samples.len() < SAMPLE_RATE as usize * 2);
            assert!(samples.iter().all(|sample| sample.is_finite()));
            assert!(samples.iter().any(|sample| sample.abs() > 0.02));
            assert!(samples.iter().all(|sample| sample.abs() <= 0.28));
        }
    }
}
