extern crate noise;

use anyhow::{anyhow, Result};
use colorgrad::Gradient;
use core::f64;
use image::ImageBuffer;
use ndarray::Array2;
use noise::{Fbm, NoiseFn, Perlin};
use num::complex::{Complex, Complex32};
use num::integer::Roots;
use ocl::Buffer;
//use ocl::ProQue;
use plotters::prelude::*;
use std::f32::consts::PI;

pub fn noise2d(n: usize) -> Array2<Complex32> {
    let s = 2.0 * f64::consts::PI / (n as f64);
    let r = 10.0;
    let mut a = Array2::<Complex32>::zeros((n, n));
    let perlin: Fbm<Perlin> = Fbm::new(12);
    for i in 0..n {
        for j in 0..n {
            a[[i, j]].re = perlin.get([
                r * (i as f64 * s).cos(),
                r * (i as f64 * s).sin(),
                r * (j as f64 * s).cos(),
                r * (j as f64 * s).sin(),
            ]) as f32;
        }
    }
    return a;
}

pub fn plot<'a, I>(data: I, name: &str) -> Result<()>
where
    I: Iterator<Item = &'a f32> + ExactSizeIterator,
{
    let n = data.len();
    let l = 2f32 * PI;
    let dx = l / n as f32;
    let root = SVGBackend::new(name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..l, -2f32..2f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            data.enumerate().map(|(i, y)| (i as f32 * dx, *y)),
            &RED,
        ))?
        .label("y = x^2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

pub fn dist(a: Vec<f32>, b: Vec<f32>, dx: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        s += (a[i] - b[i]) * (a[i] - b[i]) * dx;
    }
    return s.sqrt();
}

pub fn fftfreq(n: usize, l: f32) -> Vec<f32> {
    let mut freq = vec![0f32; n];
    let s = 2.0 * PI / l;
    for i in 0..n / 2 {
        freq[i] = i as f32 * s;
        freq[n / 2 + i] = (i as f32 - (n as f32) / 2.0) * s;
    }
    return freq;
}

pub fn get_from_gpu(buffer: &Buffer<Complex<f32>>) -> Result<Array2<Complex<f32>>> {
    let n = buffer.len().sqrt();
    let mut cpu_data = Array2::<Complex<f32>>::zeros((n, n));
    buffer
        .read(cpu_data.as_slice_mut().ok_or(anyhow!("Noo"))?)
        .enq()?;
    buffer.default_queue().unwrap().finish()?;
    return Ok(cpu_data);
}

pub fn max(arr: &Array2<f32>) -> f32 {
    return arr.into_iter().cloned().reduce(f32::max).unwrap();
}

pub fn printmax(buffer: &Buffer<Complex<f32>>, name: &str) -> Result<()> {
    let m = max(&get_from_gpu(&buffer)?.mapv(Complex32::norm));
    println!("Max {} : {}", name, m);
    return Ok(());
}

pub fn image_from_array(cpu_data: &Array2<f32>) -> Result<ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
    let n = cpu_data.len_of(ndarray::Axis(0));
    let m = max(&cpu_data.mapv(f32::abs));
    let normalized = cpu_data.mapv(|x| 0.5 + x / (2.0 * m));
    let grad = colorgrad::GradientBuilder::new()
        .html_colors(&["red", "white", "blue"])
        .build::<colorgrad::CatmullRomGradient>()?;
    let imgbuf = image::ImageBuffer::from_fn(n as u32, n as u32, |i, j| {
        let v = grad.at(normalized[[i as usize, j as usize]]).to_rgba8();
        image::Rgb(v[..3].try_into().unwrap())
    });
    return Ok(imgbuf);
}

pub fn plot_array(cpu_data: &Array2<f32>, name: &str) -> Result<()> {
    let imgbuf = image_from_array(cpu_data)?;
    imgbuf.save(name)?;
    return Ok(());
}

pub fn plot_from_gpu(buffer: &Buffer<Complex32>, name: &str) -> Result<()> {
    let cpu_data = get_from_gpu(&buffer)?.mapv(|x| x.re);
    plot_array(&cpu_data, name)?;
    return Ok(());
}

pub fn new_buffer(queue: &ocl::Queue, n: usize) -> Result<Buffer<Complex32>> {
    let buffer = Buffer::<Complex32>::builder()
        .queue(queue.clone())
        .len(n * n)
        .build()?;
    return Ok(buffer);
}
