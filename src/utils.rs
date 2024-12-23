use plotters::prelude::*;
use std::f32::consts::PI;

use anyhow::Result;

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
        .build_cartesian_2d(0f32..l, -1f32..1f32)?;

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
        freq[i] = i as f32 * s ;
        freq[n / 2 + i] = (i as f32 - (n as f32) / 2.0) * s;
    }
    return freq;
}
