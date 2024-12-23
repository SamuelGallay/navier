extern crate ndarray;
extern crate ocl;
extern crate ocl_vkfft;
extern crate rand;

pub mod utils;

use anyhow::{anyhow, Result};
use ndarray::Array2;
use num::complex::Complex;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl::{Buffer, ProQue};
use ocl_vkfft::{
    VkFFTApplication, VkFFTConfiguration, VkFFTLaunchParams, VkFFTResult_VKFFT_SUCCESS,
};
use std::f32::consts::PI;
use std::time::Instant;
use utils::plot;
use ndarray::s;

//use rand::Rng;

const N: usize = usize::pow(2, 10);
const L: f32 = 2.0 * PI;

fn trivial() -> Result<()> {
    let src = r#"
        __kernel void add(__global float2* buffer, float scalar) {
            buffer[get_global_id(0)].x += scalar;
        }
        __kernel void diff(__global float2* buffer, int N, float scalar) {
            int i = get_global_id(0);
            float x = buffer[i].x;
            float y = buffer[i].y;
            float freq = (float)i - (float)N * (2*i >= N);
            buffer[i].x = -y * scalar * freq;
            buffer[i].y =  x * scalar * freq;
        }
        // diff in first coord, scalar is 2*pi/L
        __kernel void diff_x(__global float2* buffer, int N, float scalar) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            float x = buffer[i*N +j].x;
            float y = buffer[i*N +j].y;
            float freq = scalar * ((float)i - (float)N * (2*i >= N));
            buffer[i*N +j].x = -y * freq;
            buffer[i*N +j].y =  x * freq;
        }
    "#;
    let dx = L / N as f32;

    //println!("freq : {:?}", utils::fftfreq(N, L));

    //let mut rng = rand::thread_rng();
    let pro_que = ProQue::builder().src(src).build()?;

    let mut init_data = Array2::<Complex<f32>>::zeros((N, N)); // ![0f32; 2 * N];
    for i in 0..N {
        for j in 0..N {
            init_data[[i, j]].re = f32::cos(i as f32 * dx) + f32::sin(j as f32 * dx);
        }
    }
    plot(init_data.slice(s![.., 0]).iter().map(|x| &x.re), "plot/in.svg")?;

    let fft_buffer = Buffer::<Complex<f32>>::builder()
        .queue(pro_que.queue().clone())
        .len(N*N)
        .copy_host_slice(init_data.as_slice().ok_or(anyhow!("Oh no!"))?)
        .build()?;

    let config = VkFFTConfiguration {
        FFTdim: 2,
        size: [N as u64, N as u64, 0, 0],
        numberBatches: 1,
        device: &mut pro_que.device().as_ptr(),
        context: &mut pro_que.context().as_ptr(),
        bufferSize: &mut (8 * (N*N) as u64), // 8 = sizeof(complex<f32>)
        buffer: &mut fft_buffer.as_ptr(),
        normalize: 1,
        ..Default::default()
    };
    //println!("{:?}", config);

    let mut app = VkFFTApplication {
        ..Default::default()
    };

    let res = unsafe { ocl_vkfft::initializeVkFFT(&mut app, config) };
    assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);
    println!("Initialization complete.");

    let mut launch_params = VkFFTLaunchParams {
        commandQueue: &mut pro_que.queue().as_ptr(),
        ..Default::default()
    };
    let instant = Instant::now();
    let res = unsafe { ocl_vkfft::VkFFTAppend(&mut app, -1, &mut launch_params) };
    assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);
    pro_que.queue().finish()?;
    println!("Time FFT : {:?}", instant.elapsed());

    //let instant = Instant::now();
    //let mut back_data = vec![0.0f32; fft_buffer.len()];
    //fft_buffer.read(&mut back_data).enq()?;
    //pro_que.queue().finish()?;
    //println!("Time Copy back : {:?}", instant.elapsed());
    //println!("Distance l^2 : {:?}", dist(init_data, back_data));
    //println!("{:?}", back_data);
    //
    //let s= &fft_buffer;

    unsafe {
        let kernel = pro_que
            .kernel_builder("diff_x")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&fft_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?;
        kernel.enq()?;
        let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch_params);
        assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);
    }
    pro_que.queue().finish()?;

    let mut back_data2 = Array2::<Complex<f32>>::zeros((N, N));
    fft_buffer
        .read(back_data2.as_slice_mut().ok_or(anyhow!("Noo"))?)
        .enq()?;
    pro_que.queue().finish()?;
    //println!("{:?}", back_data2);
    plot(back_data2.slice(s![.., N/4]).iter().map(|x| &x.re), "plot/out.svg")?;
    //println!("Diff : {}", (back_data2 - init_data).mapv(|x| (x*x).re).sum());


    unsafe {
        ocl_vkfft::deleteVkFFT(&mut app);
    }

    println!("End trivial.");
    Ok(())
}

fn main() {
    ocl_vkfft::say_hello();
    match trivial() {
        Ok(_) => println!("Program exited successfully."),
        Err(e) => println!("Not working : {e:?}"),
    }
}
