extern crate image;
extern crate ndarray;
extern crate ocl;
extern crate ocl_vkfft;
extern crate rand;

pub mod utils;

use anyhow::{anyhow, Result};
use core::time;
use image::GrayImage;
use ndarray::{s, Array2};
use num::complex::{Complex, Complex32};
use num::integer::Roots;
use ocl::{ocl_core::ClDeviceIdPtr, Buffer, ProQue};
use ocl_vkfft::{
    VkFFTApplication, VkFFTConfiguration, VkFFTLaunchParams, VkFFTResult_VKFFT_SUCCESS,
};
use rand::Rng;
use std::f32::consts::PI;
use std::thread::sleep;
use std::time::Instant;
use utils::{get_from_gpu, plot, printmax};

//use rand::Rng;
const SRC: &str = r#"
__kernel void add(__global float2* buffer, float scalar) {
    buffer[get_global_id(0)].x += scalar;
}
        
// diff in first coord, scalar is 2*pi/L
__kernel void mdiff_x(__global float2* buffer_in, __global float2* buffer_out, int N, float scalar) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = buffer_in[i*N +j].x;
    float y = buffer_in[i*N +j].y;
    float freq = scalar * ((float)i - (float)N * (2*i >= N));
    buffer_out[i*N +j].x =  y * freq;
    buffer_out[i*N +j].y = -x * freq;
}
__kernel void diff_y(__global float2* buffer_in, __global float2* buffer_out, int N, float scalar) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x = buffer_in[i*N +j].x;
    float y = buffer_in[i*N +j].y;
    float freq = scalar * ((float)j - (float)N * (2*j >= N));
    buffer_out[i*N +j].x = -y * freq;
    buffer_out[i*N +j].y =  x * freq;
}
__kernel void inv_mlap(__global float2* buffer_in, __global float2* buffer_out, int N, float scalar) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float re = buffer_in[i*N +j].x;
    float im = buffer_in[i*N +j].y;
    float freqi = scalar * ((float)i - (float)N * (2*i >= N));
    float freqj = scalar * ((float)j - (float)N * (2*j >= N));
    float s = freqi*freqi + freqj*freqj + ((i==0) && (j==0));
    buffer_out[i*N +j].x = re / s;
    buffer_out[i*N +j].y = im / s;
}
__kernel void advection(__global float2* w_in, __global float2* w_out, __global float2* ux, __global float2* uy, int N, float L, float dt) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float re = w_in[i*N +j].x;
    float im = w_in[i*N +j].y;
    
    float ci = (float)i - dt*ux[i*N+j].x*(float)N/L;  
    float cj = (float)j - dt*uy[i*N+j].x*(float)N/L;
    int ei = (int) floor(ci);
    int ej = (int) floor(cj);
    float di = ci - (float)ei;
    float dj = cj - (float)ej;

    float s = 0;
    s += (1-di)*(1-dj) * w_in[( ei    % N)*N +( ej    % N)].x;
    s += (1-di)*   dj  * w_in[( ei    % N)*N +((ej+1) % N)].x;
    s +=    di *(1-dj) * w_in[((ei+1) % N)*N +( ej    % N)].x;
    s +=    di *   dj  * w_in[((ei+1) % N)*N +((ej+1) % N)].x;
    
    w_out[i*N +j].x = s;
    w_out[i*N +j].y = 0;
}
"#;

const N: usize = usize::pow(2, 10);
const L: f32 = 2.0 * PI;

fn new_buffer(pro_que: &ProQue) -> Result<Buffer<Complex<f32>>> {
    let buffer = Buffer::<Complex<f32>>::builder()
        .queue(pro_que.queue().clone())
        .len(N * N)
        .build()?;
    return Ok(buffer);
}

pub fn plot_from_gpu(buffer: &Buffer<Complex<f32>>, name: &str) -> Result<()> {
    let cpu_data = get_from_gpu(&buffer)?;
    let u8_vec = cpu_data
        .map(|x| (128.0 * (x.re + 1.0)) as u8)
        .into_raw_vec();
    let a = GrayImage::from_raw(N as u32, N as u32, u8_vec).unwrap();
    a.save(name)?;
    return Ok(());
}

fn trivial() -> Result<()> {
    let dx = L / N as f32;
    let dt = 3f32;
    let niter = 10;
    //println!("freq : {:?}", utils::fftfreq(N, L));
    let mut rng = rand::thread_rng();
    let pro_que = ProQue::builder().src(SRC).build()?;

    let mut init_data = Array2::<Complex<f32>>::zeros((N, N)); // ![0f32; 2 * N];
    for i in 0..N {
        for j in 0..N {
            init_data[[i, j]].re =
                f32::cos(i as f32 * dx) * f32::sin(j as f32 * dx) + rng.gen::<f32>() * 0.01;
        }
    }
    plot(
        init_data.slice(s![.., 0]).iter().map(|x| &x.re),
        "plot/in.svg",
    )?;
    let save_w = init_data
        .map(|x| (128.0 * (x.re + 1.0)) as u8)
        .into_raw_vec();
    let a = GrayImage::from_raw(N as u32, N as u32, save_w).unwrap();
    a.save("plot/in.png")?;

    //ndarray_image::save_gray_image("plot/test.jpg", save_w.view())?;

    let w_buffer = new_buffer(&pro_que)?;
    let wnew_buffer = new_buffer(&pro_que)?;
    let what_buffer = new_buffer(&pro_que)?;
    let psihat_buffer = new_buffer(&pro_que)?;
    let dxu_buffer = new_buffer(&pro_que)?;
    let dyu_buffer = new_buffer(&pro_que)?;
    wnew_buffer
        .write(init_data.as_slice().ok_or(anyhow!("Oh no!"))?)
        .enq()?;
    plot_from_gpu(&wnew_buffer, "plot/wnew.png")?;

    let config = VkFFTConfiguration {
        FFTdim: 2,
        size: [N as u64, N as u64, 0, 0],
        numberBatches: 1,
        device: &mut pro_que.device().as_ptr(),
        context: &mut pro_que.context().as_ptr(),
        bufferSize: &mut (8 * (N * N) as u64), // 8 = sizeof(Complex<f32>)
        normalize: 1,
        isInputFormatted: 1,
        ..Default::default()
    };

    let mut app = VkFFTApplication {
        ..Default::default()
    };

    let res = unsafe { ocl_vkfft::initializeVkFFT(&mut app, config) };
    assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

    // ------------------------------------------------------------------------- //
    let mut launch1 = VkFFTLaunchParams {
        commandQueue: &mut pro_que.queue().as_ptr(),
        inputBuffer: &mut wnew_buffer.as_ptr(),
        buffer: &mut what_buffer.as_ptr(),
        ..Default::default()
    };

    // Diffusion new_w -> what -> what -> w

    let kernel_invmlap = unsafe {
        pro_que
            .kernel_builder("inv_mlap")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&what_buffer)
            .arg(&psihat_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?
    };

    let kernel_dxu = unsafe {
        pro_que
            .kernel_builder("diff_y")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&psihat_buffer)
            .arg(&psihat_buffer)
            //.arg(&dxu_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?
    };

    let kernel_dyu = unsafe {
        pro_que
            .kernel_builder("mdiff_x")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&psihat_buffer)
            .arg(&dyu_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?
    };
    let mut launch2 = VkFFTLaunchParams {
        commandQueue: &mut pro_que.queue().as_ptr(),
        inputBuffer: &mut dxu_buffer.as_ptr(),
        buffer: &mut dxu_buffer.as_ptr(),
        ..Default::default()
    };

    let mut launch3 = VkFFTLaunchParams {
        commandQueue: &mut pro_que.queue().as_ptr(),
        inputBuffer: &mut dyu_buffer.as_ptr(),
        buffer: &mut dyu_buffer.as_ptr(),
        ..Default::default()
    };

    let kernel_advection = unsafe {
        pro_que
            .kernel_builder("advection")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&w_buffer)
            .arg(&wnew_buffer)
            .arg(&dxu_buffer)
            .arg(&dyu_buffer)
            .arg(N as i32)
            .arg(L)
            .arg(dt)
            .build()?
    };

    pro_que.queue().finish()?;
    println!("Initialization complete. (fake)");

    // ------------------------------------------------------------------------- //

    let instant = Instant::now();
    unsafe {
        //for _i in 0..niter {
        let res = ocl_vkfft::VkFFTAppend(&mut app, -1, &mut launch1);
        assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);
        printmax(&what_buffer, "what")?;
        wnew_buffer.copy(&w_buffer, None, None).enq()?;
        kernel_invmlap.enq()?;

        printmax(&psihat_buffer, "psihat")?;
        kernel_dxu.enq()?;

        psihat_buffer.copy(&dxu_buffer, None, None).enq()?;

        //kernel_dyu.enq()?;

        //let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch2);
        //assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);
        //let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch3);
        //assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

        kernel_advection.enq()?;
        //}
    }
    pro_que.queue().finish()?;
    println!("Loop time: {:?}", instant.elapsed());

    // ------------------------------------------------------------------------- //

    printmax(&wnew_buffer, "wnew")?;

    printmax(&wnew_buffer, "wnew")?;

    //wnew_buffer.copy(&what_buffer, None, None).enq()?;

    printmax(&what_buffer, "what")?;
    printmax(&psihat_buffer, "psihat")?;
    printmax(&dxu_buffer, "dxu")?;

    plot_from_gpu(&wnew_buffer, "plot/end.png")?;
    plot_from_gpu(&dxu_buffer, "plot/dxu.png")?;

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
