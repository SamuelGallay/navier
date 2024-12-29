extern crate ndarray;
extern crate nix;
extern crate ocl;
extern crate ocl_vkfft;
extern crate rand;

pub mod utils;

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use num::complex::{Complex32, ComplexFloat};
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft::{
    VkFFTApplication, VkFFTConfiguration, VkFFTLaunchParams, VkFFTResult_VKFFT_SUCCESS,
};
use std::f32::consts::PI;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
use std::time::SystemTime;
use ndarray::Array2;
use utils::new_buffer;
//use std::thread;
//use core::time;

const SRC: &str = include_str!("kernels.cl");
const N: usize = usize::pow(2, 12);
const L: f32 = 2.0 * PI;

fn trivial() -> Result<()> {
    let _dx = L / N as f32;
    let dt: f32 = 3f32;
    let niter = 100;

    let platform = ocl::Platform::first()?;
    let device = ocl::Device::first(&platform)?;
    let context = ocl::Context::builder().build()?;
    let program = ocl::Program::builder().src(SRC).build(&context)?;
    let queue = ocl::Queue::new(&context, device, None)?;
    let transfer_queue = ocl::Queue::new(&context, device, None)?;

    let init_data = utils::noise2d(N);
    let mut w_back_data = Array2::<Complex32>::zeros((N, N));

    let w_buffer = new_buffer(&queue, N)?;
    let wnew_buffer = new_buffer(&queue, N)?;
    let what_buffer = new_buffer(&queue, N)?;
    let psihat_buffer = new_buffer(&queue, N)?;
    let dxu_buffer = new_buffer(&queue, N)?;
    let dyu_buffer = new_buffer(&queue, N)?;
    wnew_buffer
        .write(init_data.as_slice().ok_or(anyhow!("Oh no!"))?)
        .enq()?;
    utils::plot_from_gpu(&wnew_buffer, "plot/in.png")?;

    let config = VkFFTConfiguration {
        FFTdim: 2,
        size: [N as u64, N as u64, 0, 0],
        numberBatches: 1,
        device: &mut device.as_ptr(),
        context: &mut context.as_ptr(),
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
    let mut launch_what = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut w_buffer.as_ptr(),
        buffer: &mut what_buffer.as_ptr(),
        ..Default::default()
    };

    // Diffusion new_w -> what -> what -> w

    let kernel_invmlap = unsafe {
        ocl::Kernel::builder()
            .program(&program)
            .queue(queue.clone())
            .name("inv_mlap")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&what_buffer)
            .arg(&psihat_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?
    };

    let kernel_dxu = unsafe {
        ocl::Kernel::builder()
            .program(&program)
            .queue(queue.clone())
            .name("diff_y")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&psihat_buffer)
            .arg(&dxu_buffer)
            //.arg(&dxu_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?
    };

    let kernel_dyu = unsafe {
        ocl::Kernel::builder()
            .program(&program)
            .queue(queue.clone())
            .name("mdiff_x")
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&psihat_buffer)
            .arg(&dyu_buffer)
            .arg(N as i32)
            .arg(2.0 * PI / L)
            .build()?
    };
    let mut launch_dxu = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut dxu_buffer.as_ptr(),
        buffer: &mut dxu_buffer.as_ptr(),
        ..Default::default()
    };

    let mut launch_dyu = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut dyu_buffer.as_ptr(),
        buffer: &mut dyu_buffer.as_ptr(),
        ..Default::default()
    };

    let kernel_advection = unsafe {
        ocl::Kernel::builder()
            .program(&program)
            .queue(queue.clone())
            .name("advection")
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

    // ------------------------------------------------------------------------- //

    let sys_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let mut ffmpeg = Command::new("ffmpeg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .args([
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{}x{}", N, N),
            "-i",
            "pipe:",
            "-threads",
            "2",
            "-crf",
            "19",
            &format!("videos/{}.mp4", sys_time),
        ])
        .spawn()?;

    let mut ffmpeg_in = ffmpeg.stdin.take().unwrap();

    queue.finish()?;
    println!("Initialization complete. (fake)");
    let pb = ProgressBar::new(niter);

    // ------------------------------------------------------------------------- //
    let instant = Instant::now();
    unsafe {
        for _ in 0..niter {
            wnew_buffer
                .read(w_back_data.as_slice_mut().ok_or(anyhow!("Noo"))?)
                .queue(&transfer_queue)
                .enq()?;
            wnew_buffer.copy(&w_buffer, None, None).enq()?;

            let res = ocl_vkfft::VkFFTAppend(&mut app, -1, &mut launch_what);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            kernel_invmlap.enq()?;

            kernel_dyu.enq()?;
            let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch_dyu);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            kernel_dxu.enq()?;
            let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch_dxu);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            transfer_queue.finish()?;
            kernel_advection.enq()?;

            let im = utils::image_from_array(&w_back_data.mapv(|x| x.re()))?;
            ffmpeg_in.write_all(&im.to_vec())?;
            queue.finish()?;
            pb.inc(1);
        }
    }
    queue.finish()?;
    println!("Loop time: {:?}", instant.elapsed());

    // ------------------------------------------------------------------------- //

    ffmpeg_in.flush()?;
    std::mem::drop(ffmpeg_in);
    let ffmpeg_output = ffmpeg.wait_with_output().unwrap();

    match ffmpeg_output.status.code() {
        Some(0) => println!(
            "OK FFMPEG: {}",
            String::from_utf8_lossy(&ffmpeg_output.stdout)
        ),
        Some(code) => println!("Error {}", code),
        None => {}
    }

    utils::plot_from_gpu(&wnew_buffer, "plot/out.png")?;
    utils::plot_from_gpu(&dxu_buffer, "plot/dxu.png")?;
    utils::plot_from_gpu(&dyu_buffer, "plot/dyu.png")?;

    utils::printmax(&w_buffer, "w")?;
    utils::printmax(&wnew_buffer, "wnew")?;
    utils::printmax(&dxu_buffer, "dxu")?;
    utils::printmax(&dyu_buffer, "dyu")?;

    unsafe {
        ocl_vkfft::deleteVkFFT(&mut app);
    }

    println!("End trivial.");
    Ok(())
}

fn main() {
    match trivial() {
        Ok(()) => println!("Program exited successfully."),
        Err(e) => println!("Not working : {e:?}"),
    }
}
