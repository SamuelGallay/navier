extern crate ocl;
extern crate ocl_vkfft;
extern crate rand;

use core::ffi::c_void;
use ocl::{Buffer, ProQue};
use ocl_vkfft::{VkFFTApplication, VkFFTConfiguration};
use rand::Rng;

fn trivial() -> ocl::Result<()> {
    const N: usize = 32;
    let mut rng = rand::thread_rng();

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
        __kernel void sub(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] -= scalar;
        }
    "#;

    let pro_que = ProQue::builder().src(src).dims(2 * N).build()?;

    let init_data: [f32; 2 * N] = core::array::from_fn(|_| rng.gen::<f32>());
    //println!("{:?}", init_data);
    let fft_buffer = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(2 * N)
        .copy_host_slice(&init_data)
        .build()?;

    //let buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("add")
        .arg(&fft_buffer)
        .arg(1f32)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut back_data = vec![0.0f32; fft_buffer.len()];
    fft_buffer.read(&mut back_data).enq()?;
    //println!("{:?}", back_data);

    //let config = VkFFTConfiguration {
    //    FFTdim: 1,
    //    size: [N as u64, 0, 0, 0],
    //    context: pro_que.context().as_ptr() as *mut *mut c_void,
    //    commandQueue: pro_que.queue().as_ptr() as *mut *mut c_void,
    //    ..Default::default()
    //};
    //println!("{:?}", config);

    //let app = VkFFTApplication {
    //    ..Default::default()
    //};

    //let _ = unsafe { ocl_vkfft::test(42) };
    //let res = unsafe { ocl_vkfft::initializeVkFFT(&app, config) };
    //let res = unsafe { initializeVkFFT(&app, config) };
    //println!("Result : {:?}", res);

    Ok(())
}

fn main() {
    match trivial() {
        Ok(_) => println!("Program exited successfully."),
        Err(e) => println!("Not working : {e:?}"),
    }
}
