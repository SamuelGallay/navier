#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate cl_sys;
use cl_sys::{cl_platform_id, cl_command_queue, cl_mem, cl_device_id, cl_context, cl_program, cl_kernel};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

extern "C" {
    pub fn initializeVkFFT(app: *const VkFFTApplication, conf: VkFFTConfiguration) -> VkFFTResult;
    pub fn blorb_dont_call_that(app: *const VkFFTApplication, conf: VkFFTConfiguration) -> VkFFTResult;
    pub fn test(i: i32);
    //pub fn blorb_dont_call_that();
}

pub fn say_hello() {
    println!("{}", concat!(env!("OUT_DIR"), "/bindings.rs"));
}

