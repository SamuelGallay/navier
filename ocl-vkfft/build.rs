use cc;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=src/hello.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search=foo");

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("vkFFT/vkFFT.h")
        .clang_arg("-IvkFFT")
        .clang_arg("-DVKFFT_BACKEND=3")
        .clang_arg("-DCL_TARGET_OPENCL_VERSION=300")
        .wrap_static_fns(true)
        .allowlist_recursively(true)
        .allowlist_type("VkFFTConfiguration")
        .allowlist_type("VkFFTApplication")
        .allowlist_type("VkFFTLaunchParams")
        .allowlist_type("VkFFTResult")
        .blocklist_type("cl_platform_id")
        .blocklist_type("_cl_platform_id")
        .blocklist_type("cl_device_id")
        .blocklist_type("_cl_device_id")
        .blocklist_type("cl_command_queue")
        .blocklist_type("_cl_command_queue")
        .blocklist_type("cl_mem")
        .blocklist_type("_cl_mem")
        .blocklist_type("cl_context")
        .blocklist_type("_cl_context")
        .blocklist_type("cl_program")
        .blocklist_type("_cl_program")
        .blocklist_type("cl_kernel")
        .blocklist_type("_cl_kernel")
        //.allowlist_function("initializeVkFFT")
        //.allowlist_function("VkFFTAppend")
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");


    let output_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // This is the path to the object file.
    let obj_path = output_path.join("foo.o");
    // This is the path to the static library file.
    let lib_path = output_path.join("libfoo.a");

    let s = std::env::temp_dir().unwrap().join("bindgen").join("extern.c");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");


    cc::Build::new()
        .file("src/foo.c")
        .include(Path::new("vkFFT"))
        .flag("-DVKFFT_BACKEND=3")
        .flag("-DCL_TARGET_OPENCL_VERSION=300")
        .flag("-fkeep-inline-functions")
        .flag("-lOpenCL")
        .flag("-lm")
//        .static_flag(true)
        .cargo_metadata(true)
        .compile("foo");
}
