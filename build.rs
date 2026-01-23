fn main() {
    let cuda_path = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    cc::Build::new()
        .file("src/cupti_wrapper.c")
        .file("src/stall_counters.c")
        .include(format!("{}/include", cuda_path))
        .opt_level(2)
        .compile("cupti_wrapper");

    println!("cargo:rustc-link-lib=cupti");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rerun-if-changed=src/cupti_wrapper.c");
    println!("cargo:rerun-if-changed=src/stall_counters.c");
}
