fn main() {
    println!("cargo:rerun-if-changed=ui/app.slint");
    println!("cargo:rerun-if-changed=assets/icons/app.ico");
    println!("cargo:rerun-if-changed=assets/windows/app.manifest");
    slint_build::compile("ui/app.slint").expect("Slint UI compilation failed");

    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/icons/app.ico");
        resource.set_manifest_file("assets/windows/app.manifest");
        resource
            .compile()
            .expect("Windows resources compilation failed");
    }
}
