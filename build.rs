fn main() {
    println!("cargo:rerun-if-changed=ui/app.slint");
    slint_build::compile("ui/app.slint").expect("Slint UI compilation failed");
}
