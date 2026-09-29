fn main() {
    // The Python backend is embedded with include_dir!, which stable rustc does
    // not track on its own. Rebuild whenever a backend file changes.
    println!("cargo:rerun-if-changed=../python_backend");
    tauri_build::build()
}
