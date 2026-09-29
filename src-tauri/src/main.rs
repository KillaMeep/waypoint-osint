// Release builds are a GUI-subsystem exe: no console window behind the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    waypoint_lib::run()
}
