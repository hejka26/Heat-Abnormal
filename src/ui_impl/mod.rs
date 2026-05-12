pub mod actions;
mod helper;

use crate::{ImageContainer, ImageStore, MainWindow, RustActions};
use slint::{ComponentHandle, ModelRc, VecModel};
use std::rc::Rc;

fn handle_action<F>(action: F)
where
    F: FnOnce() -> Result<(), String>,
{
    if let Err(e) = action() {
        eprintln!("Sending to UI: {}", e);
    }
}

/// Registers all Slint UI callbacks and initializes the main state
pub fn setup_callbacks(ui: &MainWindow, images_model: &Rc<VecModel<ImageContainer>>) {
    // 1. Give the empty model to Slint initially
    ui.global::<ImageStore>()
        .set_loaded_images(ModelRc::from(images_model.clone()));

    // 2. Bind Open File
    ui.global::<RustActions>().on_open_file({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::open_file(&ui_handle, &images_model))
    });

    // 3. Bind Convert Color
    ui.global::<RustActions>().on_convert_color({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::convert_color(&ui_handle, &images_model))
    });

    // 4. Bind Calculate Gray Histogram
    ui.global::<RustActions>().on_calculate_gray_histogram({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::calculate_gray_histogram(&ui_handle, &images_model))
    });

    // 5. Bind Equalize Histogram
    ui.global::<RustActions>().on_equalize_histogram({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::equalize_histogram(&ui_handle, &images_model))
    });

    // 6. Bind Selective Stretch
    ui.global::<RustActions>().on_selective_stretch({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        // Assuming Slint passes these as i32, cast them to u8 for your action function
        move |min_val, max_val| {
            handle_action(|| {
                actions::selective_stretch(&ui_handle, &images_model, min_val as u8, max_val as u8)
            })
        }
    });

    ui.global::<RustActions>().on_posterize({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        // Slint przesyła int (i32), więc rzutujemy na u8
        move |levels| handle_action(|| actions::posterize(&ui_handle, &images_model, levels as u8))
    });
}
