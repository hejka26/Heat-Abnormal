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

    ui.global::<RustActions>().on_save_file({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::save_file(&ui_handle, &images_model))
    });

    ui.global::<RustActions>().on_close_file({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move |index| {
            handle_action(|| actions::close_file(&ui_handle, &images_model, index as usize))
        }
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

    // 5.5 Bind Skeletonize
    ui.global::<RustActions>().on_skeletonize({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::skeletonize(&ui_handle, &images_model))
    });

    // 6. Bind Selective Stretch
    ui.global::<RustActions>().on_selective_stretch({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move |p1, p2, q3, q4| {
            handle_action(|| {
                actions::selective_stretch(
                    &ui_handle,
                    &images_model,
                    p1 as u8,
                    p2 as u8,
                    q3 as u8,
                    q4 as u8,
                )
            })
        }
    });

    ui.global::<RustActions>().on_negate({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::negate(&ui_handle, &images_model))
    });

    ui.global::<RustActions>().on_convert_to_hsv({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::convert_to_hsv(&ui_handle, &images_model))
    });

    ui.global::<RustActions>().on_convert_to_lab({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::convert_to_lab(&ui_handle, &images_model))
    });

    ui.global::<RustActions>().on_split_rgb({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || handle_action(|| actions::split_rgb(&ui_handle, &images_model))
    });

    ui.global::<RustActions>().on_show_about(|| actions::show_about());

    ui.global::<RustActions>().on_posterize({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        // Slint przesyła int (i32), więc rzutujemy na u8
        move |levels| handle_action(|| actions::posterize(&ui_handle, &images_model, levels as u8))
    });

    ui.global::<RustActions>().on_segment({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move |threshold| handle_action(|| actions::segment(&ui_handle, &images_model, threshold as u8))
    });

    ui.global::<RustActions>().on_median_filter({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move |kernel_size| {
            handle_action(|| actions::median_filter(&ui_handle, &images_model, kernel_size))
        }
    });

    ui.global::<RustActions>().on_dilate({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move |iterations| {
            handle_action(|| actions::dilate(&ui_handle, &images_model, iterations))
        }
    });

    ui.global::<RustActions>().on_custom_filter({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move |m0, m1, m2, m3, m4, m5, m6, m7, m8| {
            handle_action(|| {
                actions::custom_filter(
                    &ui_handle,
                    &images_model,
                    [m0, m1, m2, m3, m4, m5, m6, m7, m8],
                )
            })
        }
    });
}
