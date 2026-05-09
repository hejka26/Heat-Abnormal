mod ui_impl; // Tells Rust to compile the new actions.rs file

use slint::{ModelRc, VecModel};
use std::rc::Rc;

// This macro generates MainWindow, ImageContainer, etc., making them available to actions.rs via `crate::`
slint::include_modules!();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ui = MainWindow::new()?;

    // Create a reactive VecModel to hold your images
    let images_model: Rc<VecModel<ImageContainer>> = Rc::new(VecModel::default());

    // Give the model to Slint
    ui.global::<ImageStore>()
        .set_loaded_images(ModelRc::from(images_model.clone()));

    // Register the callback
    ui.global::<RustActions>().on_open_file({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();

        move || {
            // Call the function we moved to actions.rs
            ui_impl::actions::open_file(&ui_handle, &images_model);
        }
    });

    ui.global::<RustActions>().on_convert_color({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();

        move || {
            // Call the function we moved to actions.rs
            ui_impl::actions::convert_color(&ui_handle, &images_model);
        }
    });

    ui.global::<RustActions>().on_calculate_gray_histogram({
        let images_model = images_model.clone();
        let ui_handle = ui.as_weak();
        move || {
            ui_impl::actions::calculate_gray_histogram(&ui_handle, &images_model);
        }
    });

    ui.run()?;
    Ok(())
}
