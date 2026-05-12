mod ui_impl; // Tells Rust to compile the new actions.rs file

use slint::VecModel;
use std::rc::Rc;

// This macro generates MainWindow, ImageContainer, etc., making them available to actions.rs via `crate::`
slint::include_modules!();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ui = MainWindow::new()?;
    let images_model: Rc<VecModel<ImageContainer>> = Rc::new(VecModel::default());

    ui_impl::setup_callbacks(&ui, &images_model);

    // Run the app
    ui.run()?;

    Ok(())
}
