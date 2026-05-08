use crate::ui_impl::helper;
use opencv::{core::MatTraitConst, imgcodecs};
use slint::{ComponentHandle, Model, SharedString, VecModel, Weak};
use std::rc::Rc;

// Import the auto-generated Slint types from main.rs
use crate::{ImageContainer, ImageStore, MainWindow};

pub fn open_file(ui_handle: &Weak<MainWindow>, images_model: &Rc<VecModel<ImageContainer>>) {
    // 1. Open the dialog. Use `let else` for an early return if the user cancels.
    let Some(file_path) = rfd::FileDialog::new()
        .set_title("Select an Image")
        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tiff"])
        .pick_file()
    else {
        return;
    };

    let path_str = file_path.to_string_lossy();
    let filename = file_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mat = match imgcodecs::imread(&path_str, imgcodecs::IMREAD_COLOR) {
        Ok(m) if !m.empty() => m,
        _ => {
            eprintln!("OpenCV failed to load the image at: {}", path_str);
            return;
        }
    };

    let slint_img = match helper::bga_to_slint(&mat) {
        Ok(img) => img,
        Err(e) => {
            // 'e' contains the error string returned by your function
            eprintln!("{}", e);
            return;
        }
    };

    // 6. Append it to your VecModel
    images_model.push(ImageContainer {
        img: slint_img,
        label: SharedString::from(filename),
        color: true,
    });

    if let Some(ui) = ui_handle.upgrade() {
        let new_index = (images_model.row_count() - 1) as i32;
        ui.global::<ImageStore>().set_selected_image(new_index);
    }
}

pub fn convert_color(ui_handle: &Weak<MainWindow>, images_model: &Rc<VecModel<ImageContainer>>) {
    let Some(ui) = ui_handle.upgrade() else {
        return;
    };

    let selected_idx = ui.global::<ImageStore>().get_selected_image() as usize;

    let Some(mut img) = images_model.row_data(selected_idx) else {
        eprintln!("Couldn't retrieve img");
        return;
    };

    let conversion_result = if img.color {
        helper::slint_to_gray(&img.img).and_then(|mat| helper::gray_to_slint(&mat))
    } else {
        helper::slint_to_rgb(&img.img).and_then(|mat| helper::rgb_to_slint(&mat))
    };

    match conversion_result {
        Ok(new_img) => {
            img.img = new_img;
            img.color = !img.color;
            images_model.set_row_data(selected_idx, img);
        }
        Err(e) => eprintln!("{}", e),
    }
}
