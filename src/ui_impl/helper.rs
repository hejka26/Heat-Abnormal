use opencv::{
    core::{CV_8UC3, Scalar},
    imgproc,
    prelude::*,
};

// Update the Slint imports to include the model/handle traits
use slint::{ComponentHandle, Image, Model, Rgb8Pixel, SharedPixelBuffer};

// Import your generated Slint types from the root crate
use crate::{ImageContainer, ImageStore, MainWindow};

pub fn bgr_to_slint(mat: &Mat) -> Result<Image, String> {
    let mut rgb_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        mat,
        &mut rgb_mat,
        imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        return Err(format!("failed to convert color space: {}", e));
    }

    let width = rgb_mat.cols() as u32;
    let height = rgb_mat.rows() as u32;

    let Ok(pixel_bytes) = rgb_mat.data_bytes() else {
        return Err("failed to read raw bytes from image matrix".to_string());
    };

    Ok(Image::from_rgb8(
        SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(pixel_bytes, width, height),
    ))
}

pub fn rgb_to_slint(mat: &Mat) -> Result<Image, String> {
    let width = mat.cols() as u32;
    let height = mat.rows() as u32;

    let Ok(pixel_bytes) = mat.data_bytes() else {
        return Err("failed to read raw bytes from image matrix".to_string());
    };

    Ok(Image::from_rgb8(
        SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(pixel_bytes, width, height),
    ))
}

pub fn gray_to_slint(mat: &Mat) -> Result<Image, String> {
    let mut rgb_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        mat,
        &mut rgb_mat,
        imgproc::COLOR_GRAY2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        return Err(format!("failed to convert color space: {}", e));
    }

    let width = rgb_mat.cols() as u32;
    let height = rgb_mat.rows() as u32;

    let Ok(pixel_bytes) = rgb_mat.data_bytes() else {
        return Err("failed to read raw bytes from image matrix".to_string());
    };

    Ok(Image::from_rgb8(
        SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(pixel_bytes, width, height),
    ))
}

pub fn get_current_image(
    ui_handle: &slint::Weak<MainWindow>,
    images_model: &std::rc::Rc<slint::VecModel<ImageContainer>>,
) -> Result<(MainWindow, usize, ImageContainer), String> {
    let ui = ui_handle
        .upgrade()
        .ok_or("Failed to upgrade UI handle: Window might be closed".to_string())?;

    let selected_idx = ui.global::<ImageStore>().get_selected_image() as usize;

    let img = images_model
        .row_data(selected_idx)
        .ok_or(format!("No image found at index {}", selected_idx))?;

    Ok((ui, selected_idx, img))
}

fn slint_to_base_mat(img: &Image) -> Result<Mat, String> {
    let pixel_buffer = img.to_rgb8().ok_or("Failed to get RGB buffer")?;
    let mut rgb_mat = Mat::new_rows_cols_with_default(
        pixel_buffer.height() as i32,
        pixel_buffer.width() as i32,
        CV_8UC3,
        Scalar::default(),
    )
    .map_err(|e| e.to_string())?;

    rgb_mat
        .data_bytes_mut()
        .map_err(|_| "failed to extract bytes")?
        .copy_from_slice(pixel_buffer.as_bytes());

    Ok(rgb_mat)
}

pub fn slint_to_rgb(img: &Image) -> Result<Mat, String> {
    slint_to_base_mat(img)
}

pub fn slint_to_gray(img: &Image) -> Result<Mat, String> {
    let rgb_mat = slint_to_base_mat(img)?;
    let mut gray_mat = Mat::default();
    imgproc::cvt_color(
        &rgb_mat,
        &mut gray_mat,
        imgproc::COLOR_RGB2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .map_err(|e| e.to_string())?;
    Ok(gray_mat)
}
