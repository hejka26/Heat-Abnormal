use opencv::{
    core::{CV_8UC1, CV_8UC3, Scalar},
    imgproc,
    prelude::*,
};
use slint::{Image, Rgb8Pixel, SharedPixelBuffer};

pub fn bga_to_slint(mat: &Mat) -> Result<Image, String> {
    let mut rgb_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        &mat,
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
        &mat,
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

pub fn slint_to_rgb(img: &Image) -> Result<Mat, String> {
    let Some(pixel_buffer) = img.to_rgb8() else {
        return Err("failed to convert to rgb buffer from image".to_string());
    };

    let width = pixel_buffer.width() as i32;
    let height = pixel_buffer.height() as i32;

    let mut rgb_mat =
        match Mat::new_rows_cols_with_default(height, width, CV_8UC3, Scalar::default()) {
            Ok(mat) => mat,
            Err(e) => {
                return Err(format!("Failed to create matrix: {}", e));
            }
        };

    let Ok(mat_bytes) = rgb_mat.data_bytes_mut() else {
        return Err("failed to extract bytes from dummy mat".to_string());
    };
    mat_bytes.copy_from_slice(pixel_buffer.as_bytes());

    Ok(rgb_mat)
}

pub fn slint_to_gray(img: &Image) -> Result<Mat, String> {
    let Some(pixel_buffer) = img.to_rgb8() else {
        return Err("failed to convert to rgb buffer from image".to_string());
    };

    let width = pixel_buffer.width() as i32;
    let height = pixel_buffer.height() as i32;

    let mut rgb_mat =
        match Mat::new_rows_cols_with_default(height, width, CV_8UC3, Scalar::default()) {
            Ok(mat) => mat,
            Err(e) => {
                return Err(format!("Failed to create matrix: {}", e));
            }
        };

    let Ok(mat_bytes) = rgb_mat.data_bytes_mut() else {
        return Err("failed to extract bytes from dummy mat".to_string());
    };
    mat_bytes.copy_from_slice(pixel_buffer.as_bytes());

    let mut gray_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        &rgb_mat,
        &mut gray_mat,
        imgproc::COLOR_RGB2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        return Err(format!("failed to convert color space: {}", e));
    };

    Ok(gray_mat)
}
