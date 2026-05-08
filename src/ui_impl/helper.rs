use opencv::{
    core::{CV_8UC3, Scalar},
    imgproc,
    prelude::*,
};
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};

pub fn bga_to_slint(mat: &Mat) -> Result<Image, String> {
    let mut rgba_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        &mat,
        &mut rgba_mat,
        imgproc::COLOR_BGR2RGBA,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        return Err(format!("failed to convert color space: {}", e));
    }

    let width = rgba_mat.cols() as u32;
    let height = rgba_mat.rows() as u32;

    let Ok(pixel_bytes) = rgba_mat.data_bytes() else {
        return Err("failed to read raw bytes from image matrix".to_string());
    };

    Ok(Image::from_rgba8(
        SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(pixel_bytes, width, height),
    ))
}

pub fn rgb_to_slint(mat: &Mat) -> Result<Image, String> {
    let mut rgba_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        &mat,
        &mut rgba_mat,
        imgproc::COLOR_RGB2RGBA,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        return Err(format!("failed to convert color space: {}", e));
    }

    let width = rgba_mat.cols() as u32;
    let height = rgba_mat.rows() as u32;

    let Ok(pixel_bytes) = rgba_mat.data_bytes() else {
        return Err("failed to read raw bytes from image matrix".to_string());
    };

    Ok(Image::from_rgba8(
        SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(pixel_bytes, width, height),
    ))
}

pub fn gray_to_slint(mat: &Mat) -> Result<Image, String> {
    let mut rgba_mat = Mat::default();
    if let Err(e) = imgproc::cvt_color(
        &mat,
        &mut rgba_mat,
        imgproc::COLOR_GRAY2RGBA,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        return Err(format!("failed to convert color space: {}", e));
    }

    let width = rgba_mat.cols() as u32;
    let height = rgba_mat.rows() as u32;

    let Ok(pixel_bytes) = rgba_mat.data_bytes() else {
        return Err("failed to read raw bytes from image matrix".to_string());
    };

    Ok(Image::from_rgba8(
        SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(pixel_bytes, width, height),
    ))
}

fn slint_to_rgb(img: &Image) -> Result<Mat, String> {
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

fn slint_to_gray(img: &Image) -> Result<Mat, String> {
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
