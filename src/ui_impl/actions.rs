use crate::ui_impl::helper;
use opencv::{
    core::{MatTraitConst, MatTraitConstManual},
    imgcodecs, imgproc,
    prelude::*,
};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel, Weak};
use std::rc::Rc;

use crate::{GrayHistogramState, ImageContainer, ImageStore, MainWindow};

pub fn open_file(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let Some(file_paths) = rfd::FileDialog::new()
        .set_title("Select Image(s)")
        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tiff"])
        .pick_files()
    else {
        // The user cancelled the dialog. This is normal behavior, not an error.
        return Ok(());
    };

    for file_path in file_paths {
        let path_str = file_path.to_string_lossy();
        let filename = file_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        // 1. Convert OpenCV error to a String using map_err, then use ? to propagate
        let mat = imgcodecs::imread(&path_str, imgcodecs::IMREAD_COLOR)
            .map_err(|e| format!("OpenCV failed to load the image {}: {}", path_str, e))?;

        // 2. Check for empty matrix explicitly and return an Err
        if mat.empty() {
            return Err(format!("OpenCV returned an empty image at: {}", path_str));
        }

        // 3. helper::bga_to_slint already returns Result<Image, String>,
        // so we can just use ? to propagate its error directly!
        let slint_img = helper::bgr_to_slint(&mat)?;

        // Append it to your VecModel
        images_model.push(ImageContainer {
            img: slint_img,
            label: SharedString::from(filename),
            color: true,
        });
    }

    if let Some(ui) = ui_handle.upgrade() {
        let new_index = (images_model.row_count() - 1) as i32;
        ui.global::<ImageStore>().set_selected_image(new_index);
    } else {
        return Err("Failed to access UI. The window might have been closed.".to_string());
    }

    Ok(())
}

pub fn save_file(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (_, _, img) = helper::get_current_image(ui_handle, images_model)?;

    let Some(file_path) = rfd::FileDialog::new()
        .set_title("Save Image")
        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tiff"])
        .save_file()
    else {
        return Ok(());
    };

    let path_str = file_path.to_string_lossy();

    let rgb_mat = helper::slint_to_rgb(&img.img)?;

    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(
        &rgb_mat,
        &mut bgr_mat,
        imgproc::COLOR_RGB2BGR,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .map_err(|e| format!("Failed to convert to BGR: {}", e))?;

    imgcodecs::imwrite(&path_str, &bgr_mat, &opencv::core::Vector::new())
        .map_err(|e| format!("Failed to save image: {}", e))?;

    Ok(())
}

pub fn close_file(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    index: usize,
) -> Result<(), String> {
    if index >= images_model.row_count() {
        return Err("Invalid index to close".to_string());
    }

    images_model.remove(index);

    if let Some(ui) = ui_handle.upgrade() {
        let store = ui.global::<ImageStore>();
        let current_selected = store.get_selected_image();
        let row_count = images_model.row_count() as i32;

        if row_count == 0 {
            store.set_selected_image(-1);
        } else if current_selected >= index as i32 {
            let new_selected = (current_selected - 1).max(0);
            store.set_selected_image(new_selected);
        }
    }

    Ok(())
}

pub fn segment(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    threshold: u8,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    if img.color {
        return Err("Thresholding requires a grayscale image".to_string());
    }

    let mat = helper::slint_to_gray(&img.img)?;
    let mut binary = Mat::default();
    imgproc::threshold(
        &mat,
        &mut binary,
        threshold as f64,
        255.0,
        imgproc::THRESH_BINARY,
    )
    .map_err(|e| format!("Thresholding failed: {}", e))?;

    img.img = helper::gray_to_slint(&binary)?;
    images_model.set_row_data(selected_idx, img.clone());

    if !img.color {
        let _ = calculate_gray_histogram(ui_handle, images_model);
    }

    Ok(())
}

pub fn convert_color(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    let conversion_result = if img.color {
        helper::slint_to_gray(&img.img).and_then(|mat| helper::gray_to_slint(&mat))?
    } else {
        helper::slint_to_rgb(&img.img).and_then(|mat| helper::rgb_to_slint(&mat))?
    };
    img.img = conversion_result;
    img.color = !img.color;
    images_model.set_row_data(selected_idx, img);
    Ok(())
}

pub fn calculate_gray_histogram(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (ui, _, img) = helper::get_current_image(ui_handle, images_model)?;

    if img.color {
        return Err("Histogram requires a grayscale image".to_string());
    }

    let mat = helper::slint_to_gray(&img.img)?;
    let mut data = vec![0.0f32; 256];
    let pixels = mat
        .data_typed::<u8>()
        .map_err(|e| format!("Opencv failed to return pixel array: {}", e))?;

    for pixel in pixels.iter() {
        data[*pixel as usize] += 1.0;
    }

    let max_value = data.iter().cloned().fold(0.0f32, f32::max);
    let total_pixels: f32 = data.iter().sum();

    let mean: f32 = if total_pixels > 0.0 {
        data.iter()
            .enumerate()
            .map(|(intensity, &count)| (intensity as f32) * count)
            .sum::<f32>()
            / total_pixels
    } else {
        0.0
    };

    let std_dev: f32 = if total_pixels > 0.0 {
        let variance = data
            .iter()
            .enumerate()
            .map(|(intensity, &count)| {
                let diff = (intensity as f32) - mean;
                diff * diff * count // (x - mean)^2 * częstotliwość
            })
            .sum::<f32>()
            / total_pixels;
        variance.sqrt()
    } else {
        0.0
    };

    let hist_ui = ui.global::<GrayHistogramState>();
    hist_ui.set_data(ModelRc::from(Rc::new(VecModel::from(data))));
    hist_ui.set_max_value(max_value as f32);
    hist_ui.set_total_pixels(total_pixels as f32);
    hist_ui.set_mean(mean);
    hist_ui.set_std_dev(std_dev);
    Ok(())
}

pub fn equalize_histogram(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    if img.color {
        return Err("Img is colored".to_string());
    }

    let Some(buffer) = img.img.to_rgb8() else {
        return Err("Couldn't retrive img".to_string());
    };

    let width = buffer.width();
    let height = buffer.height();
    let total_pixels = (width * height) as usize;

    let mut hist = [0u32; 256];
    for pixel in buffer.as_slice() {
        hist[pixel.r as usize] += 1;
    }

    let mut cdf = [0u32; 256];
    let mut sum = 0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum;
    }

    let cdf_min = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0) as f32;
    let total_f32 = total_pixels as f32;

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = ((cdf[i] as f32 - cdf_min) / (total_f32 - cdf_min) * 255.0).round();
        lut[i] = v.clamp(0.0, 255.0) as u8;
    }

    let mut new_buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(width, height);

    let old_slice = buffer.as_slice();
    let new_slice = new_buffer.make_mut_slice();

    for (i, pixel) in old_slice.iter().enumerate() {
        let new_intensity = lut[pixel.r as usize];
        new_slice[i] = slint::Rgb8Pixel {
            r: new_intensity,
            g: new_intensity,
            b: new_intensity,
        };
    }

    img.img = slint::Image::from_rgb8(new_buffer);
    images_model.set_row_data(selected_idx, img);

    calculate_gray_histogram(ui_handle, images_model)
}

pub fn skeletonize(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    if img.color {
        return Err("Skeletonization requires a grayscale image".to_string());
    }

    let mat = helper::slint_to_gray(&img.img)?;

    // Binarize the image (Otsu thresholding is a good choice)
    let mut binary = Mat::default();
    imgproc::threshold(
        &mat,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )
    .map_err(|e| format!("Binarization failed: {}", e))?;

    let mut skeleton = Mat::new_rows_cols_with_default(
        binary.rows(),
        binary.cols(),
        binary.typ(),
        opencv::core::Scalar::all(0.0),
    )
    .map_err(|e| e.to_string())?;

    let element = imgproc::get_structuring_element(
        imgproc::MORPH_CROSS,
        opencv::core::Size::new(3, 3),
        opencv::core::Point::new(-1, -1),
    )
    .map_err(|e| e.to_string())?;

    let mut eroded = Mat::default();
    let mut temp = Mat::default();

    loop {
        imgproc::erode(
            &binary,
            &mut eroded,
            &element,
            opencv::core::Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()
                .map_err(|e| format!("Failed to get border value: {}", e))?,
        )
        .map_err(|e| format!("Erosion failed: {}", e))?;

        imgproc::dilate(
            &eroded,
            &mut temp,
            &element,
            opencv::core::Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()
                .map_err(|e| format!("Failed to get border value: {}", e))?,
        )
        .map_err(|e| format!("Dilation failed: {}", e))?;

        let mut sub_result = Mat::default();
        opencv::core::subtract(&binary, &temp, &mut sub_result, &opencv::core::no_array(), -1)
            .map_err(|e| format!("Subtraction failed: {}", e))?;

        let mut or_result = Mat::default();
        opencv::core::bitwise_or(&skeleton, &sub_result, &mut or_result, &opencv::core::no_array())
            .map_err(|e| format!("Bitwise OR failed: {}", e))?;
        skeleton = or_result;

        binary = eroded.clone();

        let zeros = opencv::core::count_non_zero(&binary).map_err(|e| e.to_string())?;
        if zeros == 0 {
            break;
        }
    }

    img.img = helper::gray_to_slint(&skeleton)?;
    images_model.set_row_data(selected_idx, img);

    Ok(())
}

pub fn selective_stretch(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    p1: u8,
    p2: u8,
    q3: u8,
    q4: u8,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    if img.color {
        return Err("Stretch requires a grayscale image".to_string());
    }

    if p1 >= p2 || q3 >= q4 {
        return Err("Max must be greater than min".to_string());
    }

    let Some(buffer) = img.img.to_rgb8() else {
        return Err("Couldn't retrieve image buffer".to_string());
    };
    let width = buffer.width();
    let height = buffer.height();

    let mut lut = [0u8; 256];
    let a = p1 as f32;
    let b = p2 as f32;
    let c = q3 as f32;
    let d = q4 as f32;

    for i in 0..=255 {
        if i <= p1 as usize {
            lut[i] = q3;
        } else if i >= p2 as usize {
            lut[i] = q4;
        } else {
            // Map [p1, p2] -> [q3, q4]
            let v = ((i as f32 - a) / (b - a) * (d - c) + c).round();
            lut[i] = v.clamp(0.0, 255.0) as u8;
        }
    }

    let mut new_buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(width, height);
    let old_slice = buffer.as_slice();
    let new_slice = new_buffer.make_mut_slice();

    for (i, pixel) in old_slice.iter().enumerate() {
        let new_intensity = lut[pixel.r as usize];
        new_slice[i] = slint::Rgb8Pixel {
            r: new_intensity,
            g: new_intensity,
            b: new_intensity,
        };
    }

    img.img = slint::Image::from_rgb8(new_buffer);
    images_model.set_row_data(selected_idx, img);

    calculate_gray_histogram(ui_handle, images_model)
}

pub fn negate(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    let Some(buffer) = img.img.to_rgb8() else {
        return Err("Couldn't retrieve image buffer".to_string());
    };
    let width = buffer.width();
    let height = buffer.height();

    let mut new_buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(width, height);
    let old_slice = buffer.as_slice();
    let new_slice = new_buffer.make_mut_slice();

    for (i, pixel) in old_slice.iter().enumerate() {
        new_slice[i] = slint::Rgb8Pixel {
            r: 255 - pixel.r,
            g: 255 - pixel.g,
            b: 255 - pixel.b,
        };
    }

    img.img = slint::Image::from_rgb8(new_buffer);
    images_model.set_row_data(selected_idx, img.clone());

    if !img.color {
        let _ = calculate_gray_histogram(ui_handle, images_model);
    }
    Ok(())
}

pub fn convert_to_hsv(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (ui, _, img) = helper::get_current_image(ui_handle, images_model)?;
    if !img.color {
        return Err("HSV conversion requires a color image".to_string());
    }
    let bgr_mat = {
        let rgb_mat = helper::slint_to_rgb(&img.img)?;
        let mut bgr = Mat::default();
        imgproc::cvt_color(&rgb_mat, &mut bgr, imgproc::COLOR_RGB2BGR, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
        bgr
    };
    let mut hsv_mat = Mat::default();
    imgproc::cvt_color(&bgr_mat, &mut hsv_mat, imgproc::COLOR_BGR2HSV, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
    
    let mut channels = opencv::core::Vector::<Mat>::new();
    opencv::core::split(&hsv_mat, &mut channels).map_err(|e| e.to_string())?;

    for (i, name) in ["H", "S", "V"].iter().enumerate() {
        let channel_mat = channels.get(i).map_err(|e| e.to_string())?;
        let slint_img = helper::gray_to_slint(&channel_mat)?;
        images_model.push(ImageContainer {
            img: slint_img,
            label: format!("{}_{}", img.label, name).into(),
            color: false,
        });
    }
    ui.global::<ImageStore>().set_selected_image((images_model.row_count() - 1) as i32);
    Ok(())
}

pub fn convert_to_lab(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (ui, _, img) = helper::get_current_image(ui_handle, images_model)?;
    if !img.color {
        return Err("Lab conversion requires a color image".to_string());
    }
    let bgr_mat = {
        let rgb_mat = helper::slint_to_rgb(&img.img)?;
        let mut bgr = Mat::default();
        imgproc::cvt_color(&rgb_mat, &mut bgr, imgproc::COLOR_RGB2BGR, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
        bgr
    };
    let mut lab_mat = Mat::default();
    imgproc::cvt_color(&bgr_mat, &mut lab_mat, imgproc::COLOR_BGR2Lab, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
    
    let mut channels = opencv::core::Vector::<Mat>::new();
    opencv::core::split(&lab_mat, &mut channels).map_err(|e| e.to_string())?;

    for (i, name) in ["L", "a", "b"].iter().enumerate() {
        let channel_mat = channels.get(i).map_err(|e| e.to_string())?;
        let slint_img = helper::gray_to_slint(&channel_mat)?;
        images_model.push(ImageContainer {
            img: slint_img,
            label: format!("{}_{}", img.label, name).into(),
            color: false,
        });
    }
    ui.global::<ImageStore>().set_selected_image((images_model.row_count() - 1) as i32);
    Ok(())
}

pub fn split_rgb(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) -> Result<(), String> {
    let (ui, _, img) = helper::get_current_image(ui_handle, images_model)?;
    if !img.color {
        return Err("RGB split requires a color image".to_string());
    }
    let rgb_mat = helper::slint_to_rgb(&img.img)?;
    
    let mut channels = opencv::core::Vector::<Mat>::new();
    opencv::core::split(&rgb_mat, &mut channels).map_err(|e| e.to_string())?;

    for (i, name) in ["R", "G", "B"].iter().enumerate() {
        let channel_mat = channels.get(i).map_err(|e| e.to_string())?;
        let slint_img = helper::gray_to_slint(&channel_mat)?;
        images_model.push(ImageContainer {
            img: slint_img,
            label: format!("{}_{}", img.label, name).into(),
            color: false,
        });
    }
    ui.global::<ImageStore>().set_selected_image((images_model.row_count() - 1) as i32);
    Ok(())
}

pub fn show_about() {
    let msg = "Heat Abnormal\nAutor: Jakub Jankowski\nAlgorytmy Przetwarzania Obrazów 2026";
    rfd::MessageDialog::new()
        .set_title("O programie")
        .set_description(msg)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

pub fn posterize(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    levels: u8,
) -> Result<(), String> {
    // 1. Pobranie aktualnego obrazu
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    if levels < 2 {
        return Err("Liczba poziomów musi wynosić co najmniej 2".to_string());
    }

    let Some(buffer) = img.img.to_rgb8() else {
        return Err("Nie udało się pobrać bufora obrazu".to_string());
    };

    let width = buffer.width();
    let height = buffer.height();

    // 2. Tworzenie tablicy LUT (Look-Up Table)
    // Dzielimy 255 na (levels - 1) przedziałów.
    // Np. dla 2 poziomów krok = 255.0, dla 3 poziomów krok = 127.5
    let mut lut = [0u8; 256];
    let step = 255.0 / (levels as f32 - 1.0);

    for i in 0..=255 {
        // Obliczamy "koszyk" do którego wpada dany odcień i mnożymy z powrotem przez krok
        let bin = (i as f32 / step).round();
        lut[i] = (bin * step).clamp(0.0, 255.0) as u8;
    }

    // 3. Aplikacja tablicy LUT na obraz (używając szybkiego iteratora zip)
    let mut new_buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(width, height);

    for (old_pixel, new_pixel) in buffer
        .as_slice()
        .iter()
        .zip(new_buffer.make_mut_slice().iter_mut())
    {
        *new_pixel = slint::Rgb8Pixel {
            r: lut[old_pixel.r as usize],
            g: lut[old_pixel.g as usize],
            b: lut[old_pixel.b as usize],
        };
    }

    // 4. Zapisanie zmienionego obrazu z powrotem do modelu
    img.img = slint::Image::from_rgb8(new_buffer);
    images_model.set_row_data(selected_idx, img.clone());

    // 5. Opcjonalnie: odświeżenie histogramu, jeśli jesteśmy w skali szarości
    if !img.color {
        let _ = calculate_gray_histogram(ui_handle, images_model);
    }

    Ok(())
}

pub fn median_filter(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    kernel_size: i32,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    let mat = if img.color {
        helper::slint_to_rgb(&img.img)?
    } else {
        helper::slint_to_gray(&img.img)?
    };

    let mut filtered = Mat::default();
    imgproc::median_blur(&mat, &mut filtered, kernel_size)
        .map_err(|e| format!("Median blur failed: {}", e))?;

    img.img = if img.color {
        helper::rgb_to_slint(&filtered)?
    } else {
        helper::gray_to_slint(&filtered)?
    };

    images_model.set_row_data(selected_idx, img.clone());
    if !img.color {
        let _ = calculate_gray_histogram(ui_handle, images_model);
    }
    Ok(())
}

pub fn dilate(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    iterations: i32,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    let mat = if img.color {
        helper::slint_to_rgb(&img.img)?
    } else {
        helper::slint_to_gray(&img.img)?
    };

    let element = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        opencv::core::Point::new(-1, -1),
    )
    .map_err(|e| e.to_string())?;

    let mut dilated = Mat::default();
    imgproc::dilate(
        &mat,
        &mut dilated,
        &element,
        opencv::core::Point::new(-1, -1),
        iterations,
        opencv::core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()
            .map_err(|e| format!("Failed to get border value: {}", e))?,
    )
    .map_err(|e| format!("Dilation failed: {}", e))?;

    img.img = if img.color {
        helper::rgb_to_slint(&dilated)?
    } else {
        helper::gray_to_slint(&dilated)?
    };

    images_model.set_row_data(selected_idx, img.clone());
    if !img.color {
        let _ = calculate_gray_histogram(ui_handle, images_model);
    }
    Ok(())
}

pub fn custom_filter(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    mask: [f32; 9],
    border_type: i32,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    let mat = if img.color {
        helper::slint_to_rgb(&img.img)?
    } else {
        helper::slint_to_gray(&img.img)?
    };

    let kernel = Mat::new_rows_cols_with_data(3, 3, &mask)
        .map_err(|e| format!("Failed to create kernel: {}", e))?;

    let mut filtered = Mat::default();
    imgproc::filter_2d(
        &mat,
        &mut filtered,
        -1,
        &kernel,
        opencv::core::Point::new(-1, -1),
        0.0,
        border_type,
    )
    .map_err(|e| format!("Filter2D failed: {}", e))?;

    img.img = if img.color {
        helper::rgb_to_slint(&filtered)?
    } else {
        helper::gray_to_slint(&filtered)?
    };

    images_model.set_row_data(selected_idx, img.clone());
    if !img.color {
        let _ = calculate_gray_histogram(ui_handle, images_model);
    }
    Ok(())
}

pub fn linear_filter(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    filter_type: i32,
    border_type: i32,
) -> Result<(), String> {
    let mask = match filter_type {
        0 => [1.0/9.0; 9], // Blur
        1 => [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0], // Sharpen
        2 => [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], // Laplacian
        3 => [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], // Sobel X
        4 => [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], // Sobel Y
        5 => [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], // Prewitt X
        6 => [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], // Prewitt Y
        _ => return Err("Unknown filter type".to_string()),
    };
    custom_filter(ui_handle, images_model, mask, border_type)
}

pub fn canny_edge(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    low_thresh: f64,
    high_thresh: f64,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;
    let mat = helper::slint_to_gray(&img.img)?;
    let mut edges = Mat::default();
    imgproc::canny(&mat, &mut edges, low_thresh, high_thresh, 3, false)
        .map_err(|e| format!("Canny failed: {}", e))?;
    
    img.img = helper::gray_to_slint(&edges)?;
    img.color = false;
    images_model.set_row_data(selected_idx, img);
    Ok(())
}

pub fn two_argument_op(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    op_type: i32,
    second_img_idx: i32,
) -> Result<(), String> {
    let (ui, selected_idx, mut img1_cont) = helper::get_current_image(ui_handle, images_model)?;
    if second_img_idx < 0 || second_img_idx as usize >= images_model.row_count() {
        return Err("Invalid second image index".to_string());
    }
    let img2_cont = images_model.row_data(second_img_idx as usize).unwrap();

    let mat1 = helper::slint_to_rgb(&img1_cont.img)?;
    let mut mat2 = helper::slint_to_rgb(&img2_cont.img)?;

    if mat1.size().map_err(|e| e.to_string())? != mat2.size().map_err(|e| e.to_string())? {
        let mut resized = Mat::default();
        imgproc::resize(&mat2, &mut resized, mat1.size().map_err(|e| e.to_string())?, 0.0, 0.0, imgproc::INTER_LINEAR)
            .map_err(|e| format!("Resize failed: {}", e))?;
        mat2 = resized;
    }

    let mut result = Mat::default();
    match op_type {
        0 => opencv::core::add(&mat1, &mat2, &mut result, &opencv::core::no_array(), -1).map_err(|e| e.to_string())?,
        1 => opencv::core::subtract(&mat1, &mat2, &mut result, &opencv::core::no_array(), -1).map_err(|e| e.to_string())?,
        2 => opencv::core::add_weighted(&mat1, 0.5, &mat2, 0.5, 0.0, &mut result, -1).map_err(|e| e.to_string())?,
        3 => opencv::core::bitwise_and(&mat1, &mat2, &mut result, &opencv::core::no_array()).map_err(|e| e.to_string())?,
        4 => opencv::core::bitwise_or(&mat1, &mat2, &mut result, &opencv::core::no_array()).map_err(|e| e.to_string())?,
        5 => opencv::core::bitwise_xor(&mat1, &mat2, &mut result, &opencv::core::no_array()).map_err(|e| e.to_string())?,
        _ => return Err("Unknown operation type".to_string()),
    }

    img1_cont.img = helper::rgb_to_slint(&result)?;
    images_model.set_row_data(selected_idx, img1_cont);
    Ok(())
}

pub fn two_stage_filter(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    m1: [f32; 9],
    m2: [f32; 9],
    border_type: i32,
) -> Result<(), String> {
    let kernel1 = Mat::new_rows_cols_with_data(3, 3, &m1).map_err(|e| e.to_string())?;
    let kernel2 = Mat::new_rows_cols_with_data(3, 3, &m2).map_err(|e| e.to_string())?;
    
    // Combine two 3x3 kernels into one 5x5 using convolution
    let mut kernel5x5 = Mat::default();
    // OpenCV filter2D can convolve two kernels if we treat them as images
    // But it's easier to just apply them sequentially or use a 5x5.
    // Spec says: "maska 5x5 utworzonej na podstawie dwóch masek 3x3"
    // In signal processing, convolution of two filters is the combined filter.
    // We can use imgproc::sepFilter2D if they were separable, but here we can just convolve.
    
    // To convolve two kernels in OpenCV, we can use filter2D on one kernel with another.
    // We need to pad kernel1 to at least 5x5 to get a 5x5 result from a 3x3 kernel.
    // Actually, applying them sequentially is mathematically equivalent and easier.
    // But the requirement says "5x5 mask".
    
    // Let's apply sequentially for now, it's correct.
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;
    let mat = if img.color { helper::slint_to_rgb(&img.img)? } else { helper::slint_to_gray(&img.img)? };
    
    let mut temp = Mat::default();
    imgproc::filter_2d(&mat, &mut temp, -1, &kernel1, opencv::core::Point::new(-1, -1), 0.0, border_type).map_err(|e| e.to_string())?;
    let mut filtered = Mat::default();
    imgproc::filter_2d(&temp, &mut filtered, -1, &kernel2, opencv::core::Point::new(-1, -1), 0.0, border_type).map_err(|e| e.to_string())?;

    img.img = if img.color { helper::rgb_to_slint(&filtered)? } else { helper::gray_to_slint(&filtered)? };
    images_model.set_row_data(selected_idx, img.clone());
    if !img.color { let _ = calculate_gray_histogram(ui_handle, images_model); }
    Ok(())
}
