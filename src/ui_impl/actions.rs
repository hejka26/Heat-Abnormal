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
    let Some(file_path) = rfd::FileDialog::new()
        .set_title("Select an Image")
        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tiff"])
        .pick_file()
    else {
        // The user cancelled the dialog. This is normal behavior, not an error.
        return Ok(());
    };

    let path_str = file_path.to_string_lossy();
    let filename = file_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    // 1. Convert OpenCV error to a String using map_err, then use ? to propagate
    let mat = imgcodecs::imread(&path_str, imgcodecs::IMREAD_COLOR)
        .map_err(|e| format!("OpenCV failed to load the image: {}", e))?;

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
        return Err("Img is colored".to_string());
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

pub fn selective_stretch(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    min_in: u8,
    max_in: u8,
) -> Result<(), String> {
    let (_, selected_idx, mut img) = helper::get_current_image(ui_handle, images_model)?;

    if img.color {
        return Err("Img is colored".to_string());
    }

    if min_in >= max_in {
        return Err("Max is smaller equal to min".to_string());
    }

    let Some(buffer) = img.img.to_rgb8() else {
        return Err("Couldn't retrive image".to_string());
    };
    let width = buffer.width();
    let height = buffer.height();

    let mut lut = [0u8; 256];
    let min_f32 = min_in as f32;
    let max_f32 = max_in as f32;

    (0..=255).for_each(|i| {
        if i <= min_in as usize {
            lut[i] = 0;
        } else if i >= max_in as usize {
            lut[i] = 255;
        } else {
            let v = ((i as f32 - min_f32) / (max_f32 - min_f32) * 255.0).round();
            lut[i] = v.clamp(0.0, 255.0) as u8;
        }
    });

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
