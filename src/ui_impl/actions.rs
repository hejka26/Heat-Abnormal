use crate::ui_impl::helper;
use opencv::{
    core::{MatTraitConst, MatTraitConstManual},
    imgcodecs,
};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel, Weak};
use std::rc::Rc;

use crate::{GrayHistogramState, ImageContainer, ImageStore, MainWindow};

pub fn open_file(ui_handle: &Weak<MainWindow>, images_model: &Rc<VecModel<ImageContainer>>) {
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

pub fn calculate_gray_histogram(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) {
    let Some(ui) = ui_handle.upgrade() else {
        return;
    };

    let Some(img) = images_model.row_data(ui.global::<ImageStore>().get_selected_image() as usize)
    else {
        eprintln!("Couldn't retrieve img");
        return;
    };

    if img.color {
        eprintln!("Img is colored");
        return;
    }
    let mat = match helper::slint_to_gray(&img.img) {
        Ok(mat) => mat,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    let mut max_value = 0;
    let mut data = vec![0.0f32; 256];
    let pixels = match mat.data_typed::<u8>() {
        Ok(pixels) => pixels,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    for pixel in pixels.iter() {
        data[*pixel as usize] += 1.0;
    }
    let max_value = data.iter().cloned().fold(0.0f32, f32::max);
    let total_pixels = data.iter().cloned().sum::<f32>();

    // --- NOWY KOD: OBLICZANIE ŚREDNIEJ I ODCHYLENIA ---

    // 1. Suma wszystkich pikseli (czyli N)
    let total_pixels: f32 = data.iter().sum();

    // 2. Obliczanie średniej (Mean)
    let mean: f32 = if total_pixels > 0.0 {
        data.iter()
            .enumerate()
            .map(|(intensity, &count)| (intensity as f32) * count)
            .sum::<f32>()
            / total_pixels
    } else {
        0.0
    };

    // 3. Obliczanie wariancji i odchylenia standardowego (Std Dev)
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
}

pub fn equalize_histogram(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
) {
    let Some(ui) = ui_handle.upgrade() else {
        return;
    };

    let selected_idx = ui.global::<ImageStore>().get_selected_image() as usize;

    let Some(mut img) = images_model.row_data(selected_idx) else {
        eprintln!("Nie udało się pobrać obrazu z modelu.");
        return;
    };

    // Equalizacja klasyczna działa najlepiej na obrazach w skali szarości.
    if img.color {
        eprintln!(
            "Obraz jest kolorowy. Najpierw przekonwertuj obraz na odcienie szarości (RGB2Gray)."
        );
        return;
    }

    // 1. Pobieramy bezpośrednio bufor pikseli ze Slinta
    let Some(buffer) = img.img.to_rgb8() else {
        eprintln!("Nie udało się odczytać pikseli obrazu.");
        return;
    };

    let width = buffer.width();
    let height = buffer.height();
    let total_pixels = (width * height) as usize;

    // 2. Krok 1: Obliczanie histogramu
    // Skoro to skala szarości, R == G == B, więc sprawdzamy tylko kanał R
    let mut hist = [0u32; 256];
    for pixel in buffer.as_slice() {
        hist[pixel.r as usize] += 1;
    }

    // 3. Krok 2: Obliczanie dystrybuanty (CDF)
    let mut cdf = [0u32; 256];
    let mut sum = 0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum;
    }

    // Znajdujemy minimalną niezerową wartość w CDF
    let cdf_min = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0) as f32;
    let total_f32 = total_pixels as f32;

    // 4. Krok 3: Tworzenie tablicy przekodowań (LUT - Look-Up Table) ze znormalizowanymi wartościami
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = ((cdf[i] as f32 - cdf_min) / (total_f32 - cdf_min) * 255.0).round();
        // Zabezpieczenie przed wyjściem poza zakres u8 (0-255)
        lut[i] = v.clamp(0.0, 255.0) as u8;
    }

    // 5. Krok 4: Aplikowanie LUT i tworzenie nowego obrazka dla Slinta
    let mut new_buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::new(width, height);

    // Iterujemy po starym buforze i zapisujemy do nowego korzystając z tablicy LUT
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

    // 6. Zapisanie nowego obrazu z powrotem do modelu Slinta
    img.img = slint::Image::from_rgb8(new_buffer);
    images_model.set_row_data(selected_idx, img);

    calculate_gray_histogram(ui_handle, images_model);
}

pub fn selective_stretch(
    ui_handle: &Weak<MainWindow>,
    images_model: &Rc<VecModel<ImageContainer>>,
    min_in: u8,
    max_in: u8,
) {
    let Some(ui) = ui_handle.upgrade() else {
        return;
    };

    let selected_idx = ui.global::<ImageStore>().get_selected_image() as usize;

    let Some(mut img) = images_model.row_data(selected_idx) else {
        return;
    };

    if img.color {
        eprintln!("Obraz jest kolorowy. Najpierw przekonwertuj na odcienie szarości.");
        return;
    }

    if min_in >= max_in {
        eprintln!("Błędne parametry: min_in musi być mniejsze od max_in.");
        return;
    }

    let Some(buffer) = img.img.to_rgb8() else {
        return;
    };
    let width = buffer.width();
    let height = buffer.height();

    // Tworzenie tablicy LUT dla zadanego okna [min_in, max_in]
    let mut lut = [0u8; 256];
    let min_f32 = min_in as f32;
    let max_f32 = max_in as f32;

    for i in 0..=255 {
        if i <= min_in as usize {
            lut[i] = 0; // Ucinanie ciemnych pikseli
        } else if i >= max_in as usize {
            lut[i] = 255; // Ucinanie jasnych pikseli
        } else {
            // Liniowe rozciągnięcie tylko środkowego fragmentu
            let v = ((i as f32 - min_f32) / (max_f32 - min_f32) * 255.0).round();
            lut[i] = v.clamp(0.0, 255.0) as u8;
        }
    }

    // Aplikowanie LUT
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

    calculate_gray_histogram(ui_handle, images_model);
}
