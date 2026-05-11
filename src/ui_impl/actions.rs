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

    // Na ten moment wypisujemy w konsoli, dopóki nie dodasz tego do UI

    // --- KONIEC NOWEGO KODU ---

    let hist_ui = ui.global::<GrayHistogramState>();
    hist_ui.set_data(ModelRc::from(Rc::new(VecModel::from(data))));
    hist_ui.set_max_value(max_value as f32);
    hist_ui.set_mean(mean);
    hist_ui.set_std_dev(std_dev);
}
