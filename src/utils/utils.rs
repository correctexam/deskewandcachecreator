use image::{DynamicImage};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::{Path};
use webp::Encoder;


/// Enregistre une DynamicImage au format WebP
pub fn save_as_webp(image: &DynamicImage, output_path: &Path) -> Result<(), Box<dyn Error>> {
    // Le format Rgba8 est idéal pour WebP (prend en charge la transparence)
    let rgba_image = image.to_rgba8();

    let (width, height) = rgba_image.dimensions();
    let pixel_data = rgba_image.into_raw();

    // Encoder l'image en WebP (qualité 80)
    let encoder = Encoder::new(&pixel_data, webp::PixelLayout::Rgba, width, height);
    let webp_memory = encoder.encode(80.0);

    // Sauvegarder le fichier WebP
    File::create(output_path)?.write_all(&webp_memory)?;

    Ok(())
}

