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

/// Transforme "1-3; 5" en vecteur [1, 2, 3, 5]
/// Retourne une erreur si le format est invalide.
pub fn parse_page_selection(input: &str) -> Result<Vec<u32>, String> {
    let mut pages = Vec::new();

    // 1. Découpage par point-virgule
    for part in input.split(';') {
        let part = part.trim();
        
        // Ignorer les segments vides (ex: "1;2;;3")
        if part.is_empty() {
            continue;
        }

        // 2. Vérification si c'est une plage (Range)
        if let Some((start_str, end_str)) = part.split_once('-') {
            let start = parse_u32(start_str)?;
            let end = parse_u32(end_str)?;

            if start > end {
                return Err(format!("Plage invalide : début ({}) > fin ({})", start, end));
            }

            // Ajout de tous les nombres de la plage (inclusif)
            for p in start..=end {
                pages.push(p);
            }
        } else {
            // 3. C'est un nombre simple
            let p = parse_u32(part)?;
            pages.push(p);
        }
    }

    // 4. (Optionnel) Tri et suppression des doublons
    // Utile car "2-5;4" donnerait [2,3,4,5,4] sinon
    pages.sort_unstable(); // Plus rapide que sort()
    pages.dedup();

    Ok(pages)
}

/// Helper pour parser et convertir l'erreur native en String
fn parse_u32(s: &str) -> Result<u32, String> {
    s.trim().parse::<u32>().map_err(|_| format!("'{}' n'est pas un nombre valide", s))
}

