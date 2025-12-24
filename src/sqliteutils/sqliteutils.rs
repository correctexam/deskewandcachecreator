use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, EncodableLayout, Pixel, RgbImage}; // Importez RgbImage si vous l'utilisez souvent
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use core::result::Result;
// Si vous utilisez encore le code de deskew précédent :
// use nalgebra::{Matrix3, Vector2, Vector3, Vector4};
// use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
// use rayon::prelude::*;
// use akaze::Akaze; // Ou votre impl custom...

// Pour la partie DB :
use webp::{Encoder, WebPImage}; // Ajout de la crate webp

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "details", rename_all = "lowercase")]
pub enum ImageState {
    Template, // Contient l'ID du template utilisé
    Align,    // Contient la matrice de transformation
    NonAlign, // Pas de données
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub pages: String,
}

const WEBP_DATA_PREFIX: &str = "data:image/webp;base64,";

/// Crée la table (structure identique)
pub fn create_connection(exam_id: u32) -> Result<Connection, rusqlite::Error> {
    let str = format!("{}.sqlite3", exam_id);
    let conn = Connection::open(str)?;
    create_table(&conn, exam_id)?;
    Ok(conn)
}

/// Crée la table (structure identique)
pub fn close_connection(conn: Connection) -> Result<(), rusqlite::Error> {
    let _ = conn.close();
    Ok(())
}

/// Crée la table (structure identique)
fn create_table(conn: &Connection, exam_id: u32) -> Result<(), rusqlite::Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS template (
            page INTEGER NOT NULL PRIMARY KEY,
            imageData CLOB NOT NULL,
            width INTEGER,
            height INTEGER            
        );",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS align (
            page INTEGER NOT NULL PRIMARY KEY,
            imageData CLOB NOT NULL,
            width INTEGER,
            height INTEGER            
        );",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nonalign (
            page INTEGER NOT NULL PRIMARY KEY,
            imageData CLOB NOT NULL,
            width INTEGER,
            height INTEGER            
        );",
        [],
    )?;
    conn.execute("CREATE TABLE IF NOT EXISTS exam(id INTEGER NOT NULL PRIMARY KEY, DT datetime default current_timestamp)",
        [],
    )?;
    let _ = conn.execute(
        "INSERT OR IGNORE INTO exam(id) VALUES (?)",
        params![exam_id],
    );
    Ok(())
}

/// Convertit l'image en octets (WebP) et l'insère en BLOB
pub fn save_image_to_db_webp(
    conn: &Connection,
    img: &DynamicImage,
    page: &i32,
    state: ImageState,
) -> Result<(), rusqlite::Error> {
    // --- 1. Préparation des données pour l'encodeur WebP ---
    // L'encodeur webp prend des données Rgba8 ou Rgb8.
    // Si votre DynamicImage n'est pas Rgb8, il faut la convertir.
    let img_rgb8 = match img {
        DynamicImage::ImageRgb8(rgb) => rgb.clone(),
        // Si votre image est Luma, RGBA, etc., convertissez-la ici
        _ => img.to_rgb8(), // Convertir vers Rgb8
    };
    let (w, h) = img_rgb8.dimensions();

    // Crée une image WebP à partir de nos Rgb8 bytes.
    // Le format webp::WebPImage attend des dimensions et des données Rgba8.
    // Si vous avez Rgb8, vous devez la transformer en Rgba8.

    let rgba_image = img.to_rgba8();
    let (width, height) = rgba_image.dimensions();
    let pixel_data = rgba_image.into_raw();

    // Encoder l'image en WebP (qualité 80)
    let encoder = Encoder::new(&pixel_data, webp::PixelLayout::Rgba, width, height);
    let webp_memory = encoder.encode(80.0);
    let webp_bytes = webp_memory.as_bytes();

    // 2. Encodage Base64
    let b64_raw = general_purpose::STANDARD.encode(&webp_bytes);

    // 3. Construction de la Data URL
    // Format : data:[<mediatype>][;base64],<data>
    let final_data_url = format!("{}{}", WEBP_DATA_PREFIX, b64_raw);

    let doc = Document {
        pages: final_data_url,
    };

    // 2. Sérialisation en JSON (Struct -> String JSON)
    match to_string_pretty(&doc) {
        Ok(json_output) => {
            println!("--- JSON Généré ---");
            if state == ImageState::Template {
                conn.execute(
                    "INSERT INTO template (page, width, height, imageData) VALUES (?1, ?2, ?3, ?4)",
                    params![page, w, h, json_output],
                )?;
            } else if state == ImageState::Align {
                conn.execute(
                    "INSERT INTO align (page, width, height, imageData) VALUES (?1, ?2, ?3, ?4)",
                    params![page, w, h, json_output],
                )?;
            } else {
                conn.execute(
                    "INSERT INTO nonalign (page, width, height, imageData) VALUES (?1, ?2, ?3, ?4)",
                    params![page, w, h, json_output],
                )?;
            }
        }
        Err(e) => eprintln!("Erreur de sérialisation : {}", e),
    }

    Ok(())
}
