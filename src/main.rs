use image::DynamicImage;
use pdfium::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
mod akazeai;
mod akazeopenai;
mod sqliteutils;
mod square;
mod utils;
use crate::akazeai::akazeai::deskew_constrained;
use crate::sqliteutils::sqliteutils::*;
use crate::square::square::*;
use crate::utils::utils::save_as_webp;
// Définition de la structure pour les arguments de ligne de commande
/* #[derive(clap::Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Chemin vers le premier fichier PDF
    pdf_path_1: PathBuf,
    /// Chemin vers le deuxième fichier PDF
    pdf_path_2: PathBuf,
}*/

/// Simple program to greet a person
#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Pages To Process
    #[arg(short, long)]
    pages_to_manage: String,

    /// Exam Id
    #[arg(short, long, default_value_t = 1)]
    exam_id: u32,

    /// Template Id
    #[arg(short, long, default_value_t = 1)]
    template_id: u32,

    /// Scan Id
    #[arg(short, long, default_value_t = 1)]
    scan_id: u32,

    /// Align Algo
    #[arg(short, long, default_value_t = 1)]
    algo: u8, // 0=NoAlign,1=CircleAlign,2=Akaze

    /// Debug
    #[arg(long, short, action)]
    debug: bool, //allow debug
}

pub type CircleMap = HashMap<u32, Vec<Option<Circle>>>;
pub type TemplateImageMap = HashMap<u32, DynamicImage>;

fn main() -> Result<(), Box<dyn Error>> {
    let args = <Args as clap::Parser>::parse();
    if args.algo > 2 {
        println!("\n❌ Algo d'alignement invalide !");
        return Err("Algo d'alignement invalide".into());
    }
    println!("\n--- Démarrage du processus d'alignement d'images PDF ---");
    let align_algo = match args.algo {
        0 => AlignAlgo::NoAlign,
        1 => AlignAlgo::CircleAlign,
        2 => AlignAlgo::Akaze,
        _ => AlignAlgo::NoAlign,
    };
    let pages_to_manage: Vec<u32> = if args.pages_to_manage.trim().is_empty() {
        vec![]
    } else {
        args.pages_to_manage
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    };

    let _ = process_exam(
        pages_to_manage,
        args.exam_id as u32,
        args.template_id as u32,
        args.scan_id as u32,
        align_algo,
        args.debug,
    )?;
    Ok(())
}
fn page_mod(n: usize, m: usize) -> Option<usize> {
    if m == 0 {
        None
    } else {
        Some(((n - 1) % m) + 1)
    }
}

/// Fonction principale pour traiter un seul fichier PDF
/* fn process_pdf(
    pdf_path: &Path,
    template_circle: Option<CircleMap>,
    page_to_manage: i32,
    template_path: Option<&Path>,
    exam_id: u32,
) -> Result<CircleMap, Box<dyn Error>> {
    println!("\n--- Traitement du fichier : {} ---", pdf_path.display());
    let conn = create_connection(exam_id)?;
    let mut circles_map: CircleMap = HashMap::new();
    let mut template_image_map: TemplateImageMap = HashMap::new();

    // 1. Définir les chemins de sortie et le nom de base
    let file_stem = pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

    let output_dir = pdf_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}_webp_images", file_stem));
    std::fs::create_dir_all(&output_dir)?;

    // Créer le répertoire de sortie s'il n'existe pas
    //    std_extra::fs::create_dir_if_not_exists(&output_dir)?;
    println!("Dossier de sortie créé : {}", output_dir.display());

    let document = PdfiumDocument::new_from_path(pdf_path, None).unwrap();
    // let page = document.page(0).unwrap();
    let config = PdfiumRenderConfig::new().with_height(2000);
    // 2. Charger le document PDF
    //    let document = pdfium.load_pdf_from_file(pdf_path, None)?;
    let page_count = document.pages().page_count();
    println!("{} pages trouvées.", page_count);

    if template_path.is_some() {
        let template_document =
            PdfiumDocument::new_from_path(template_path.unwrap(), None).unwrap();
        // 2. Charger le document PDF

        //            let page_count = document.pages().page_count();
        //            println!("{} pages trouvées.", page_count);
        for (index_template, template_page) in template_document.pages().enumerate() {
            let page_num = index_template + 1;
            let bitmap_template = template_page.unwrap().render(&config).unwrap();
            let template_img = match bitmap_template.as_rgb8_image() {
                Ok(template_img) => template_img,
                Err(_) => {
                    println!(
                        "    ! Avertissement : Impossible de convertir la page {} en image RGB8.",
                        page_num
                    );
                    continue;
                }
            };
            template_image_map.insert(index_template as u32 + 1, template_img);
        }
    }

    // 4. Parcourir et traiter chaque page
    for (index, page) in document.pages().enumerate() {
        let page_num = index + 1;

        if (page_to_manage == (-1 as i32)) || ((page_num as i32) == page_to_manage) {
            let output_filename = format!("{}_page_{}.webp", file_stem, page_num);
            let output_path = output_dir.join(output_filename);
            let output_align_filename = format!("{}_align_page_{}.webp", file_stem, page_num);
            let output_align_path = output_dir.join(output_align_filename);

            println!(
                "  - Rendu et conversion de la page {}/{}...",
                page_num, page_count
            );

            let bitmap = page.unwrap().render(&config).unwrap();

            // --- Remplacement ici : obtenir l'image en owned DynamicImage (mutable)
            let mut img = match bitmap.as_rgb8_image() {
                Ok(img) => img,
                Err(_) => {
                    println!(
                        "    ! Avertissement : Impossible de convertir la page {} en image RGB8.",
                        page_num
                    );
                    continue;
                }
            };

            //let circles = detect_black_circles_from_image(&mut img, 100);
            let gray = img.to_luma8();

            if template_image_map.len() > 0 {
                let pagemod = &(page_mod(page_num, template_image_map.len()).unwrap() as u32);
                let _ = save_as_webp(&img, output_path.as_path());
                println!("    -> Sauvegardé : {}", output_path.display());

                let template_img = template_image_map.get(pagemod).unwrap();


                match deskew_constrained(&template_img, &img) {
                    Ok(res) => {
                        save_as_webp(&res, &output_align_path).unwrap();
                        let _ = save_image_to_db_webp(
                            &conn,
                            &res,
                            &(page_num as i32),
                            ImageState::Align,
                        );

                        println!(
                            "    -> Sauvegardé (aligné avec n points) : {}",
                            output_align_path.display()
                        );
                        println!("Terminé ! Image sauvegardée.");
                    }
                    Err(e) => eprintln!("Erreur: {}", e),
                }

                println!("    -> Sauvegardé : {}", output_align_path.display());
            } else {
                let circles = detect_circles_in_four_corners_advanced(&gray, 300);
                // Dessin des cercles en bleu
                draw_circles_blue(&mut img, &circles, 2);

                let _ = save_as_webp(&img, output_path.as_path());
                let _ =
                    save_image_to_db_webp(&conn, &img, &(page_num as i32), ImageState::NonAlign);
                println!("    -> Sauvegardé : {}", output_path.display());

                if template_circle.is_some() {
                    let template_circle_unwrap = template_circle.as_ref().unwrap();
                    let mut src_points = [Point { x: 0.0, y: 0.0 }; 4];
                    let mut dst_points = [Point { x: 0.0, y: 0.0 }; 4];
                    let mut missing_round = 0;
                    let mut missing_round_index = 0;

                    for (i, c) in circles.iter().enumerate() {
                        let pagemod =
                            &(page_mod(page_num, template_circle_unwrap.len()).unwrap() as u32);
                        if c.is_none() {
                            missing_round = missing_round + 1;
                            missing_round_index = i;
                        } else if c.is_some() && template_circle_unwrap.get(pagemod).is_some() {
                            let template_circles = template_circle_unwrap.get(pagemod).unwrap();
                            if template_circles[i].is_some() {
                                let tc = template_circles[i].as_ref().unwrap();
                                let c_unwrap = c.as_ref().unwrap();
                                src_points[i] = Point {
                                    x: c_unwrap.cx,
                                    y: c_unwrap.cy,
                                };
                                dst_points[i] = Point { x: tc.cx, y: tc.cy };
                            }
                        }
                    }
                    if missing_round > 1 {
                        println!(
                            "    ! Avertissement : Impossible d'aligner la page {}",
                            page_num
                        );
                    } else if missing_round == 1 {
                        // Utilisation de 3 points pour l'estimation
                        let mut src_points_3 = [Point { x: 0.0, y: 0.0 }; 3];
                        let mut dst_points_3 = [Point { x: 0.0, y: 0.0 }; 3];
                        let mut idx = 0;
                        for i in 0..4 {
                            if i != missing_round_index {
                                src_points_3[idx] = src_points[i];
                                dst_points_3[idx] = dst_points[i];
                                idx += 1;
                            }
                        }
                        let transform = estimate_similarity_3pts(&src_points_3, &dst_points_3);
                        let aligned_image =
                            apply_similarity(&img, &transform, img.width(), img.height());
                        let dynoutput = DynamicImage::ImageRgba8(aligned_image);
                        save_as_webp(&dynoutput, &output_align_path).unwrap();
                        let _ = save_image_to_db_webp(
                            &conn,
                            &dynoutput,
                            &(page_num as i32),
                            ImageState::Align,
                        );

                        println!(
                            "    -> Sauvegardé (aligné avec 3 points) : {}",
                            output_align_path.display()
                        );
                    } else {
                        let transform = estimate_similarity(&src_points, &dst_points);
                        let aligned_image =
                            apply_similarity(&img, &transform, img.width(), img.height());
                        let dynoutput = DynamicImage::ImageRgba8(aligned_image);
                        save_as_webp(&dynoutput, &output_align_path).unwrap();
                        let _ = save_image_to_db_webp(
                            &conn,
                            &dynoutput,
                            &(page_num as i32),
                            ImageState::Align,
                        );
                        println!(
                            "    -> Sauvegardé (aligné avec 4 points) : {}",
                            output_align_path.display()
                        );
                    }
                } else {
                    circles_map.insert(page_num as u32, circles);
                }
            }
        }
    }

    drop(document); // Demonstrate that the page can be used after the document is dropped.
    close_connection(conn);
    Ok(circles_map)
} */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignAlgo {
    NoAlign,     // Pas d'alignement
    CircleAlign, // Contient des marques
    Akaze,       // Pass de marques, utilise AKAZE
}

/// Fonction principale pour traiter un seul fichier PDF
fn process_exam(
    pages_to_manage: Vec<u32>,
    exam_id: u32,
    template_id: u32,
    scan_id: u32,
    align_algo: AlignAlgo,
    debug: bool,
) -> Result<(), Box<dyn Error>> {
    println!("\n--- Traitement de l'exam : {} ---", exam_id);
    let conn = create_connection(exam_id)?;
    //        pdf_path: &Path,
    let mut template_circle_map: CircleMap = HashMap::new();
    //    let mut circles_map: CircleMap = HashMap::new();
    let mut template_image_map: TemplateImageMap = HashMap::new();

    let templatepath = &format!("template-{}.pdf", template_id);
    let scanpath = &format!("scan-{}.pdf", scan_id);
    let pdf_template_path = Path::new(&templatepath);
    let pdf_scan_path = Path::new(scanpath);
    let document_template = PdfiumDocument::new_from_path(pdf_template_path, None).unwrap();
    let document_scan = PdfiumDocument::new_from_path(pdf_scan_path, None).unwrap();
    let config = PdfiumRenderConfig::new().with_height(2000);

    // 2. Charger le document PDF
    //    let document = pdfium.load_pdf_from_file(pdf_path, None)?;
    let page_count_template = document_template.pages().page_count();
    println!("{} pages template trouvées.", page_count_template);
    let page_count_scan = document_template.pages().page_count();
    println!("{} pages scan trouvées.", page_count_scan);

    for (index_template, template_page) in document_template.pages().enumerate() {
        let page_num = index_template + 1;
        let bitmap_template = template_page.unwrap().render(&config).unwrap();
        let template_img = match bitmap_template.as_rgb8_image() {
            Ok(template_img) => template_img,
            Err(_) => {
                println!(
                    "    ! Avertissement : Impossible de convertir la page {} en image RGB8.",
                    page_num
                );
                continue;
            }
        };

        template_image_map.insert(page_num as u32, template_img);
        let _ = save_image_to_db_webp(
            &conn,
            template_image_map.get(&(page_num as u32)).unwrap(),
            &(page_num as i32),
            ImageState::Template,
        );
        println!("    -> Sauvegardé : {}", page_num);

        let tgray = template_image_map
            .get(&(page_num as u32))
            .unwrap()
            .to_luma8();

        if AlignAlgo::CircleAlign == align_algo {
            let tcircles = detect_circles_in_four_corners_advanced(&tgray, 300);
            template_circle_map.insert(page_num as u32, tcircles.clone());
        }
        // Dessin des cercles en bleu
        if debug {
            let mut template_imgd = match bitmap_template.as_rgb8_image() {
                Ok(template_imgd) => template_imgd,
                Err(_) => {
                    println!(
                        "    ! Avertissement : Impossible de convertir la page {} en image RGB8.",
                        page_num
                    );
                    continue;
                }
            };
            let tcircles = template_circle_map.get(&(page_num as u32));
            if (tcircles.is_some()) {
                draw_circles_blue(&mut template_imgd, &tcircles.unwrap(), 2);
            }
            let file_stem = pdf_template_path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

            let output_dir = pdf_template_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(format!("{}_webp_images", file_stem));
            std::fs::create_dir_all(&output_dir)?;
            let output_filename = format!("{}_page_{}.webp", file_stem, page_num);
            let output_path = output_dir.join(output_filename);
            let _ = save_as_webp(&template_imgd, output_path.as_path());
        }
    }

    // 4. Parcourir et traiter chaque page
    for (index, page) in document_scan.pages().enumerate() {
        let page_num = index + 1;

        if pages_to_manage.len() == 0 || pages_to_manage.contains(&(page_num as u32)) {
            println!(
                "  - Rendu et conversion de la page {}/{}...",
                page_num, page_count_template
            );
            let bitmap = page.unwrap().render(&config).unwrap();
            // --- Remplacement ici : obtenir l'image en owned DynamicImage (mutable)
            let mut img = match bitmap.as_rgb8_image() {
                Ok(img) => img,
                Err(_) => {
                    println!(
                        "    ! Avertissement : Impossible de convertir la page {} en image RGB8.",
                        page_num
                    );
                    continue;
                }
            };

            let _ = save_image_to_db_webp(&conn, &img, &(page_num as i32), ImageState::NonAlign);
            println!("    -> Sauvegardé : {}", page_num);

            if AlignAlgo::Akaze == align_algo {
                let pagemod = &(page_mod(page_num, template_image_map.len()).unwrap() as u32);
                let template_img = template_image_map.get(pagemod).unwrap();
                match deskew_constrained(&template_img, &img) {
                    Ok(res) => {
                        let _ = save_image_to_db_webp(
                            &conn,
                            &res,
                            &(page_num as i32),
                            ImageState::Align,
                        );

                        println!("    -> Sauvegardé (aligné avec n points) : {}", page_num);

                        if debug {
                            let imgd = match bitmap.as_rgb8_image() {
                                Ok(imgd) => imgd,
                                Err(_) => {
                                    println!(
                                        "    ! Avertissement : Impossible de convertir la page {} en image RGB8.",
                                        page_num
                                    );
                                    continue;
                                }
                            };
                            let file_stem =
                                pdf_scan_path.file_stem().and_then(|s| s.to_str()).ok_or(
                                    "Le chemin du fichier n'a pas de nom de base valide (stem).",
                                )?;

                            let output_dir = pdf_scan_path
                                .parent()
                                .unwrap_or_else(|| Path::new("."))
                                .join(format!("{}_webp_images", file_stem));
                            std::fs::create_dir_all(&output_dir)?;
                            let output_filename = format!("{}_page_{}.webp", file_stem, page_num);
                            let output_aligned_filename =
                                format!("{}_page_aligned_{}.webp", file_stem, page_num);
                            let output_path = output_dir.join(output_filename);
                            let output_aligned_path = output_dir.join(output_aligned_filename);
                            let _ = save_as_webp(&imgd, output_path.as_path());
                            let _ = save_as_webp(&res, output_aligned_path.as_path());
                        }
                    }
                    Err(e) => eprintln!("Erreur: {}", e),
                }
            } else if AlignAlgo::CircleAlign == align_algo {
                let gray = img.to_luma8();
                let circles = detect_circles_in_four_corners_advanced(&gray, 300);
                // Dessin des cercles en bleu
                if debug {
                    let file_stem = pdf_template_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

                    let output_dir = pdf_template_path
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join(format!("{}_webp_images", file_stem));
                    std::fs::create_dir_all(&output_dir)?;
                    let output_filename = format!("{}_page_{}.webp", file_stem, page_num);
                    let output_path = output_dir.join(output_filename);
                    draw_circles_blue(&mut img, &circles, 2);
                    let _ = save_as_webp(&img, output_path.as_path());
                }

                let mut src_points = [Point { x: 0.0, y: 0.0 }; 4];
                let mut dst_points = [Point { x: 0.0, y: 0.0 }; 4];
                let mut missing_round = 0;
                let mut missing_round_index = 0;
                for (i, c) in circles.iter().enumerate() {
                    let pagemod = &(page_mod(page_num, template_circle_map.len()).unwrap() as u32);
                    if c.is_none() {
                        missing_round = missing_round + 1;
                        missing_round_index = i;
                    } else if c.is_some() && template_circle_map.get(pagemod).is_some() {
                        let template_circles = template_circle_map.get(pagemod).unwrap();
                        if template_circles[i].is_some() {
                            let tc = template_circles[i].as_ref().unwrap();
                            let c_unwrap = c.as_ref().unwrap();
                            src_points[i] = Point {
                                x: c_unwrap.cx,
                                y: c_unwrap.cy,
                            };
                            dst_points[i] = Point { x: tc.cx, y: tc.cy };
                        }
                    }
                }
                if missing_round > 1 {
                    println!(
                        "    ! Avertissement : Impossible d'aligner la page retour AKAZE {}",
                        page_num
                    );
                    let pagemod = &(page_mod(page_num, template_image_map.len()).unwrap() as u32);
                    let template_img = template_image_map.get(pagemod).unwrap();
                    match deskew_constrained(&template_img, &img) {
                        Ok(res) => {
                            let _ = save_image_to_db_webp(
                                &conn,
                                &res,
                                &(page_num as i32),
                                ImageState::Align,
                            );
                            println!("    -> Sauvegardé (aligné avec n points) : {}", page_num);
                            if debug {
                                let file_stem = pdf_scan_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

                                let output_dir = pdf_scan_path
                                    .parent()
                                    .unwrap_or_else(|| Path::new("."))
                                    .join(format!("{}_webp_images", file_stem));
                                std::fs::create_dir_all(&output_dir)?;
                                let output_aligned_filename =
                                    format!("{}_page_aligned_{}.webp", file_stem, page_num);
                                let output_aligned_path = output_dir.join(output_aligned_filename);
                                let _ = save_as_webp(&res, output_aligned_path.as_path());
                            }
                        }
                        Err(e) => eprintln!("Erreur: {}", e),
                    }
                } else if missing_round == 1 {
                    // Utilisation de 3 points pour l'estimation
                    let mut src_points_3 = [Point { x: 0.0, y: 0.0 }; 3];
                    let mut dst_points_3 = [Point { x: 0.0, y: 0.0 }; 3];
                    let mut idx = 0;
                    for i in 0..4 {
                        if i != missing_round_index {
                            src_points_3[idx] = src_points[i];
                            dst_points_3[idx] = dst_points[i];
                            idx += 1;
                        }
                    }
                    let transform = estimate_similarity_3pts(&src_points_3, &dst_points_3);
                    let aligned_image =
                        apply_similarity(&img, &transform, img.width(), img.height());
                    let dynoutput = DynamicImage::ImageRgba8(aligned_image);
                    let _ = save_image_to_db_webp(
                        &conn,
                        &dynoutput,
                        &(page_num as i32),
                        ImageState::Align,
                    );
                    println!("    -> Sauvegardé (aligné avec 3 points) : {}", page_num);
                    if debug {
                        let file_stem = pdf_scan_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

                        let output_dir = pdf_scan_path
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join(format!("{}_webp_images", file_stem));
                        std::fs::create_dir_all(&output_dir)?;
                        let output_aligned_filename =
                            format!("{}_page_aligned_{}.webp", file_stem, page_num);
                        let output_aligned_path = output_dir.join(output_aligned_filename);
                        let _ = save_as_webp(&dynoutput, output_aligned_path.as_path());
                    }
                } else {
                    let transform = estimate_similarity(&src_points, &dst_points);
                    let aligned_image =
                        apply_similarity(&img, &transform, img.width(), img.height());
                    let dynoutput = DynamicImage::ImageRgba8(aligned_image);
                    let _ = save_image_to_db_webp(
                        &conn,
                        &dynoutput,
                        &(page_num as i32),
                        ImageState::Align,
                    );
                    println!("    -> Sauvegardé (aligné avec 4 points) : {}", page_num);
                                        if debug {
                        let file_stem = pdf_scan_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

                        let output_dir = pdf_scan_path
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join(format!("{}_webp_images", file_stem));
                        std::fs::create_dir_all(&output_dir)?;
                        let output_aligned_filename =
                            format!("{}_page_aligned_{}.webp", file_stem, page_num);
                        let output_aligned_path = output_dir.join(output_aligned_filename);
                        let _ = save_as_webp(&dynoutput, output_aligned_path.as_path());
                    }


                }
            } else {
                let _ = save_image_to_db_webp(&conn, &img, &(page_num as i32), ImageState::Align);
                    if debug {
                        let file_stem = pdf_scan_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .ok_or("Le chemin du fichier n'a pas de nom de base valide (stem).")?;

                        let output_dir = pdf_scan_path
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join(format!("{}_webp_images", file_stem));
                        std::fs::create_dir_all(&output_dir)?;
                        let output_aligned_filename =
                            format!("{}_page_aligned_{}.webp", file_stem, page_num);
                        let output_aligned_path = output_dir.join(output_aligned_filename);
                        let _ = save_as_webp(&img, output_aligned_path.as_path());
                    }
            }
        }
    }

    drop(document_scan); // Demonstrate that the page can be used after the document is dropped.
    drop(document_template); // Demonstrate that the page can be used after the document is dropped.
    close_connection(conn);
    Ok(())
}
