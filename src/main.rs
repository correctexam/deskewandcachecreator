use image::DynamicImage;
use pdfium::*;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
mod akazeai;
mod akazeopenai;
mod square;
mod utils;
use crate::akazeai::akazeai::deskew_constrained;
use crate::square::square::*;
use crate::utils::utils::{save_as_webp};
// Définition de la structure pour les arguments de ligne de commande
#[derive(clap::Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Chemin vers le premier fichier PDF
    pdf_path_1: PathBuf,

    /// Chemin vers le deuxième fichier PDF
    pdf_path_2: PathBuf,
}

pub type CircleMap = HashMap<i32, Vec<Option<Circle>>>;
pub type TemplateImageMap = HashMap<i32, DynamicImage>;

fn main() -> Result<(), Box<dyn Error>> {
    let args = <Args as clap::Parser>::parse();

    // Traitement des deux fichiers PDF
    let circles = process_pdf(&args.pdf_path_1, None, -1, None);
    if circles.is_err() {
        println!("\n❌ Erreur lors du traitement du premier fichier !");
        return Err(circles.err().unwrap());
    } else {
        let templatecircle = circles.unwrap();
        process_pdf(
            &args.pdf_path_2,
            Some(templatecircle),
            -1,
            Some(&args.pdf_path_1),
            //            None
        )?;

        println!("\n✅ Processus terminé avec succès pour les deux fichiers !");
    }
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
fn process_pdf(
    pdf_path: &Path,
    template_circle: Option<CircleMap>,
    page_to_manage: i32,
    template_path: Option<&Path>,
) -> Result<CircleMap, Box<dyn Error>> {
    println!("\n--- Traitement du fichier : {} ---", pdf_path.display());

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
            template_image_map.insert(index_template as i32 + 1, template_img);
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
                let pagemod = &(page_mod(page_num, template_image_map.len()).unwrap() as i32);
                let _ = save_as_webp(&img, output_path.as_path());
                println!("    -> Sauvegardé : {}", output_path.display());

                let template_img = template_image_map.get(pagemod).unwrap();

                /*
                                // 2️⃣ Définir les coins (x, y) dans l'image
                                let roi_size = 400; // taille du carré autour de chaque coin
                                let width = gray.width() as usize;
                                let height = gray.height() as usize;

                                let corners = vec![
                                    (0, 0),                                // coin haut-gauche
                                    (width - roi_size, 0),                 // coin haut-droit
                                    (0, height - roi_size),                // coin bas-gauche
                                    (width - roi_size, height - roi_size), // coin bas-droit
                                ];

                                let transforms = akaze_ransac_per_corner(&template_gray, &gray, &corners, roi_size);
                                println!("Transformations trouvées par coin : {:?}", transforms);
                                if transforms.is_empty() {
                                    println!("Aucune transformation fiable !");
                                    let _ = save_as_webp(&img, &output_align_path.as_path());
                                    println!("    -> Sauvegardé : {}", output_align_path.display());
                                } else {
                                    // Fusionner les transformations
                                let valid: Vec<_> = transforms.into_iter().flatten().collect();
                                let global = fuse_transforms(valid).unwrap();
                                    println!("Transformation globale estimée : {:?} {:?} {:?} {:?}", global.tx, global.ty, global.scale, global.rotation);

                                    // Appliquer la transformation globale pour deskew
                                    let deskewed = apply_similarity_transform_rgba(
                                        &img.to_rgba8(),
                                        &global,
                                        template_img.width(),
                                        template_img.height(),
                                    );
                                    let dynoutput = DynamicImage::ImageRgba8(deskewed);
                */

                match deskew_constrained(&template_img,&img) {
                    Ok(res) => {
                        save_as_webp(&res, &output_align_path).unwrap();
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
                println!("    -> Sauvegardé : {}", output_path.display());

                if template_circle.is_some() {
                    let template_circle_unwrap = template_circle.as_ref().unwrap();
                    let mut src_points = [Point { x: 0.0, y: 0.0 }; 4];
                    let mut dst_points = [Point { x: 0.0, y: 0.0 }; 4];
                    let mut missing_round = 0;
                    let mut missing_round_index = 0;

                    for (i, c) in circles.iter().enumerate() {
                        let pagemod =
                            &(page_mod(page_num, template_circle_unwrap.len()).unwrap() as i32);
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
                        println!(
                            "    -> Sauvegardé (aligné avec 4 points) : {}",
                            output_align_path.display()
                        );
                    }
                } else {
                    circles_map.insert(page_num as i32, circles);
                }
            }
        }
    }

    drop(document); // Demonstrate that the page can be used after the document is dropped.

    Ok(circles_map)
}
