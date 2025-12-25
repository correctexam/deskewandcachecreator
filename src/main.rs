use image::DynamicImage;
use pdfium::*;
use serde::{Deserialize, Serialize};
use urlencoding::encode;
use std::{collections::HashMap};
use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use futures_lite::stream::StreamExt;
use lapin::{
    BasicProperties, Channel, Connection, ConnectionProperties, ExchangeKind, auth::Credentials, options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, ExchangeDeclareOptions, QueueDeclareOptions}, types::FieldTable
};
mod akazeai;
mod s3fs;
mod sqliteutils;
mod square;
mod utils;
use crate::akazeai::akazeai::deskew_constrained;
use crate::s3fs::s3fs::{get_object, put_object};
use crate::sqliteutils::sqliteutils::*;
use crate::square::square::*;
use crate::utils::utils::{parse_page_selection, save_as_webp};
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
    #[arg(long, short, action, default_value_t = false)]
    debug: bool, //allow debug

    #[arg(long, short, action, default_value_t = false)]
    mq: bool, //allow debug

    #[arg(long, short, default_value = "localhost")]
    mq_server_host: String,

    #[arg(long, short, default_value = "5672")]
    mq_server_port: u32,


    #[arg(long, short, default_value = "rabbitmq")]
    mq_server_login: String,
    #[arg(long, short, default_value = "rabbitmq")]
    mq_server_pass: String,
    #[arg(long, short, default_value = "scan_jobs_queue")]
    mq_server_inputqueue: String,
    #[arg(long, short, default_value = "scan_status_exchange")]
    mq_server_ouputqueue: String,

    #[arg(long, short, default_value = "http://localhost:9000")]
    server_url: String,
    #[arg(long, short, default_value = "test")]
    bucket_name: String,
    #[arg(long, short, default_value = "admin")]
    login: String,
    #[arg(long, short, default_value = "minioadmin")]
    pass: String,
}

// 1. Le message reçu (Ordre de travail)
#[derive(Debug, Deserialize)]
struct ScanRequest {
    pages_to_manage: String,
    #[serde(default = "default_u32_one")]
    exam_id: u32,
    #[serde(default = "default_u32_one")]
    template_id: u32,
    #[serde(default = "default_u32_one")]
    scan_id: u32,
    #[serde(default = "default_algo")]
    algo: u8,
}

fn default_u32_one() -> u32 {
    1
}
fn default_algo() -> u8 {
    1
}

// 2. Le message envoyé (État d'avancement)
#[derive(Debug, Serialize)]
struct ScanStatus {
    scan_id: u32,
    page: u32,
    status: String, // "processing", "success", "error"
    progress: u32,  // Pourcentage global
    details: String,
}

pub type CircleMap = HashMap<u32, Vec<Option<Circle>>>;
pub type TemplateImageMap = HashMap<u32, DynamicImage>;

#[tokio::main]
// async fn main() -> Result<()> {
async fn main() -> Result<(), Box<dyn Error>> {
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
    let pages_to_manage: Vec<u32> = if args.pages_to_manage.trim() == "all" {
        vec![]
    } else {
        match parse_page_selection(&args.pages_to_manage) {
            Ok(pages) => pages,
            Err(_) => vec![],
        }
    };

    if args.mq {
        println!("🔌 Connexion à RabbitMQ...");

        let connectionp: ConnectionProperties = ConnectionProperties::default();
        let encoded_pass = encode(&args.mq_server_pass);
        let addr = format!("amqp://{}:{}@{}:{}/%2f", args.mq_server_login, &encoded_pass, args.mq_server_host, args.mq_server_port);

//        connectionp.set_credentials(credentials);
        let conn = Connection::connect(&addr, connectionp).await?;
        let channel = conn.create_channel().await?;

        // 2. Déclaration de la topologie (Idempotent)

        // A. Queue d'entrée (Durable)
        let _queue = channel
            .queue_declare(
                &args.mq_server_inputqueue,
                QueueDeclareOptions {
                    durable: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await?;

        // B. Exchange de sortie (Topic)
        channel
            .exchange_declare(
                &args.mq_server_ouputqueue,
                ExchangeKind::Topic,
                ExchangeDeclareOptions {
                    durable: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await?;

        println!(
            "✅ Worker prêt. En attente de jobs dans '{}'...",
            &args.mq_server_inputqueue
        );

        // 3. Création du Consumer
        let mut consumer = channel
            .basic_consume(
                &args.mq_server_inputqueue,
                "scan_worker_tag", // Nom unique du consommateur
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await?;

        // 4. Boucle de traitement des messages
        while let Some(delivery) = consumer.next().await {
            if let Ok(delivery) = delivery {
                println!("📥 Message reçu !");
                // Traitement (Payload -> JSON -> Logique)
                match process_message(&channel, &delivery.data,
                            args.debug,
            args.server_url.as_str(),
            args.bucket_name.as_str(),
            args.login.as_str(),
            args.pass.as_str(),
            &args.mq_server_ouputqueue,

                 ).await {
                    Ok(_) => {
                        // Succès : On acquitte le message (ACK)
                        delivery.ack(BasicAckOptions::default()).await?;
                        println!("✅ Job terminé et acquitté.");
                    }
                    Err(e) => {
                        eprintln!("❌ Erreur traitement : {}", e);
                        // En cas d'erreur fatale, on peut soit NACK (requeue) soit ACK (pour jeter le message corrompu)
                        // Ici on ACK pour éviter une boucle infinie si le JSON est invalide
                        delivery.ack(BasicAckOptions::default()).await?;
                    }
                }
            }
        }
    } else {
        let _ = process_exam(
            pages_to_manage,
            args.exam_id as u32,
            args.template_id as u32,
            args.scan_id as u32,
            align_algo,
            args.debug,
            args.server_url.as_str(),
            args.bucket_name.as_str(),
            args.login.as_str(),
            args.pass.as_str(),
            None, 
            &args.mq_server_ouputqueue,
        ).await?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignAlgo {
    NoAlign,     // Pas d'alignement
    CircleAlign, // Contient des marques
    Akaze,       // Pass de marques, utilise AKAZE
}

/// Fonction principale pour traiter un seul fichier PDF
 async fn process_exam(
    pages_to_manage: Vec<u32>,
    exam_id: u32,
    template_id: u32,
    scan_id: u32,
    align_algo: AlignAlgo,
    debug: bool,
    server_url: &str,
    bucket_name: &str,
    login: &str,
    pass: &str,
    chan : Option<&Channel>,
    outputchan : &String,

) -> Result<(), Box<dyn Error>> {
    println!("\n--- Traitement de l'exam : {} ---", exam_id);
    if pages_to_manage.len() > 0 {
        let existingcache = get_object(
            server_url,
            bucket_name,
            &format!("cache/{}.sqlite3", exam_id),
            login,
            pass,
        )?;
        fs::write(format!("{}.sqlite3", exam_id), existingcache.bytes())?;
    }

    let conn = create_connection(exam_id)?;
    //        pdf_path: &Path,
    let mut template_circle_map: CircleMap = HashMap::new();
    //    let mut circles_map: CircleMap = HashMap::new();
    let mut template_image_map: TemplateImageMap = HashMap::new();

    let templatepath = &format!("template-{}.pdf", template_id);
    let scanpath = &format!("scan-{}.pdf", scan_id);
    let pdf_template_path = Path::new(&templatepath);
    let pdf_scan_path = Path::new(scanpath);
    let template = get_object(
        server_url,
        bucket_name,
        &format!("template/{}.pdf", template_id),
        login,
        pass,
    )?;
    let templatedata = template.bytes().to_vec();
    let cursor = Cursor::new(templatedata);
    let document_template = PdfiumDocument::new_from_reader(cursor, None).unwrap();
    let scan = get_object(
        server_url,
        bucket_name,
        &format!("scan/{}.pdf", scan_id),
        login,
        pass,
    )?;
    let scan_data = scan.bytes().to_vec();
    let cursorscan = Cursor::new(scan_data);
    let document_scan = PdfiumDocument::new_from_reader(cursorscan, None).unwrap();

    //    let document_scan = PdfiumDocument::new_from_path(pdf_scan_path, None).unwrap();
    let config = PdfiumRenderConfig::new().with_height(2000);

    // 2. Charger le document PDF
    //    let document = pdfium.load_pdf_from_file(pdf_path, None)?;
    let page_count_template = document_template.pages().page_count();
    println!("{} pages template trouvées.", page_count_template);
    let page_count_scan = document_scan.pages().page_count();
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
            if tcircles.is_some() {
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
    let mut realpage = 0;
    // 4. Parcourir et traiter chaque page
    for (index, page) in document_scan.pages().enumerate() {
        let page_num = index + 1;

        if pages_to_manage.len() == 0 || pages_to_manage.contains(&(page_num as u32)) {
            println!(
                "  - Rendu et conversion de la page {}/{}...",
                page_num, page_count_scan
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


            if let Some(channel) = chan {
                realpage = realpage + 1;
                let mut numberpages = pages_to_manage.len();
                if numberpages == 0 {
                    numberpages = page_count_scan as usize;
                }
                let progress = ((realpage as f32 / numberpages as f32) * 100.0) as u32;
                let routing_key = format!("scan.status.{}", exam_id);
                let status_msg = ScanStatus {
                    scan_id: exam_id,
                    page: page_num as u32,
                    status: "running".to_string(),
                    progress: progress,
                    details: format!("Page {} traitée", page_num),
                };
                let payload = serde_json::to_vec(&status_msg)?;
                channel.basic_publish(
                    outputchan,
                    &routing_key,
                    BasicPublishOptions::default(),
                    &payload,
                    BasicProperties::default(),
                ).await?; // Note: await sur publish attend juste la confirmation d'envoi réseau
            }
        }
    }

    drop(document_scan); // Demonstrate that the page can be used after the document is dropped.
    drop(document_template); // Demonstrate that the page can be used after the document is dropped.
    let _ = close_connection(conn);
    let content = fs::read(format!("{}.sqlite3", exam_id))?;

    let _ = put_object(
        server_url,
        bucket_name,
        content.as_slice(),
        &format!("cache/{}.sqlite3", exam_id),
        login,
        pass,
    );
    fs::remove_file(format!("{}.sqlite3", exam_id))?;

    Ok(())
}


/// Logique cœur du Worker
async fn process_message(channel: &Channel, data: &[u8],
    debug: bool,
    server_url: &str,
    bucket_name: &str,
    login: &str,
    pass: &str,
       mq_server_ouputqueue: &String
) -> Result<(), Box<dyn Error>> {
    // 1. Désérialisation
    let request: ScanRequest = serde_json::from_slice(data)?;
    println!("   Détails : Scan ID {}, Pages '{}', Algo {}", request.scan_id, request.pages_to_manage, request.algo);

    // 2. Parsing des pages
    let pages = parse_page_selection(&request.pages_to_manage).unwrap_or_default();
    let total = pages.len();

    if total == 0 {
        println!("   Aucune page à traiter.");
        return Ok(());
    }

    process_exam(
        pages,
        request.exam_id,
        request.template_id,
        request.scan_id,
        match request.algo {
            0 => AlignAlgo::NoAlign,
            1 => AlignAlgo::CircleAlign,
            2 => AlignAlgo::Akaze,
            _ => AlignAlgo::NoAlign,
        },
        debug,
        server_url,
        bucket_name,
        login,
        pass,
        Some(channel),
        mq_server_ouputqueue
    ).await?;

 
    // 5. Message final de complétion
    let final_key = format!("scan.status.{}", request.scan_id);
    let final_msg = ScanStatus {
        scan_id: request.exam_id,
        page: 0,
        status: "finished".to_string(),
        progress: 100,
        details: "Traitement complet terminé".to_string(),
    };
    
    channel.basic_publish(
        mq_server_ouputqueue,
        &final_key,
        BasicPublishOptions::default(),
        &serde_json::to_vec(&final_msg)?,
        BasicProperties::default(),
    ).await?;

    Ok(())
}