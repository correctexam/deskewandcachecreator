use akaze::Akaze;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use nalgebra::{Matrix3, Vector2, Vector3, Matrix4, Vector4};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::thread_rng;
use rayon::iter::{ParallelBridge, ParallelIterator};
// =======================================================================
// CONFIGURATION
// =======================================================================
const ROI_SIZE: u32 = 300; // Taille des zones de coins (300x300)
const RANSAC_ITER: usize = 2000;
const RANSAC_THRESH: f64 = 3.0; // Tolérance en pixels

type Point2 = Vector2<f64>;

#[derive(Clone, Copy, Debug)]
struct Match {
    src: Point2, // Point Global Scan
    dst: Point2, // Point Global Ref
}

fn main() {
    // Remplacer par vos images réelles
    // let img_ref = image::open("ref.jpg").unwrap();
    // let img_scan = image::open("scan.jpg").unwrap();
    
    // Simulation pour l'exemple
    let img_ref = DynamicImage::ImageRgb8(ImageBuffer::new(2000, 3000));
    let img_scan = DynamicImage::ImageRgb8(ImageBuffer::new(2100, 3100)); // Scan légèrement plus grand/décalé

    match deskew_constrained(&img_ref, &img_scan) {
        Ok(res) => {
            res.save("deskewed_similarity.jpg").unwrap();
            println!("Terminé ! Image sauvegardée.");
        }
        Err(e) => eprintln!("Erreur: {}", e),
    }
}

/// Fonction principale
pub fn deskew_constrained(img_ref: &DynamicImage, img_scan: &DynamicImage) -> Result<DynamicImage, String> {
    
    // 1. Extraction et Matching par zones (4 coins)
    println!("Extraction des features sur les 4 zones...");
    let all_matches = compute_roi_matches(img_ref, img_scan)?;
    println!("Total correspondances fusionnées : {}", all_matches.len());

    // 2. Calcul de la transformation de Similitude (Rot + Trans + Scale) via RANSAC
    println!("Calcul géométrique (Similitude)...");
    let transform_matrix = compute_similarity_ransac(&all_matches, RANSAC_ITER, RANSAC_THRESH)
        .ok_or("RANSAC n'a pas trouvé de modèle cohérent.")?;

    println!("Matrice de transformation trouvée :\n{:.4}", transform_matrix);
    
    // Analyse rapide de la matrice pour info
    let a = transform_matrix[(0,0)];
    let b = transform_matrix[(1,0)];
    let scale = (a*a + b*b).sqrt();
    let angle = b.atan2(a).to_degrees();
    println!("Rotation détectée: {:.2}°, Échelle: {:.4}", angle, scale);

    // 3. Application de la transformation (Warp)
    // On conserve la taille de l'image de référence
    let (out_w, out_h) = img_ref.dimensions();
    let warped = warp_similarity(img_scan, &transform_matrix, out_w, out_h);

    Ok(DynamicImage::ImageRgb8(warped))
}

// =======================================================================
// ÉTAPE 1 : GESTION DES ROI (ZONES)
// =======================================================================

fn compute_roi_matches(img_ref: &DynamicImage, img_scan: &DynamicImage) -> Result<Vec<Match>, String> {
    let mut fused_matches = Vec::new();
    let akaze = Akaze::default();

    // Définition des offsets pour les 4 coins
    // (TopLeft, TopRight, BottomRight, BottomLeft)
    let rois = vec![
        (0, 0), // TL
        (1, 0), // TR
        (1, 1), // BR
        (0, 1)  // BL
    ];

    let (w_ref, h_ref) = img_ref.dimensions();
    let (w_scan, h_scan) = img_scan.dimensions();

    for (pos_x, pos_y) in rois {
        // Calcul des coordonnées du coin supérieur gauche de la ROI
        let ref_x = if pos_x == 0 { 0 } else { w_ref.saturating_sub(ROI_SIZE) };
        let ref_y = if pos_y == 0 { 0 } else { h_ref.saturating_sub(ROI_SIZE) };
        
        let scan_x = if pos_x == 0 { 0 } else { w_scan.saturating_sub(ROI_SIZE) };
        let scan_y = if pos_y == 0 { 0 } else { h_scan.saturating_sub(ROI_SIZE) };

        // Extraction des sous-images
        let roi_ref = img_ref.view(ref_x, ref_y, ROI_SIZE, ROI_SIZE).to_image();
        let roi_scan = img_scan.view(scan_x, scan_y, ROI_SIZE, ROI_SIZE).to_image();
        
        // AKAZE sur les crops (conversion implicite DynamicImage via to_image qui rend un ImageBuffer)
        // Akaze 0.7 prend des bytes Luma8.
        let (kps_ref, desc_ref) = akaze_extract(&DynamicImage::ImageRgba8(roi_ref));
        let (kps_scan, desc_scan) = akaze_extract(&DynamicImage::ImageRgba8(roi_scan));

        // Matching local
        let local_matches = match_features(&kps_scan, &desc_scan, &kps_ref, &desc_ref);

        // Conversion en coordonnées GLOBALES
        for m in local_matches {
            fused_matches.push(Match {
                src: Vector2::new(m.src.x + scan_x as f64, m.src.y + scan_y as f64),
                dst: Vector2::new(m.dst.x + ref_x as f64, m.dst.y + ref_y as f64),
            });
        }
    }

    if fused_matches.len() < 2 {
        return Err("Pas assez de points trouvés dans les coins.".to_string());
    }

    Ok(fused_matches)
}

fn akaze_extract(img: &DynamicImage) -> (Vec<akaze::KeyPoint>, Vec<Vec<u8>>) {
    let gray = img.to_luma8();
     let dgray = DynamicImage::ImageLuma8(gray);
    let akaze = Akaze::default();
    let    t = akaze.extract(&dgray);
    let desc=    t.1.iter().map(|b| b.bytes.to_vec()).collect::<Vec<Vec<u8>>>();
    (t.0, desc)
}

// =======================================================================
// ÉTAPE 2 : MATHS & RANSAC (SIMILITUDE)
// =======================================================================

/// Résout une transformation de similitude (s * R + t) avec seulement 2 paires de points
/// Modèle : 
/// x' = a*x - b*y + tx
/// y' = b*x + a*y + ty
/// Où a = s*cos(theta), b = s*sin(theta)
fn solve_similarity_2_points(m1: &Match, m2: &Match) -> Option<Matrix3<f64>> {
    // Système linéaire A * [a, b, tx, ty]^T = B
    // Avec 2 points, on a 4 équations, exactement ce qu'il faut.
    
    let x1 = m1.src.x; let y1 = m1.src.y;
    let u1 = m1.dst.x; let v1 = m1.dst.y;
    
    let x2 = m2.src.x; let y2 = m2.src.y;
    let u2 = m2.dst.x; let v2 = m2.dst.y;

    // Construction matrice A (4x4) et vecteur B (4x1)
    // Eq 1: x1*a - y1*b + tx = u1
    // Eq 2: y1*a + x1*b + ty = v1
    // ... idem point 2
    
    let a_mat = Matrix4::new(
        x1, -y1, 1.0, 0.0,
        y1,  x1, 0.0, 1.0,
        x2, -y2, 1.0, 0.0,
        y2,  x2, 0.0, 1.0,
    );
    
    let b_vec = Vector4::new(u1, v1, u2, v2);

    // Résolution via inversion (rapide pour 4x4) ou LU decomposition
    if let Some(inv_a) = a_mat.try_inverse() {
        let res = inv_a * b_vec; // [a, b, tx, ty]
        
        let a = res[0];
        let b = res[1];
        let tx = res[2];
        let ty = res[3];

        // Matrice de transformation 3x3 homogène
        return Some(Matrix3::new(
             a, -b, tx,
             b,  a, ty,
           0.0, 0.0, 1.0
        ));
    }
    
    None
}

fn compute_similarity_ransac(matches: &[Match], iter: usize, thresh: f64) -> Option<Matrix3<f64>> {
    let mut best_h = Matrix3::identity();
    let mut max_inliers = 0;
    let mut rng = thread_rng();

    if matches.len() < 2 { return None; }

    for _ in 0..iter {
        // On a besoin de seulement 2 points pour une similitude !
        let sample: Vec<_> = matches.choose_multiple(&mut rng, 2).cloned().collect();
        if sample.len() < 2 { break; }

        if let Some(h) = solve_similarity_2_points(&sample[0], &sample[1]) {
            let inliers_cnt = matches.iter().filter(|m| {
                // Projection : H * src
                let src_vec = Vector3::new(m.src.x, m.src.y, 1.0);
                let dst_est = h * src_vec;
                // Pas de division perspective nécessaire pour similitude (z=1), mais par sécurité :
                let est_pt = Vector2::new(dst_est.x, dst_est.y);
                
                (est_pt - m.dst).norm() < thresh
            }).count();

            if inliers_cnt > max_inliers {
                max_inliers = inliers_cnt;
                best_h = h;
            }
        }
    }
    
    if max_inliers > 4 { // Seuil très bas car on a filtré par zones
        Some(best_h)
    } else {
        None
    }
}

// =======================================================================
// ÉTAPE 3 : WARPING OPTIMISÉ
// =======================================================================

fn warp_similarity(
    src_img: &DynamicImage,
    transform: &Matrix3<f64>,
    out_w: u32,
    out_h: u32,
) -> RgbImage {
    let h_inv = transform.try_inverse().expect("Matrice non inversible");
    
    let mut out_img = ImageBuffer::new(out_w, out_h);
    let src_rgb = src_img.to_rgb8();
    let (src_w, src_h) = src_rgb.dimensions();

    // Parallélisme rayon
    out_img.enumerate_pixels_mut().par_bridge().for_each(|(x, y, pixel)| {
        let dest_vec = Vector3::new(x as f64, y as f64, 1.0);
        
        // Transformation inverse : de dest vers source
        let src_loc = h_inv * dest_vec;
        
        // Similitude préserve le plan à l'infini, z reste 1.0 normalement.
        let u = src_loc.x;
        let v = src_loc.y;

        if u >= 0.0 && u < (src_w as f64 - 1.0) && v >= 0.0 && v < (src_h as f64 - 1.0) {
            *pixel = interpolate_bilinear(&src_rgb, u, v);
        } else {
            *pixel = Rgb([0, 0, 0]); // Fond noir si hors champ
        }
    });

    out_img
}

// (Reprise de la fonction d'interpolation précédente pour être complet)
fn interpolate_bilinear(img: &RgbImage, u: f64, v: f64) -> Rgb<u8> {
    let x1 = u.floor() as u32;
    let y1 = v.floor() as u32;
    let x2 = x1 + 1;
    let y2 = y1 + 1;

    let ratio_x = u - x1 as f64;
    let ratio_y = v - y1 as f64;

    let p11 = img.get_pixel(x1, y1).0;
    let p21 = img.get_pixel(x2, y1).0;
    let p12 = img.get_pixel(x1, y2).0;
    let p22 = img.get_pixel(x2, y2).0;

    let mut final_pixel = [0u8; 3];
    for i in 0..3 {
        let val11 = p11[i] as f64;
        let val21 = p21[i] as f64;
        let val12 = p12[i] as f64;
        let val22 = p22[i] as f64;

        let ip_y1 = val11 * (1.0 - ratio_x) + val21 * ratio_x;
        let ip_y2 = val12 * (1.0 - ratio_x) + val22 * ratio_x;
        let res = ip_y1 * (1.0 - ratio_y) + ip_y2 * ratio_y;
        final_pixel[i] = res.round() as u8;
    }
    Rgb(final_pixel)
}

// Helper Matching (identique précédemment)
fn match_features(kps_src: &[akaze::KeyPoint], desc_src: &[Vec<u8>], kps_dst: &[akaze::KeyPoint], desc_dst: &[Vec<u8>]) -> Vec<Match> {
    let mut matches = Vec::new();
    let ratio = 0.8; 
    // Optimization possible: si desc_dst est vide, skip
    if desc_dst.is_empty() { return matches; }

    for (i, d_src) in desc_src.iter().enumerate() {
        let mut best_dist = u32::MAX;
        let mut second_dist = u32::MAX;
        let mut best_idx = 0;

        for (j, d_dst) in desc_dst.iter().enumerate() {
            let dist = hamming_distance(d_src, d_dst);
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = j;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }
        if (best_dist as f32) < (ratio * second_dist as f32) {
             matches.push(Match {
                src: Vector2::new(kps_src[i].point.0 as f64, kps_src[i].point.1 as f64),
                dst: Vector2::new(kps_dst[best_idx].point.0 as f64, kps_dst[best_idx].point.1 as f64),
            });
        }
    }
    matches
}

fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}