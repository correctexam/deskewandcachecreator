use image::{GenericImage, GrayImage, Rgba, RgbaImage};
use image::{DynamicImage,  GenericImageView};

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct Circle {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
}



#[derive(Debug)]
pub struct SimilarityTransform {
    pub scale: f32,
    pub rotation: f32,
    pub tx: f32,
    pub ty: f32,
}


/// Seuils de reconnaissance d'un rond de calage plein.
const MIN_DISK_FILL_RATIO: f32 = 0.80; // un carré plein donne 4/π ≈ 1,27
const MAX_DISK_FILL_RATIO: f32 = 1.15;
const MAX_DISK_ASPECT_RATIO: f32 = 1.3; // largeur / hauteur de la boîte englobante
/// Tolérance de l'ajustement : 5 % du rayon, mais jamais moins d'un pixel
/// (l'erreur de discrétisation du contour approche 0,5 px sur les petits ronds).
const MAX_RMS_RATIO: f32 = 0.05;
const MIN_RMS_TOLERANCE_PX: f32 = 1.0;

/// Composante connexe (4-connexité) d'une image binaire.
struct Blob {
    pixels: Vec<(usize, usize)>,
    minx: usize,
    maxx: usize,
    miny: usize,
    maxy: usize,
}

impl Blob {
    fn width(&self) -> usize {
        self.maxx - self.minx + 1
    }
    fn height(&self) -> usize {
        self.maxy - self.miny + 1
    }
}

/// Étiquetage sans limite de nombre de composantes (l'ancienne étiquette `u8`
/// débordait au-delà de 254 composantes et pouvait boucler indéfiniment).
fn connected_components(bin: &[Vec<u8>], w: usize, h: usize) -> Vec<Blob> {
    let mut visited = vec![vec![false; w]; h];
    let mut blobs = Vec::new();

    for y0 in 0..h {
        for x0 in 0..w {
            if bin[y0][x0] != 1 || visited[y0][x0] {
                continue;
            }
            let mut blob = Blob { pixels: Vec::new(), minx: x0, maxx: x0, miny: y0, maxy: y0 };
            let mut stack = vec![(x0, y0)];
            visited[y0][x0] = true;
            while let Some((x, y)) = stack.pop() {
                blob.pixels.push((x, y));
                blob.minx = blob.minx.min(x);
                blob.maxx = blob.maxx.max(x);
                blob.miny = blob.miny.min(y);
                blob.maxy = blob.maxy.max(y);
                let neighbours = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];
                for (nx, ny) in neighbours {
                    if nx < w && ny < h && bin[ny][nx] == 1 && !visited[ny][nx] {
                        visited[ny][nx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            blobs.push(blob);
        }
    }
    blobs
}

/// Une composante est un rond plein candidat si elle est entièrement dans la
/// zone (un rond coupé par le bord donne une corde droite), à peu près aussi
/// large que haute et remplie comme un disque.
fn is_disk_candidate(blob: &Blob, w: usize, h: usize, min_radius: f32) -> bool {
    // erode/dilate mettent à 0 la première et la dernière ligne/colonne
    if blob.minx <= 1 || blob.miny <= 1 || blob.maxx + 2 >= w || blob.maxy + 2 >= h {
        return false;
    }
    let bw = blob.width() as f32;
    let bh = blob.height() as f32;
    if bw.max(bh) / bw.min(bh) > MAX_DISK_ASPECT_RATIO {
        return false;
    }
    let radius = (bw + bh) / 4.0;
    if radius <= min_radius {
        return false;
    }
    let fill_ratio = blob.pixels.len() as f32 / (std::f32::consts::PI * radius * radius);
    (MIN_DISK_FILL_RATIO..=MAX_DISK_FILL_RATIO).contains(&fill_ratio)
}

/// Contour (pixels ayant un voisin hors composante) d'une seule composante.
fn blob_border_points(blob: &Blob, x0: u32, y0: u32) -> Vec<(f32, f32)> {
    let w = blob.width() + 2;
    let h = blob.height() + 2;
    let mut mask = vec![vec![false; w]; h];
    for &(x, y) in &blob.pixels {
        mask[y - blob.miny + 1][x - blob.minx + 1] = true;
    }
    blob.pixels
        .iter()
        .filter(|&&(x, y)| {
            let (lx, ly) = (x - blob.minx + 1, y - blob.miny + 1);
            !(mask[ly - 1][lx] && mask[ly + 1][lx] && mask[ly][lx - 1] && mask[ly][lx + 1])
        })
        .map(|&(x, y)| (x as f32 + x0 as f32, y as f32 + y0 as f32))
        .collect()
}

/// Conservé pour compatibilité : ne garde que les composantes qui ressemblent
/// à un rond plein entièrement contenu dans la zone.
pub fn filter_filled_circles(bin: &Vec<Vec<u8>>, w: usize, h: usize, min_radius: f32) -> Vec<Vec<u8>> {
    let mut output = vec![vec![0u8; w]; h];
    for blob in connected_components(bin, w, h) {
        if is_disk_candidate(&blob, w, h, min_radius) {
            for &(px, py) in &blob.pixels {
                output[py][px] = 1;
            }
        }
    }
    output
}

/// Écart quadratique moyen (en pixels) entre les points et le cercle.
pub fn circle_rms(circle: &Circle, points: &[(f32, f32)]) -> f32 {
    let err2: f32 = points
        .iter()
        .map(|(x, y)| (((x - circle.cx).powi(2) + (y - circle.cy).powi(2)).sqrt() - circle.r).powi(2))
        .sum();
    (err2 / points.len() as f32).sqrt()
}

pub fn validate_rms(circle: &Circle, points: &[(f32, f32)], max_ratio: f32) -> bool {
    circle_rms(circle, points) < (circle.r * max_ratio).max(MIN_RMS_TOLERANCE_PX)
}

pub fn detect_circles_in_four_corners_advanced(
    gray: &GrayImage,
    roi_size: u32,
    min_radius: f32
) -> Vec<Option<Circle>> {
    let w = gray.width();
    let h = gray.height();
    vec![
        // Coin haut-gauche
        detect_circle_roi_advanced(gray, 1, 1, roi_size - 1,min_radius),
        // Coin haut-droit
        detect_circle_roi_advanced(gray, w - roi_size, 0, roi_size,min_radius),
        // Coin bas-gauche
        detect_circle_roi_advanced(gray, 1, h - roi_size, roi_size - 1,min_radius),
        // Coin bas-droit
        detect_circle_roi_advanced(gray, w - roi_size, h - roi_size, roi_size,min_radius),
    ]
}

fn extract_border_points(bin: &Vec<Vec<u8>>, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut pts = Vec::new();

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if bin[y][x] == 1 {
                let mut edge = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if bin[(y as isize + dy) as usize][(x as isize + dx) as usize] == 0 {
                            edge = true;
                        }
                    }
                }
                if edge {
                    pts.push((x, y));
                }
            }
        }
    }
    pts
}

pub fn detect_circle_roi_advanced(gray: &GrayImage, x0: u32, y0: u32, size: u32, min_radius: f32) -> Option<Circle> {
    let roi = gray.view(x0, y0, size, size).to_image();
    // Otsu : la classe sombre regroupe les niveaux <= t
    let t = otsu_threshold(&roi);
    let n = size as usize;

    let mut bin = vec![vec![0u8; n]; n];
    for y in 0..size {
        for x in 0..size {
            bin[y as usize][x as usize] = if roi.get_pixel(x, y)[0] <= t { 1 } else { 0 };
        }
    }
    let bin = dilate(&erode(&bin, n, n), n, n);

    // Un seul rond est ajusté : le plus grand candidat. Avant, le contour de
    // toutes les composantes retenues (texte, case noircie, ombre de bord…)
    // était ajusté comme un seul cercle, d'où un RMS élevé.
    let blobs = connected_components(&bin, n, n);
    let Some(blob) = blobs
        .iter()
        .filter(|b| is_disk_candidate(b, n, n, min_radius))
        .max_by_key(|b| b.pixels.len())
    else {
        println!("    ! Aucun rond plein entier dans la zone ({x0}, {y0}, {size}px)");
        return None;
    };

    let points = blob_border_points(blob, x0, y0);
    let circle = fit_circle_taubin(&points)?;
    let rms = circle_rms(&circle, &points);
    if !validate_rms(&circle, &points, MAX_RMS_RATIO) {
        println!(
            "    ! Avertissement : RMS trop élevé pour le cercle détecté (zone ({x0}, {y0}), rayon {:.1}px, RMS {:.2}px)",
            circle.r, rms
        );
        return None;
    }

    Some(circle)
}

pub fn validate_radial_gradient(gray: &GrayImage, circle: &Circle, points: &[(f32, f32)]) -> bool {
    let (w, h) = gray.dimensions();
    let mut score = 0;
    let mut count = 0;

    for (x, y) in points {
        let xi = *x as i32;
        let yi = *y as i32;

        if xi <= 0 || yi <= 0 || xi >= (w - 1) as i32 || yi >= (h - 1) as i32 {
            continue;
        }

        let gx = gray.get_pixel((xi + 1) as u32, yi as u32)[0] as f32
            - gray.get_pixel((xi - 1) as u32, yi as u32)[0] as f32;
        let gy = gray.get_pixel(xi as u32, (yi + 1) as u32)[0] as f32
            - gray.get_pixel(xi as u32, (yi - 1) as u32)[0] as f32;

        let vx = x - circle.cx;
        let vy = y - circle.cy;

        //        let dot = gx * vx + gy * vy;

        let grad_norm = (gx * gx + gy * gy).sqrt();
        let rad_norm = (vx * vx + vy * vy).sqrt();

        if grad_norm > 1.0 && rad_norm > 1.0 {
            let cos_angle = (gx * vx + gy * vy) / (grad_norm * rad_norm);

            // Cercle noir → gradient vers le centre
            if cos_angle < -0.7 {
                score += 1;
            }
        }

        /*        if dot < 0.0 {
            score += 1;
        } */
        count += 1;
    }

    count > 0 && (score as f32 / count as f32) > 0.7
}

pub fn fit_circle_pratt(points: &[(f32, f32)]) -> Option<Circle> {
    let n = points.len();
    if n < 20 {
        return None;
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;

    for (x, y) in points {
        mean_x += x;
        mean_y += y;
    }
    mean_x /= n as f32;
    mean_y /= n as f32;

    let mut suu = 0.0;
    let mut svv = 0.0;
    let mut suv = 0.0;
    let mut suuu = 0.0;
    let mut svvv = 0.0;
    let mut suvv = 0.0;
    let mut svuu = 0.0;

    for (x, y) in points {
        let u = x - mean_x;
        let v = y - mean_y;

        let u2 = u * u;
        let v2 = v * v;

        suu += u2;
        svv += v2;
        suv += u * v;
        suuu += u2 * u;
        svvv += v2 * v;
        suvv += u * v2;
        svuu += v * u2;
    }

    let a = suu;
    let b = suv;
    let c = svv;
    let d = 0.5 * (suuu + suvv);
    let e = 0.5 * (svvv + svuu);

    let det = a * c - b * b;
    if det.abs() < 1e-6 {
        return None;
    }

    let uc = (d * c - b * e) / det;
    let vc = (a * e - b * d) / det;

    let cx = uc + mean_x;
    let cy = vc + mean_y;

    let r = points
        .iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .sum::<f32>()
        / n as f32;

    Some(Circle { cx, cy, r })
}

pub fn detect_black_circles_from_image(img: &DynamicImage, roi_size: u32) -> Vec<Circle> {
    // Conversion en niveaux de gris
    let gray = img.to_luma8();

    // Détection des cercles dans les 4 coins
    let results = detect_corners(&gray, roi_size);

    let mut circles = Vec::new();

    let corner_names = ["Haut-Gauche", "Haut-Droit", "Bas-Gauche", "Bas-Droit"];

    for (i, res) in results.iter().enumerate() {
        match res {
            Some(c) => {
                println!(
                    "[{}] Cercle détecté : centre=({:.2}, {:.2}), rayon={:.2}",
                    corner_names[i], c.cx, c.cy, c.r
                );
                circles.push(c.clone());
            }
            None => {
                println!("[{}] Aucun cercle détecté", corner_names[i]);
            }
        }
    }

    circles
}

fn erode(bin: &Vec<Vec<u8>>, w: usize, h: usize) -> Vec<Vec<u8>> {
    let mut out = vec![vec![0; w]; h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut ok = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if bin[(y as isize + dy) as usize][(x as isize + dx) as usize] == 0 {
                        ok = false;
                    }
                }
            }
            out[y][x] = if ok { 1 } else { 0 };
        }
    }
    out
}

fn fit_circle_taubin(points: &[(f32, f32)]) -> Option<Circle> {
    let n = points.len();
    if n < 20 {
        return None;
    }

    // Centrage
    let (mut mx, mut my) = (0.0f32, 0.0f32);
    for (x, y) in points {
        mx += *x;
        my += *y;
    }
    mx /= n as f32;
    my /= n as f32;

    // Moments
    let mut suu = 0.0;
    let mut svv = 0.0;
    let mut suv = 0.0;
    let mut suuu = 0.0;
    let mut svvv = 0.0;
    let mut suvv = 0.0;
    let mut svuu = 0.0;

    for (x, y) in points {
        let u = x - mx;
        let v = y - my;

        let u2 = u * u;
        let v2 = v * v;

        suu += u2;
        svv += v2;
        suv += u * v;
        suuu += u2 * u;
        svvv += v2 * v;
        suvv += u * v2;
        svuu += v * u2;
    }

    let a = suu;
    let b = suv;
    let c = svv;
    let d = 0.5 * (suuu + suvv);
    let e = 0.5 * (svvv + svuu);

    let det = a * c - b * b;
    if det.abs() < 1e-6 {
        return None;
    }

    let uc = (d * c - b * e) / det;
    let vc = (a * e - b * d) / det;

    let cx = uc + mx;
    let cy = vc + my;

    let r = points
        .iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .sum::<f32>()
        / n as f32;

    Some(Circle { cx, cy, r })
}

fn otsu_threshold(img: &image::GrayImage) -> u8 {
    let mut hist = [0u32; 256];

    for p in img.pixels() {
        hist[p[0] as usize] += 1;
    }

    let total: u32 = img.width() * img.height();
    let mut sum = 0u32;
    for i in 0..256 {
        sum += (i as u32) * hist[i];
    }

    let mut sum_b = 0u32;
    let mut w_b = 0u32;
    let mut max_var = 0.0;
    let mut threshold = 0u8;

    for i in 0..256 {
        w_b += hist[i];
        if w_b == 0 {
            continue;
        }

        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }

        sum_b += (i as u32) * hist[i];

        let m_b = sum_b as f32 / w_b as f32;
        let m_f = (sum - sum_b) as f32 / w_f as f32;

        let var_between = (w_b as f32) * (w_f as f32) * (m_b - m_f).powi(2);

        if var_between > max_var {
            max_var = var_between;
            threshold = i as u8;
        }
    }

    threshold
}

fn dilate(bin: &Vec<Vec<u8>>, w: usize, h: usize) -> Vec<Vec<u8>> {
    let mut out = vec![vec![0; w]; h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut found = false;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if bin[(y as isize + dy) as usize][(x as isize + dx) as usize] == 1 {
                        found = true;
                    }
                }
            }
            out[y][x] = if found { 1 } else { 0 };
        }
    }
    out
}

fn fit_circle_kasa(points: &Vec<(f32, f32)>) -> Option<Circle> {
    if points.len() < 20 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x3 = 0.0;
    let mut sum_y3 = 0.0;
    let mut sum_x1y2 = 0.0;
    let mut sum_x2y1 = 0.0;

    let n = points.len() as f32;

    for (x, y) in points {
        let x2 = x * x;
        let y2 = y * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x2;
        sum_y2 += y2;
        sum_xy += x * y;
        sum_x3 += x2 * x;
        sum_y3 += y2 * y;
        sum_x1y2 += x * y2;
        sum_x2y1 += x2 * y;
    }

    let c = n * sum_x2 - sum_x * sum_x;
    let d = n * sum_xy - sum_x * sum_y;
    let e = n * sum_y2 - sum_y * sum_y;
    let g = 0.5 * (n * sum_x3 + n * sum_x1y2 - (sum_x2 + sum_y2) * sum_x);
    let h = 0.5 * (n * sum_y3 + n * sum_x2y1 - (sum_x2 + sum_y2) * sum_y);

    let denom = c * e - d * d;
    if denom.abs() < 1e-6 {
        return None;
    }

    let cx = (g * e - d * h) / denom;
    let cy = (c * h - d * g) / denom;

    let r = points
        .iter()
        .map(|(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .sum::<f32>()
        / n;

    Some(Circle { cx, cy, r })
}

pub fn detect_circle_roi(img: &GrayImage, x0: u32, y0: u32, size: u32) -> Option<Circle> {
    let mut roi = GrayImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            roi.put_pixel(x, y, *img.get_pixel(x0 + x, y0 + y));
        }
    }

    let t = otsu_threshold(&roi);

    let w = size as usize;
    let h = size as usize;
    let mut bin = vec![vec![0u8; w]; h];

    for y in 0..h {
        for x in 0..w {
            bin[y][x] = if roi.get_pixel(x as u32, y as u32)[0] < t {
                1
            } else {
                0
            };
        }
    }

    let bin = dilate(&erode(&bin, w, h), w, h);

    let mut points = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if bin[y][x] == 1 {
                points.push((x as f32 + x0 as f32, y as f32 + y0 as f32));
            }
        }
    }

    fit_circle_kasa(&points)
}

pub fn detect_corners(img: &GrayImage, roi: u32) -> Vec<Option<Circle>> {
    let w = img.width();
    let h = img.height();

    vec![
        detect_circle_roi(img, 0, 0, roi),
        detect_circle_roi(img, w - roi, 0, roi),
        detect_circle_roi(img, 0, h - roi, roi),
        detect_circle_roi(img, w - roi, h - roi, roi),
    ]
}


pub fn draw_circles_blue(img: &mut DynamicImage, circles: &Vec<Option<Circle>>, thickness: i32) {
    let (w, h) = img.dimensions();
    let blue = Rgba([0, 0, 255, 255]);

    for c in circles {
        if c.is_some() {
            let circle = c.as_ref().unwrap();
            let cx = circle.cx.round() as i32;
            let cy = circle.cy.round() as i32;
            let r = circle.r.round() as i32;

            // Dessin du contour par échantillonnage angulaire
            let steps = (2.0 * std::f32::consts::PI * r as f32) as i32;

            for i in 0..steps {
                let theta = i as f32 / steps as f32 * 2.0 * std::f32::consts::PI;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                for t in -thickness..=thickness {
                    let x = cx + ((r + t) as f32 * cos_t) as i32;
                    let y = cy + ((r + t) as f32 * sin_t) as i32;

                    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                        img.put_pixel(x as u32, y as u32, blue);
                    }
                }
            }
        }
    }
}

pub fn estimate_similarity(src: &[Point; 4], dst: &[Point; 4]) -> SimilarityTransform {
    // 1. Barycentres
    let cs = Point {
        x: src.iter().map(|p| p.x).sum::<f32>() / 4.0,
        y: src.iter().map(|p| p.y).sum::<f32>() / 4.0,
    };

    let cd = Point {
        x: dst.iter().map(|p| p.x).sum::<f32>() / 4.0,
        y: dst.iter().map(|p| p.y).sum::<f32>() / 4.0,
    };

    // 2. Centrage
    let src_c: Vec<Point> = src
        .iter()
        .map(|p| Point {
            x: p.x - cs.x,
            y: p.y - cs.y,
        })
        .collect();

    let dst_c: Vec<Point> = dst
        .iter()
        .map(|p| Point {
            x: p.x - cd.x,
            y: p.y - cd.y,
        })
        .collect();

    // 3. Échelle
    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..4 {
        num += (dst_c[i].x.powi(2) + dst_c[i].y.powi(2)).sqrt();
        den += (src_c[i].x.powi(2) + src_c[i].y.powi(2)).sqrt();
    }

    let scale = num / den;

    // 4. Rotation
    let mut a = 0.0;
    let mut b = 0.0;

    for i in 0..4 {
        a += src_c[i].x * dst_c[i].y - src_c[i].y * dst_c[i].x;
        b += src_c[i].x * dst_c[i].x + src_c[i].y * dst_c[i].y;
    }

    let rotation = a.atan2(b);

    // 5. Translation
    let cos_t = rotation.cos();
    let sin_t = rotation.sin();

    let tx = cd.x - scale * (cos_t * cs.x - sin_t * cs.y);
    let ty = cd.y - scale * (sin_t * cs.x + cos_t * cs.y);

    SimilarityTransform {
        scale,
        rotation,
        tx,
        ty,
    }
}

pub fn apply_similarity(
    img: &DynamicImage,
    t: &SimilarityTransform,
    out_w: u32,
    out_h: u32,
) -> RgbaImage {
    let mut out = RgbaImage::new(out_w, out_h);

    let cos_t = t.rotation.cos();
    let sin_t = t.rotation.sin();

    for y in 0..out_h {
        for x in 0..out_w {
            let xf = x as f32;
            let yf = y as f32;

            // Inverse mapping
            let xs = (xf - t.tx) / t.scale;
            let ys = (yf - t.ty) / t.scale;

            let src_x = cos_t * xs + sin_t * ys;
            let src_y = -sin_t * xs + cos_t * ys;

            if src_x >= 0.0
                && src_y >= 0.0
                && src_x < img.width() as f32
                && src_y < img.height() as f32
            {
                let px = img.get_pixel(src_x as u32, src_y as u32);
                out.put_pixel(x, y, px);
            } else {
                out.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
    }

    out
}

pub fn estimate_similarity_3pts(src: &[Point; 3], dst: &[Point; 3]) -> SimilarityTransform {
    // 1. Barycentres
    let cs = Point {
        x: (src[0].x + src[1].x + src[2].x) / 3.0,
        y: (src[0].y + src[1].y + src[2].y) / 3.0,
    };

    let cd = Point {
        x: (dst[0].x + dst[1].x + dst[2].x) / 3.0,
        y: (dst[0].y + dst[1].y + dst[2].y) / 3.0,
    };

    // 2. Centrage
    let src_c = [
        Point {
            x: src[0].x - cs.x,
            y: src[0].y - cs.y,
        },
        Point {
            x: src[1].x - cs.x,
            y: src[1].y - cs.y,
        },
        Point {
            x: src[2].x - cs.x,
            y: src[2].y - cs.y,
        },
    ];

    let dst_c = [
        Point {
            x: dst[0].x - cd.x,
            y: dst[0].y - cd.y,
        },
        Point {
            x: dst[1].x - cd.x,
            y: dst[1].y - cd.y,
        },
        Point {
            x: dst[2].x - cd.x,
            y: dst[2].y - cd.y,
        },
    ];

    // 3. Échelle
    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..3 {
        num += (dst_c[i].x.powi(2) + dst_c[i].y.powi(2)).sqrt();
        den += (src_c[i].x.powi(2) + src_c[i].y.powi(2)).sqrt();
    }

    let scale = num / den;

    // 4. Rotation
    let mut a = 0.0;
    let mut b = 0.0;

    for i in 0..3 {
        a += src_c[i].x * dst_c[i].y - src_c[i].y * dst_c[i].x;
        b += src_c[i].x * dst_c[i].x + src_c[i].y * dst_c[i].y;
    }

    let rotation = a.atan2(b);

    // 5. Translation
    let cos_t = rotation.cos();
    let sin_t = rotation.sin();

    let tx = cd.x - scale * (cos_t * cs.x - sin_t * cs.y);
    let ty = cd.y - scale * (sin_t * cs.x + cos_t * cs.y);

    SimilarityTransform {
        scale,
        rotation,
        tx,
        ty,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    fn page(w: u32, h: u32, shapes: &[&dyn Fn(f32, f32) -> bool]) -> GrayImage {
        GrayImage::from_fn(w, h, |x, y| {
            let (xf, yf) = (x as f32 + 0.5, y as f32 + 0.5);
            if shapes.iter().any(|s| s(xf, yf)) { Luma([20]) } else { Luma([235]) }
        })
    }
    fn disk(cx: f32, cy: f32, r: f32) -> impl Fn(f32, f32) -> bool {
        move |x, y| (x - cx).powi(2) + (y - cy).powi(2) <= r * r
    }
    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> impl Fn(f32, f32) -> bool {
        move |x, y| x >= x0 && x < x1 && y >= y0 && y < y1
    }
    fn assert_close(c: &Circle, cx: f32, cy: f32) {
        assert!((c.cx - cx).abs() < 1.0 && (c.cy - cy).abs() < 1.0, "{c:?} attendu ({cx},{cy})");
    }

    #[test]
    fn rond_seul_detecte() {
        let img = page(300, 300, &[&disk(120.0, 140.0, 25.0)]);
        assert_close(&detect_circle_roi_advanced(&img, 0, 0, 300, 10.0).expect("rond"), 120.0, 140.0);
    }
    #[test]
    fn petit_rond_accepte_malgre_la_discretisation() {
        let img = page(300, 300, &[&disk(150.3, 149.7, 11.0)]);
        assert!(detect_circle_roi_advanced(&img, 0, 0, 300, 10.0).is_some());
    }
    #[test]
    fn rond_et_rectangle_noir_dans_le_meme_coin() {
        let img = page(300, 300, &[&disk(100.0, 100.0, 25.0), &rect(200.0, 40.0, 260.0, 70.0)]);
        assert_close(&detect_circle_roi_advanced(&img, 0, 0, 300, 10.0).expect("rond"), 100.0, 100.0);
    }
    #[test]
    fn rond_et_carre_noir_plus_grand() {
        let img = page(300, 300, &[&disk(80.0, 220.0, 20.0), &rect(150.0, 50.0, 230.0, 130.0)]);
        assert_close(&detect_circle_roi_advanced(&img, 0, 0, 300, 10.0).expect("rond"), 80.0, 220.0);
    }
    #[test]
    fn rond_coupe_par_le_bord_de_la_zone_rejete() {
        let img = page(300, 300, &[&disk(10.0, 150.0, 30.0)]);
        assert!(detect_circle_roi_advanced(&img, 0, 0, 300, 10.0).is_none());
    }
    #[test]
    fn beaucoup_de_taches_ne_bloquent_pas() {
        let mut shapes: Vec<Box<dyn Fn(f32, f32) -> bool>> = Vec::new();
        for i in 0..20 { for j in 0..20 {
            let (x, y) = (5.0 + i as f32 * 9.0, 5.0 + j as f32 * 9.0);
            shapes.push(Box::new(rect(x, y, x + 4.0, y + 4.0)));
        }}
        shapes.push(Box::new(disk(240.0, 240.0, 25.0)));
        let refs: Vec<&dyn Fn(f32, f32) -> bool> = shapes.iter().map(|b| b.as_ref()).collect();
        let img = page(300, 300, &refs);
        assert_close(&detect_circle_roi_advanced(&img, 0, 0, 300, 10.0).expect("rond"), 240.0, 240.0);
    }
    #[test]
    fn coin_bas_droit_coordonnees_page() {
        let img = page(1400, 2000, &[&disk(1330.0, 1925.0, 22.0)]);
        let f = detect_circles_in_four_corners_advanced(&img, 300, 10.0);
        assert!(f[0].is_none() && f[1].is_none() && f[2].is_none());
        assert_close(f[3].as_ref().unwrap(), 1330.0, 1925.0);
    }
}
