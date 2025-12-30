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


pub fn flood_fill(
    bin: &mut Vec<Vec<u8>>,
    x0: usize,
    y0: usize,
    label: u8,
    w: usize,
    h: usize,
    pixels: &mut Vec<(usize, usize)>,
) {
    let mut stack = vec![(x0, y0)];
    bin[y0][x0] = label;

    while let Some((x, y)) = stack.pop() {
        pixels.push((x, y));

        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x as isize + dx;
            let ny = y as isize + dy;

            if nx >= 0 && ny >= 0 && nx < w as isize && ny < h as isize {
                let nx = nx as usize;
                let ny = ny as usize;
                if bin[ny][nx] == 1 {
                    bin[ny][nx] = label;
                    stack.push((nx, ny));
                }
            }
        }
    }
}

pub fn filter_filled_circles(bin: &Vec<Vec<u8>>, w: usize, h: usize, min_radius: f32) -> Vec<Vec<u8>> {
    let mut bin = bin.clone();
    let mut label: u8 = 2;

    let mut output = vec![vec![0u8; w]; h];

    for y in 0..h {
        for x in 0..w {
            if bin[y][x] == 1 {
                let mut pixels = Vec::new();
                flood_fill(&mut bin, x, y, label, w, h, &mut pixels);

                let mut minx = w;
                let mut maxx = 0;
                let mut miny = h;
                let mut maxy = 0;

                for &(px, py) in &pixels {
                    minx = minx.min(px);
                    maxx = maxx.max(px);
                    miny = miny.min(py);
                    maxy = maxy.max(py);
                }

                let width = (maxx - minx + 1) as f32;
                let height = (maxy - miny + 1) as f32;
                let radius = (width.min(height)) / 2.0;

                let area_pixels = pixels.len() as f32;
                let area_circle = std::f32::consts::PI * radius * radius;
                let fill_ratio = area_pixels / area_circle;

                if radius > min_radius && fill_ratio > 0.7 {
                    // ✔ cercle plein valide → on conserve
                    for &(px, py) in &pixels {
                        output[py][px] = 1;
                    }
                }

                label += 1;
            }
        }
    }

    output
}

pub fn validate_rms(circle: &Circle, points: &[(f32, f32)], max_ratio: f32) -> bool {
    let mut err2 = 0.0;

    for (x, y) in points {
        let d = ((x - circle.cx).powi(2) + (y - circle.cy).powi(2)).sqrt();
        err2 += (d - circle.r).powi(2);
    }

    let rms = (err2 / points.len() as f32).sqrt();
    rms < circle.r * max_ratio
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
    // (Seuil Otsu + morphologie identiques à avant)
    let roi = gray.view(x0, y0, size, size).to_image();
    let t = otsu_threshold(&roi);

    //    println!(" {} ",t);
    let mut bin = vec![vec![0u8; size as usize]; size as usize];
    for y in 0..size {
        for x in 0..size {
            bin[y as usize][x as usize] = if roi.get_pixel(x, y)[0] < t { 1 } else { 0 };
        }
    }

    let bin = dilate(
        &erode(&bin, size as usize, size as usize),
        size as usize,
        size as usize,
    );

    let bin = filter_filled_circles(
        &bin,
        size as usize,
        size as usize,
        min_radius, // rayon min
    );

    let border_pixels = extract_border_points(&bin, size as usize, size as usize);

    let mut points = Vec::new();
    for (x, y) in border_pixels {
        points.push((x as f32 + x0 as f32, y as f32 + y0 as f32));
    }

    let circle = fit_circle_taubin(&points)?;
    if !validate_rms(&circle, &points, 0.05) {
        println!("    ! Avertissement : RMS trop élevé pour le cercle détecté.");
        return None;
    }

    /*    if !validate_radial_gradient(gray, &circle, &points) {
        println!("    ! Avertissement : Gradient radial non valide.");
        // return None;
    } */

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

