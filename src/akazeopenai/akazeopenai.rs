
use image::GrayImage;
use image::{ GenericImageView};
use rand::seq::{IndexedRandom};

use crate::square::square::{ Point, SimilarityTransform};
use image::{RgbaImage, Rgba};



#[derive(Clone)]
pub struct Match {
    pub p_src: Point,
    pub p_dst: Point,
}

#[derive(Clone, Debug)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub desc: [u8; 32], // BRIEF 256 bits
}

// --- Hamming distance ---

fn hamming(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn match_descriptors(a: &[KeyPoint], b: &[KeyPoint]) -> Vec<Match> {
    let mut matches = Vec::new();

    for ka in a {
        let mut best = None;
        let mut best_dist = u32::MAX;

        for kb in b {
            let d = hamming(&ka.desc, &kb.desc);
            if d < best_dist {
                best_dist = d;
                best = Some(kb);
            }
        }

        if let Some(kb) = best {
            matches.push(Match {
                p_src: Point { x: ka.x, y: ka.y },
                p_dst: Point { x: kb.x, y: kb.y },
            });
        }
    }
    matches
}

fn compute_brief(image: &GrayImage, kp: &KeyPoint) -> [u8; 32] {
    let mut desc = [0u8; 32];

    for i in 0..256 {
        let dx1 = (i % 16) as i32 - 8;
        let dy1 = (i / 16) as i32 - 8;
        let dx2 = dx1 + 1;
        let dy2 = dy1 + 1;

        let x1 = (kp.x as i32 + dx1).clamp(0, image.width() as i32 - 1) as u32;
        let y1 = (kp.y as i32 + dy1).clamp(0, image.height() as i32 - 1) as u32;
        let x2 = (kp.x as i32 + dx2).clamp(0, image.width() as i32 - 1) as u32;
        let y2 = (kp.y as i32 + dy2).clamp(0, image.height() as i32 - 1) as u32;

        let v1 = image.get_pixel(x1, y1)[0];
        let v2 = image.get_pixel(x2, y2)[0];

        if v1 < v2 {
            desc[i / 8] |= 1 << (i % 8);
        }
    }

    desc
}


fn estimate_similarity(p1: &Match, p2: &Match) -> Option<SimilarityTransform> {
    let dx1 = p2.p_src.x - p1.p_src.x;
    let dy1 = p2.p_src.y - p1.p_src.y;
    let dx2 = p2.p_dst.x - p1.p_dst.x;
    let dy2 = p2.p_dst.y - p1.p_dst.y;

    let d1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let d2 = (dx2 * dx2 + dy2 * dy2).sqrt();
    if d1 < 1e-3 { return None; }

    let scale = d2 / d1;
    let rotation = dy2.atan2(dx2) - dy1.atan2(dx1);

    let cos_t = rotation.cos();
    let sin_t = rotation.sin();

    let tx = p1.p_dst.x - scale * (cos_t * p1.p_src.x - sin_t * p1.p_src.y);
    let ty = p1.p_dst.y - scale * (sin_t * p1.p_src.x + cos_t * p1.p_src.y);

    Some(SimilarityTransform { scale, rotation, tx, ty })
}

fn reprojection_error(t: &SimilarityTransform, m: &Match) -> f32 {
    let cos_t = t.rotation.cos();
    let sin_t = t.rotation.sin();

    let x = t.scale * (cos_t * m.p_src.x - sin_t * m.p_src.y) + t.tx;
    let y = t.scale * (sin_t * m.p_src.x + cos_t * m.p_src.y) + t.ty;

    ((x - m.p_dst.x).powi(2) + (y - m.p_dst.y).powi(2)).sqrt()
}

fn ransac(matches: &[Match], iters: usize, thresh: f32)
    -> Option<SimilarityTransform> {

    if matches.len() < 2 { return None; }

    let mut rng = rand::thread_rng();
    let mut best = None;
    let mut best_count = 0;

    for _ in 0..iters {
        let pair: Vec<_> = matches.choose_multiple(&mut rng, 2).collect();
        let model = estimate_similarity(pair[0], pair[1])?;

        let inliers = matches.iter()
            .filter(|m| reprojection_error(&model, m) < thresh)
            .count();

        if inliers > best_count {
            best_count = inliers;
            best = Some(model);
        }
    }

    best
}


pub fn akaze_ransac_per_corner(
    img_a: &GrayImage,
    img_b: &GrayImage,
    corners: &[(usize, usize)],
    roi: usize,
) -> Vec<Option<SimilarityTransform>> {

    let mut results = Vec::new();

    for &(x, y) in corners {
        let roi_a = img_a.view(x as u32, y as u32, roi as u32, roi as u32).to_image();
        let roi_b = img_b.view(x as u32, y as u32, roi as u32, roi as u32).to_image();

        let mut kpa = Vec::new();
        let mut kpb = Vec::new();

        for (px, py, _) in roi_a.enumerate_pixels().step_by(20) {
            kpa.push(KeyPoint { x: px as f32, y: py as f32, desc: [0; 32] });
        }
        for (px, py, _) in roi_b.enumerate_pixels().step_by(20) {
            kpb.push(KeyPoint { x: px as f32, y: py as f32, desc: [0; 32] });
        }

        for k in &mut kpa { k.desc = compute_brief(&roi_a, k); }
        for k in &mut kpb { k.desc = compute_brief(&roi_b, k); }

        let matches = match_descriptors(&kpa, &kpb);
        let t = ransac(&matches, 200, 3.0);
        results.push(t);
    }

    results
}
fn compute_brief_descriptor(image: &GrayImage, kp: &KeyPoint) -> [u8; 32] {
    let mut desc = [0u8; 32];
//    let patch_radius = 8;
    for i in 0..256 {
        let (dx1, dy1) = ((i % 16) as i32 - 8, (i / 16) as i32 - 8);
        let (dx2, dy2) = ((i % 16) as i32 - 7, (i / 16) as i32 - 7);
        let x1 = (kp.x as i32 + dx1).clamp(0, image.width() as i32 - 1) as u32;
        let y1 = (kp.y as i32 + dy1).clamp(0, image.height() as i32 - 1) as u32;
        let x2 = (kp.x as i32 + dx2).clamp(0, image.width() as i32 - 1) as u32;
        let y2 = (kp.y as i32 + dy2).clamp(0, image.height() as i32 - 1) as u32;
        let v1 = image.get_pixel(x1, y1)[0];
        let v2 = image.get_pixel(x2, y2)[0];
        if v1 < v2 {
            desc[i / 8] |= 1 << (i % 8);
        }
    }
    desc
}


pub fn fuse_transforms(ts: Vec<SimilarityTransform>) -> Option<SimilarityTransform> {
    if ts.is_empty() { return None; }

    let scale = ts.iter().map(|t| t.scale).sum::<f32>() / ts.len() as f32;
    let sin = ts.iter().map(|t| t.rotation.sin()).sum::<f32>();
    let cos = ts.iter().map(|t| t.rotation.cos()).sum::<f32>();
    let rotation = sin.atan2(cos);

    let tx = ts.iter().map(|t| t.tx).sum::<f32>() / ts.len() as f32;
    let ty = ts.iter().map(|t| t.ty).sum::<f32>() / ts.len() as f32;

    Some(SimilarityTransform { scale, rotation, tx, ty })
}

pub fn apply_transform(
    img: &GrayImage,
    t: &SimilarityTransform,
    w: u32,
    h: u32,
) -> GrayImage {

    let mut out = GrayImage::new(w, h);
    let cos_t = t.rotation.cos();
    let sin_t = t.rotation.sin();

    for y in 0..h {
        for x in 0..w {
            let xf = (x as f32 - t.tx) / t.scale;
            let yf = (y as f32 - t.ty) / t.scale;

            let xs =  cos_t * xf + sin_t * yf;
            let ys = -sin_t * xf + cos_t * yf;

            if xs >= 0.0 && ys >= 0.0 &&
               xs < img.width() as f32 &&
               ys < img.height() as f32 {
                out.put_pixel(x, y, *img.get_pixel(xs as u32, ys as u32));
            }
        }
    }
    out
}


pub fn apply_similarity_transform_rgba(
    img: &RgbaImage,
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

            // --- inverse mapping ---
            let xs = (xf - t.tx) / t.scale;
            let ys = (yf - t.ty) / t.scale;

            let src_x =  cos_t * xs + sin_t * ys;
            let src_y = -sin_t * xs + cos_t * ys;

            if src_x >= 0.0 &&
               src_y >= 0.0 &&
               src_x < img.width() as f32 &&
               src_y < img.height() as f32 {

                let px = img.get_pixel(src_x as u32, src_y as u32);
                out.put_pixel(x, y, *px);
            } else {
                // fond blanc
                out.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
    }

    out
}

