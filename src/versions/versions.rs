//! Copies reconstruites (T25e, ADR 0005) : une copie imprimée dans une version B, C… est reconstruite dans la mise
//! en page de la version A juste après l'alignement, pour que tout ce qui suit (découpe des zones, détection des QCM,
//! correction, exports) fonctionne comme avec une seule version.
//!
//! **Même algorithme que le front** (`corrigeExamFront`, `src/main/webapp/app/scanexam/typst/version-rebuild.ts`) ;
//! les deux implémentations passent les mêmes cas (`tests/fixtures/version-rebuild-cases.json`, copie du fichier
//! `typst/fixtures/version-rebuild-cases.json` du front).
//!
//! Pour chaque page p de la copie reconstruite :
//!  - fond : la page p du modèle (version A, vide) ;
//!  - chaque partie d'une question placée page p dans la version A est découpée à sa place dans la version de la copie
//!    (éventuellement sur une autre page) et collée à sa place dans A (même taille, vérifiée par le front) ;
//!  - un QCM garde le fond de A (choix dans l'ordre canonique) et reçoit chaque case de la copie à la place de la case
//!    de même lettre canonique, avec une marge de 2 pt ;
//!  - les zones d'identité et la bande du code de page sont recopiées telles quelles.

use image::{DynamicImage, Rgba, RgbaImage};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::pagecodes::pagecodes::{parse_page_code, read_page_code, EditionInfo, PageCode};
use crate::sqliteutils::sqliteutils::{align_pages, read_align_page, save_image_to_db_webp_marked, ImageState};

/// Positions des zones : 1/100000 de la page (comme les `Zone` de l'examen).
const SCALE: f64 = 100000.0;
const PAGE_WIDTH_PT: f64 = 595.28;
const PAGE_WIDTH_MM: f64 = 210.0;
const PAGE_HEIGHT_MM: f64 = 297.0;
pub const BOX_MARGIN_PT: f64 = 2.0;
const CODE_BAND_HALF_WIDTH_MM: f64 = 32.0;
const CODE_BAND_CENTER_MM: f64 = 9.0;
const CODE_BAND_HALF_HEIGHT_MM: f64 = 6.5;

/// Zone au format du front (`IZone`).
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize, Serialize)]
pub struct Zone {
    #[serde(rename = "pageNumber", default)]
    pub page_number: Option<f64>,
    #[serde(rename = "xInit", default)]
    pub x_init: Option<f64>,
    #[serde(rename = "yInit", default)]
    pub y_init: Option<f64>,
    #[serde(default)]
    pub width: Option<f64>,
    #[serde(default)]
    pub height: Option<f64>,
}

impl Zone {
    fn page(&self) -> usize {
        self.page_number.unwrap_or(1.0).max(1.0) as usize
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LayoutPart {
    pub key: String,
    pub part: u32,
    pub zone: Zone,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LayoutBox {
    pub key: String,
    pub part: u32,
    pub letter: String,
    pub zone: Zone,
}

/// Mise en page d'une version (format 1), écrite par le front (`VersionLayout`).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VersionLayout {
    pub format: u32,
    pub version: u32,
    pub label: String,
    pub pages: u32,
    pub parts: Vec<LayoutPart>,
    pub boxes: Vec<LayoutBox>,
    #[serde(default)]
    pub identity: BTreeMap<String, Zone>,
}

/// Version d'un examen transmise par le back dans `ScanRequestDto.versions`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ExamVersion {
    pub index: u32,
    pub label: String,
    pub layout: VersionLayout,
}

/// Rectangle en pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct PixelRect {
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
}

/// Arrondi de `Math.round` (JavaScript) : les deux implémentations donnent les mêmes pixels.
fn js_round(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

/// Version de la copie `copy` (à partir de 1) dans un tirage de `versions` versions : tour à tour.
pub fn version_of_copy(copy: u32, versions: u32) -> u32 {
    let k = versions.max(1) as i64;
    ((copy as i64 - 1).rem_euclid(k)) as u32
}

pub fn zone_to_pixels(zone: &Zone, width: u32, height: u32, margin_pt: f64) -> PixelRect {
    let (w, h) = (width as f64, height as f64);
    let m = (margin_pt / PAGE_WIDTH_PT) * w;
    PixelRect {
        x: js_round((zone.x_init.unwrap_or(0.0) / SCALE) * w - m),
        y: js_round((zone.y_init.unwrap_or(0.0) / SCALE) * h - m),
        w: js_round((zone.width.unwrap_or(0.0) / SCALE) * w + 2.0 * m),
        h: js_round((zone.height.unwrap_or(0.0) / SCALE) * h + 2.0 * m),
    }
}

/// Bande du code de page en bas de page (code et libellé « B · 3 · p. 2 »).
pub fn code_band(width: u32, height: u32) -> PixelRect {
    let px = width as f64 / PAGE_WIDTH_MM;
    let py = height as f64 / PAGE_HEIGHT_MM;
    let w = 2.0 * CODE_BAND_HALF_WIDTH_MM * px;
    let h = 2.0 * CODE_BAND_HALF_HEIGHT_MM * py;
    PixelRect {
        x: js_round(width as f64 / 2.0 - w / 2.0),
        y: js_round(height as f64 - (CODE_BAND_CENTER_MM + CODE_BAND_HALF_HEIGHT_MM) * py),
        w: js_round(w),
        h: js_round(h),
    }
}

/// Copie `from` de `src` dans `to` de `dst` (pixel le plus proche) ; ce qui sort des pages est ignoré.
pub fn blit(src: &RgbaImage, from: PixelRect, dst: &mut RgbaImage, to: PixelRect) {
    if from.w <= 0 || from.h <= 0 || to.w <= 0 || to.h <= 0 {
        return;
    }
    let (sw, sh) = (src.width() as i64, src.height() as i64);
    let (dw, dh) = (dst.width() as i64, dst.height() as i64);
    for y in 0..to.h {
        let dy = to.y + y;
        if dy < 0 || dy >= dh {
            continue;
        }
        let sy = from.y + (from.h - 1).min((y * from.h) / to.h);
        if sy < 0 || sy >= sh {
            continue;
        }
        for x in 0..to.w {
            let dx = to.x + x;
            if dx < 0 || dx >= dw {
                continue;
            }
            let sx = from.x + (from.w - 1).min((x * from.w) / to.w);
            if sx < 0 || sx >= sw {
                continue;
            }
            let p = *src.get_pixel(sx as u32, sy as u32);
            dst.put_pixel(dx as u32, dy as u32, p);
        }
    }
}

/// Écart signalé pendant la reconstruction (clés `scanexam.versions.issue.*` du front).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RebuildIssue {
    pub key: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub question: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub part: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub letter: Option<String>,
}

/// Reconstruit les pages d'une copie imprimée dans la version `layout`, dans la mise en page `canonical` (A).
pub fn rebuild_copy(
    copy: &[RgbaImage],
    templates: &[RgbaImage],
    canonical: &VersionLayout,
    layout: &VersionLayout,
) -> (Vec<RgbaImage>, Vec<RebuildIssue>) {
    let mut issues = Vec::new();
    let mut pages: Vec<RgbaImage> = templates.to_vec();
    let qcm: HashSet<&str> = canonical.boxes.iter().map(|b| b.key.as_str()).collect();
    let part_of: HashMap<(String, u32), Zone> = layout.parts.iter().map(|p| ((p.key.clone(), p.part), p.zone)).collect();
    let box_of: HashMap<(String, u32, String), Zone> =
        layout.boxes.iter().map(|b| ((b.key.clone(), b.part, b.letter.clone()), b.zone)).collect();

    let paste = |pages: &mut Vec<RgbaImage>, zone_v: &Zone, zone_a: &Zone, margin: f64| -> bool {
        let src = match copy.get(zone_v.page() - 1) {
            Some(s) => s,
            None => return false,
        };
        let dst = match pages.get_mut(zone_a.page() - 1) {
            Some(d) => d,
            None => return false,
        };
        let from = zone_to_pixels(zone_v, src.width(), src.height(), margin);
        let to = zone_to_pixels(zone_a, dst.width(), dst.height(), margin);
        blit(src, from, dst, to);
        true
    };

    for part in &canonical.parts {
        if qcm.contains(part.key.as_str()) {
            continue;
        }
        match part_of.get(&(part.key.clone(), part.part)) {
            None => issues.push(RebuildIssue { key: "missingPart".into(), question: Some(part.key.clone()), part: Some(part.part), letter: None }),
            Some(zone) => {
                if !paste(&mut pages, zone, &part.zone, 0.0) {
                    issues.push(RebuildIssue { key: "missingPage".into(), question: Some(part.key.clone()), part: Some(part.part), letter: None });
                }
            }
        }
    }
    for b in &canonical.boxes {
        match box_of.get(&(b.key.clone(), b.part, b.letter.clone())) {
            None => issues.push(RebuildIssue {
                key: "missingBox".into(),
                question: Some(b.key.clone()),
                part: Some(b.part),
                letter: Some(b.letter.clone()),
            }),
            Some(zone) => {
                if !paste(&mut pages, zone, &b.zone, BOX_MARGIN_PT) {
                    issues.push(RebuildIssue {
                        key: "missingPage".into(),
                        question: Some(b.key.clone()),
                        part: Some(b.part),
                        letter: Some(b.letter.clone()),
                    });
                }
            }
        }
    }
    for (role, zone_a) in &canonical.identity {
        if let Some(zone_v) = layout.identity.get(role) {
            paste(&mut pages, zone_v, zone_a, 0.0);
        }
    }
    // le code et son libellé restent ceux de la page imprimée
    for (i, page) in pages.iter_mut().enumerate() {
        if let Some(src) = copy.get(i) {
            let from = code_band(src.width(), src.height());
            let to = code_band(page.width(), page.height());
            blit(src, from, page, to);
        }
    }
    (pages, issues)
}

/// Copie du scan : pages consécutives, comme CorrectExam découpe les copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct CopySlot {
    pub first_page: u32,
    pub pages: u32,
}

pub fn copy_slots(total_pages: u32, pages_per_copy: u32) -> Vec<CopySlot> {
    let mut slots = Vec::new();
    if pages_per_copy == 0 {
        return slots;
    }
    let mut first = 1;
    while first <= total_pages {
        slots.push(CopySlot { first_page: first, pages: pages_per_copy.min(total_pages - first + 1) });
        first += pages_per_copy;
    }
    slots
}

/// Où mène un code de page lu sur une copie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyVersion {
    Version { edition: u64, copy: u32, version: u32 },
    /// Aucun code d'un tirage de cet examen.
    Unknown,
    /// Le code lu n'est pas à sa place : le scan n'est pas dans l'ordre des copies.
    Disordered { page: u32, code: PageCode },
}

pub fn version_from_code(code: Option<PageCode>, page_in_copy: u32, editions: &[EditionInfo]) -> CopyVersion {
    let code = match code {
        Some(c) => c,
        None => return CopyVersion::Unknown,
    };
    let edition = match editions.iter().find(|e| e.id == code.edition) {
        Some(e) if code.copy >= 1 && code.copy <= e.copies => e,
        _ => return CopyVersion::Unknown,
    };
    if code.page != page_in_copy {
        return CopyVersion::Disordered { page: page_in_copy, code };
    }
    CopyVersion::Version { edition: code.edition, copy: code.copy, version: version_of_copy(code.copy, edition.versions) }
}

/// Bilan de la reconstruction, envoyé au front dans le statut `finished` (`{"versions": …}`), même forme que le
/// `RebuildReport` du front (champs en snake_case).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RebuildSummary {
    pub copies: usize,
    pub version_a: usize,
    pub already_rebuilt: usize,
    pub rebuilt: Vec<RebuiltCopy>,
    pub not_rebuilt: Vec<NotRebuilt>,
    /// Version de chaque copie dont la version est connue (A, reconstruites, déjà reconstruites), pour les
    /// statistiques par version (T25e6b) ; le front l'enregistre dans le back.
    #[serde(default)]
    pub copy_versions: Vec<CopyVersionEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopyVersionEntry {
    pub first_page: u32,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub edition: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub copy: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RebuiltCopy {
    pub first_page: u32,
    pub edition: u64,
    pub copy: u32,
    pub label: String,
    pub issues: Vec<RebuildIssue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NotRebuilt {
    pub first_page: u32,
    /// `unknown`, `disordered`, `incomplete`, `partial`, `noLayout`, `failed` (comme le front).
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub detail: Option<String>,
}

// ---------------------------------------------------------------------------------------------------------------
// Passe de reconstruction sur le cache SQLite (même déroulé que `VersionRebuildService` du front)

/// Vrai quand un tirage de l'examen a plusieurs versions et que le back a transmis les mises en page.
pub fn applies(editions: &[EditionInfo], versions: &[ExamVersion]) -> bool {
    editions.iter().any(|e| e.versions > 1) && versions.iter().any(|v| v.index == 0)
}

/// Reconstruit les copies des versions B, C… du cache aligné : copies de N pages (N = pages de la version A),
/// version lue sur le premier code de page lisible, pages reconstruites marquées, copies non reconstruites signalées.
pub fn rebuild_cache(
    conn: &Connection,
    editions: &[EditionInfo],
    versions: &[ExamVersion],
    templates: &HashMap<u32, DynamicImage>,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<RebuildSummary, Box<dyn std::error::Error>> {
    let canonical = versions.iter().find(|v| v.index == 0).ok_or("version A absente")?;
    let n = canonical.layout.pages;
    let template_pages: Vec<RgbaImage> = (1..=n)
        .map(|p| templates.get(&p).map(|t| t.to_rgba8()).ok_or_else(|| format!("page {} du modèle absente", p)))
        .collect::<Result<_, _>>()?;
    let total = align_pages(conn)?.into_iter().max().unwrap_or(0).max(0) as u32;
    let slots = copy_slots(total, n);
    let mut summary = RebuildSummary { copies: slots.len(), ..Default::default() };
    for (i, slot) in slots.iter().enumerate() {
        if let Err(e) = rebuild_slot(conn, slot, n, editions, versions, canonical, &template_pages, &mut summary) {
            eprintln!("    ! Copie commençant page {} non reconstruite : {}", slot.first_page, e);
            summary.not_rebuilt.push(NotRebuilt { first_page: slot.first_page, reason: "failed".into(), detail: Some(e.to_string()) });
        }
        on_progress(i + 1, slots.len());
    }
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn rebuild_slot(
    conn: &Connection,
    slot: &CopySlot,
    n: u32,
    editions: &[EditionInfo],
    versions: &[ExamVersion],
    canonical: &ExamVersion,
    templates: &[RgbaImage],
    summary: &mut RebuildSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let not = |reason: &str, detail: Option<String>| NotRebuilt { first_page: slot.first_page, reason: reason.into(), detail };
    if slot.pages < n {
        summary.not_rebuilt.push(not("incomplete", None));
        return Ok(());
    }
    let mut images = Vec::with_capacity(n as usize);
    let mut marked = 0;
    let mut mark: Option<String> = None;
    for p in slot.first_page..slot.first_page + n {
        let (img, rebuilt) = read_align_page(conn, p as i32)?.ok_or_else(|| format!("page alignée {} absente", p))?;
        if rebuilt.is_some() {
            marked += 1;
            mark = rebuilt;
        }
        images.push(img);
    }
    if marked == images.len() {
        summary.already_rebuilt += 1;
        if let Some(label) = mark {
            summary.copy_versions.push(CopyVersionEntry { first_page: slot.first_page, label, edition: None, copy: None });
        }
        return Ok(());
    }
    if marked > 0 {
        summary.not_rebuilt.push(not("partial", None));
        return Ok(());
    }
    let mut found = CopyVersion::Unknown;
    for (j, img) in images.iter().enumerate() {
        let read = read_page_code(img, slot.first_page + j as u32);
        if let Some(code) = read.text.as_deref().and_then(parse_page_code) {
            found = version_from_code(Some(code), j as u32 + 1, editions);
            break;
        }
    }
    let (edition, copy, version) = match found {
        CopyVersion::Unknown => {
            summary.not_rebuilt.push(not("unknown", None));
            return Ok(());
        }
        CopyVersion::Disordered { page, code } => {
            summary.not_rebuilt.push(not("disordered", Some(format!("{}|{}|{} @ {}", code.edition, code.copy, code.page, page))));
            return Ok(());
        }
        CopyVersion::Version { edition, copy, version } => (edition, copy, version),
    };
    if version == 0 {
        summary.version_a += 1;
        let label = versions.iter().find(|v| v.index == 0).map(|v| v.label.clone()).unwrap_or_else(|| "A".into());
        summary.copy_versions.push(CopyVersionEntry { first_page: slot.first_page, label, edition: Some(edition), copy: Some(copy) });
        return Ok(());
    }
    let Some(layout) = versions.iter().find(|v| v.index == version) else {
        summary.not_rebuilt.push(not("noLayout", Some(version.to_string())));
        return Ok(());
    };
    let copy_pages: Vec<RgbaImage> = images.iter().map(|i| i.to_rgba8()).collect();
    let (mut pages, issues) = rebuild_copy(&copy_pages, templates, &canonical.layout, &layout.layout);
    for (j, page) in pages.iter_mut().enumerate() {
        draw_mention(page, &mention(&layout.label));
        let img = DynamicImage::ImageRgba8(page.clone());
        save_image_to_db_webp_marked(conn, &img, &((slot.first_page + j as u32) as i32), ImageState::Align, Some(layout.label.clone()))?;
    }
    println!("    -> Copie {} (tirage {}) reconstruite depuis la version {}", copy, edition, layout.label);
    summary.copy_versions.push(CopyVersionEntry {
        first_page: slot.first_page,
        label: layout.label.clone(),
        edition: Some(edition),
        copy: Some(copy),
    });
    summary.rebuilt.push(RebuiltCopy { first_page: slot.first_page, edition, copy, label: layout.label.clone(), issues });
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Mention « VERSION B - PAGE RECONSTRUITE » : police matricielle 5 × 7 (majuscules sans accent), pour ne dépendre
// d'aucun fichier de police dans le conteneur.

fn glyph(c: char) -> [u8; 7] {
    match c {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        _ => [0x00; 7],
    }
}

/// Écrit `text` (majuscules) en gris dans la marge haute, à (22 mm, 7 mm) comme le front, hauteur ≈ 3 mm.
pub fn draw_mention(page: &mut RgbaImage, text: &str) {
    let mm = page.width() as f64 / PAGE_WIDTH_MM;
    let unit = ((3.0 * mm) / 7.0).round().max(1.0) as u32;
    let x0 = (22.0 * mm).round() as u32;
    let y0 = ((7.0 * mm).round() as u32).saturating_sub(7 * unit);
    let colour = Rgba([85, 85, 85, 255]);
    for (i, c) in text.to_uppercase().chars().enumerate() {
        let rows = glyph(c);
        let gx = x0 + i as u32 * 6 * unit;
        for (ry, row) in rows.iter().enumerate() {
            for rx in 0..5u32 {
                if row & (0x10 >> rx) != 0 {
                    for dy in 0..unit {
                        for dx in 0..unit {
                            let (x, y) = (gx + rx * unit + dx, y0 + ry as u32 * unit + dy);
                            if x < page.width() && y < page.height() {
                                page.put_pixel(x, y, colour);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn mention(label: &str) -> String {
    format!("VERSION {} - PAGE RECONSTRUITE", label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CASES: &str = include_str!("../../tests/fixtures/version-rebuild-cases.json");

    fn cases() -> Value {
        serde_json::from_str(CASES).unwrap()
    }

    fn rect(v: &Value) -> PixelRect {
        serde_json::from_value(v.clone()).unwrap()
    }

    #[test]
    fn shared_zone_to_pixels() {
        for c in cases()["zoneToPixels"].as_array().unwrap() {
            let zone: Zone = serde_json::from_value(c["zone"].clone()).unwrap();
            let got = zone_to_pixels(&zone, c["width"].as_u64().unwrap() as u32, c["height"].as_u64().unwrap() as u32, c["marginPt"].as_f64().unwrap());
            assert_eq!(got, rect(&c["expected"]), "{}", c);
        }
    }

    #[test]
    fn shared_code_band() {
        for c in cases()["codeBand"].as_array().unwrap() {
            assert_eq!(code_band(c["width"].as_u64().unwrap() as u32, c["height"].as_u64().unwrap() as u32), rect(&c["expected"]));
        }
    }

    #[test]
    fn shared_version_of_copy_and_slots() {
        for c in cases()["versionOfCopy"].as_array().unwrap() {
            let v: Vec<u32> = serde_json::from_value(c.clone()).unwrap();
            assert_eq!(version_of_copy(v[0], v[1]), v[2]);
        }
        for c in cases()["copySlots"].as_array().unwrap() {
            let expected: Vec<(u32, u32)> = serde_json::from_value(c["expected"].clone()).unwrap();
            let got: Vec<(u32, u32)> = copy_slots(c["total"].as_u64().unwrap() as u32, c["pagesPerCopy"].as_u64().unwrap() as u32)
                .iter()
                .map(|s| (s.first_page, s.pages))
                .collect();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn shared_version_from_code() {
        for c in cases()["versionFromCode"].as_array().unwrap() {
            let code: Option<PageCode> = if c["code"].is_null() {
                None
            } else {
                Some(PageCode {
                    edition: c["code"]["edition"].as_u64().unwrap(),
                    copy: c["code"]["copy"].as_u64().unwrap() as u32,
                    page: c["code"]["page"].as_u64().unwrap() as u32,
                })
            };
            let editions: Vec<EditionInfo> = serde_json::from_value(c["editions"].clone()).unwrap();
            let got = version_from_code(code, c["page"].as_u64().unwrap() as u32, &editions);
            let expected = &c["expected"];
            match got {
                CopyVersion::Version { version, .. } => {
                    assert_eq!(expected["kind"], "version", "{}", c);
                    assert_eq!(expected["version"].as_u64().unwrap() as u32, version, "{}", c);
                }
                CopyVersion::Unknown => assert_eq!(expected["kind"], "unknown", "{}", c),
                CopyVersion::Disordered { .. } => assert_eq!(expected["kind"], "disordered", "{}", c),
            }
        }
    }

    #[test]
    fn shared_rebuild() {
        for c in cases()["rebuild"].as_array().unwrap() {
            let (w, h) = (c["width"].as_u64().unwrap() as u32, c["height"].as_u64().unwrap() as u32);
            let n = c["pages"].as_u64().unwrap() as usize;
            let level = |v: &Value| v.as_u64().unwrap() as u8;
            let plain = |l: u8| RgbaImage::from_pixel(w, h, Rgba([l, l, l, l]));
            let templates: Vec<RgbaImage> = (0..n).map(|_| plain(level(&c["template"]))).collect();
            let mut copy: Vec<RgbaImage> = (0..n).map(|_| plain(level(&c["paper"]))).collect();
            for ink in c["ink"].as_array().unwrap() {
                let r = rect(ink);
                let page = &mut copy[ink["page"].as_u64().unwrap() as usize - 1];
                for y in r.y..r.y + r.h {
                    for x in r.x..r.x + r.w {
                        page.put_pixel(x as u32, y as u32, Rgba([0, 0, 0, 0]));
                    }
                }
            }
            let canonical: VersionLayout = serde_json::from_value(c["canonical"].clone()).unwrap();
            let layout: VersionLayout = serde_json::from_value(c["layout"].clone()).unwrap();
            let (pages, issues) = rebuild_copy(&copy, &templates, &canonical, &layout);
            let expected_issues: Vec<RebuildIssue> = serde_json::from_value(c["issues"].clone()).unwrap();
            assert_eq!(issues, expected_issues, "{}", c["name"]);
            for p in c["probes"].as_array().unwrap() {
                let page = &pages[p["page"].as_u64().unwrap() as usize - 1];
                let got = page.get_pixel(p["x"].as_u64().unwrap() as u32, p["y"].as_u64().unwrap() as u32)[0];
                assert_eq!(got as u64, p["value"].as_u64().unwrap(), "{} {}", c["name"], p);
            }
        }
    }

    /// Passe complète sur un vrai cache SQLite : copie 1 en A, copie 2 en B (code sur sa page 2), seconde passe sans
    /// effet. Le code de page est un vrai Data Matrix, dessiné comme à l'impression.
    #[test]
    fn rebuild_pass_on_a_sqlite_cache() {
        use crate::sqliteutils::sqliteutils::{create_connection, read_align_page, save_image_to_db_webp};
        use rxing::{BarcodeFormat, Writer};
        let (w, h) = (1240u32, 1754u32);
        let mm = w as f64 / 210.0;
        let with_code = |text: &str| {
            let mut img = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));
            let matrix = rxing::datamatrix::DataMatrixWriter.encode(text, &BarcodeFormat::DATA_MATRIX, 0, 0).unwrap();
            let module = ((11.0 * mm) / matrix.width() as f64).floor() as u32;
            let (x0, y0) = (w / 2 - matrix.width() * module / 2, h - (9.0 * mm) as u32 - matrix.height() * module / 2);
            for y in 0..matrix.height() {
                for x in 0..matrix.width() {
                    if matrix.get(x, y) {
                        for dy in 0..module {
                            for dx in 0..module {
                                img.put_pixel(x0 + x * module + dx, y0 + y * module + dy, Rgba([0, 0, 0, 255]));
                            }
                        }
                    }
                }
            }
            img
        };
        let zone = |page: f64, y: f64| Zone { page_number: Some(page), x_init: Some(10000.0), y_init: Some(y), width: Some(50000.0), height: Some(10000.0) };
        let layout = |version: u32, label: &str, page: f64, y: f64| VersionLayout {
            format: 1,
            version,
            label: label.into(),
            pages: 2,
            parts: vec![LayoutPart { key: "q1".into(), part: 1, zone: zone(page, y) }],
            boxes: vec![],
            identity: BTreeMap::new(),
        };
        let versions = vec![
            ExamVersion { index: 0, label: "A".into(), layout: layout(0, "A", 1.0, 20000.0) },
            ExamVersion { index: 1, label: "B".into(), layout: layout(1, "B", 2.0, 40000.0) },
        ];
        let editions = [EditionInfo { id: 41, copies: 30, pages_per_copy: 2, versions: 2 }];
        let exam_id = 900_000 + std::process::id() % 1000;
        let _ = std::fs::remove_file(format!("/tmp/{}.sqlite3", exam_id));
        let conn = create_connection(exam_id).unwrap();
        // copie 1 (A) : pages 1-2 ; copie 2 (B) : pages 3-4, le code de sa page 1 est illisible (page blanche)
        let mut b2 = with_code("CE1|41|2|2");
        let r = zone_to_pixels(&zone(2.0, 40000.0), w, h, 0.0);
        for y in r.y..r.y + 20 {
            for x in r.x..r.x + 20 {
                b2.put_pixel(x as u32, y as u32, Rgba([0, 0, 0, 255]));
            }
        }
        let pages = [with_code("CE1|41|1|1"), with_code("CE1|41|1|2"), RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255])), b2];
        for (i, p) in pages.iter().enumerate() {
            save_image_to_db_webp(&conn, &DynamicImage::ImageRgba8(p.clone()), &(i as i32 + 1), ImageState::Align).unwrap();
        }
        let templates: HashMap<u32, DynamicImage> =
            (1..=2).map(|p| (p, DynamicImage::ImageRgba8(RgbaImage::from_pixel(w, h, Rgba([240, 240, 240, 255]))))).collect();
        assert!(applies(&editions, &versions));

        let summary = rebuild_cache(&conn, &editions, &versions, &templates, |_, _| {}).unwrap();
        assert_eq!(summary.copies, 2);
        assert_eq!(summary.version_a, 1);
        assert_eq!(summary.rebuilt.len(), 1, "{:?}", summary);
        assert_eq!(
            summary.copy_versions.iter().map(|c| (c.first_page, c.label.as_str(), c.copy)).collect::<Vec<_>>(),
            vec![(1, "A", Some(1)), (3, "B", Some(2))]
        );
        assert_eq!((summary.rebuilt[0].first_page, summary.rebuilt[0].copy, summary.rebuilt[0].label.as_str()), (3, 2, "B"));
        // page 3 reconstruite : l'encre de la page 2 de B est à la place de q1 dans A (page 1 de la copie)
        let (p3, mark) = read_align_page(&conn, 3).unwrap().unwrap();
        assert_eq!(mark.as_deref(), Some("B"));
        let a = zone_to_pixels(&zone(1.0, 20000.0), w, h, 0.0);
        let p3 = p3.to_rgba8();
        assert!(p3.get_pixel(a.x as u32 + 5, a.y as u32 + 5)[0] < 64);
        assert!(p3.get_pixel(a.x as u32 + 40, a.y as u32 + 40)[0] > 200);
        assert!(read_align_page(&conn, 1).unwrap().unwrap().1.is_none());

        let again = rebuild_cache(&conn, &editions, &versions, &templates, |_, _| {}).unwrap();
        assert_eq!((again.already_rebuilt, again.version_a, again.rebuilt.len()), (1, 1, 0));
        assert_eq!(
            again.copy_versions.iter().map(|c| (c.first_page, c.label.as_str(), c.copy)).collect::<Vec<_>>(),
            vec![(1, "A", Some(1)), (3, "B", None)]
        );
        let _ = std::fs::remove_file(format!("/tmp/{}.sqlite3", exam_id));
    }

    /// Le bilan envoyé au front a la forme que le front lit (`parseServerVersions`, même fichier de cas).
    #[test]
    fn shared_finished_details() {
        let c = cases();
        let details: Value = serde_json::from_str(c["finishedDetails"]["details"].as_str().unwrap()).unwrap();
        let summary: RebuildSummary = serde_json::from_value(details["versions"].clone()).unwrap();
        assert_eq!(serde_json::to_value(&summary).unwrap(), details["versions"]);
        assert_eq!(summary.rebuilt[0].label, "B");
    }

    #[test]
    fn mention_is_written_in_the_top_margin() {
        let mut page = RgbaImage::from_pixel(1240, 1754, Rgba([255, 255, 255, 255]));
        draw_mention(&mut page, &mention("B"));
        let dark = page.enumerate_pixels().filter(|(_, _, p)| p[0] < 128).count();
        assert!(dark > 500, "{}", dark);
        // rien sous 10 mm : titre, repères et zones intacts
        let mm = 1240.0 / 210.0;
        assert!(page.enumerate_pixels().filter(|(_, y, p)| *y as f64 > 10.0 * mm && p[0] < 128).count() == 0);
    }
}
