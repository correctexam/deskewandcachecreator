//! Codes de page des copies imprimées depuis CorrectExam (T25d3, ADR 0004 du back).
//!
//! Chaque page imprimée porte un Data Matrix `CE1|<tirage>|<copie>|<page>` en bas au centre. Avant l'alignement,
//! le worker lit le code de chaque page du scan, puis traite les pages dans l'ordre attendu par le modèle :
//! copies dans l'ordre du tirage, page retournée redressée, page blanche à la place d'une page manquante,
//! pages en double, étrangères ou illisibles en fin de lot. Même logique que le front
//! (`scanexam/typst/page-codes.ts`, T25d2) : les deux modes d'alignement donnent le même ordre.

use std::collections::{HashMap, HashSet};

use image::{DynamicImage, GrayImage};
use rxing::{BarcodeFormat, DecodeHints};
use serde::{Deserialize, Serialize};

/// Tirage d'un examen, transmis par le back dans `ScanRequestDto.page_codes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct EditionInfo {
    pub id: u64,
    pub copies: u32,
    pub pages_per_copy: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PageCode {
    pub edition: u64,
    pub copy: u32,
    pub page: u32,
}

/// Ce que la lecture a trouvé sur une page du scan (numérotée à partir de 1).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScannedPage {
    pub scan_page: u32,
    pub text: Option<String>,
    pub upside_down: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PageStatus {
    Ok,
    Inferred,
    Unreadable,
    Foreign,
    Duplicate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysedPage {
    pub scan_page: u32,
    pub upside_down: bool,
    pub status: PageStatus,
    pub code: Option<PageCode>,
}

/// Une place de l'ordre cible : une page du scan, ou une page blanche à la place d'une page manquante.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Scan { scan_page: u32, rotate: bool },
    Blank { edition: u64, copy: u32, page: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CopyReport {
    pub edition: u64,
    pub copy: u32,
    pub missing: Vec<u32>,
}

/// Résultat de l'analyse ; sa partie [`Summary`] est renvoyée au front dans le statut final.
#[derive(Debug, Clone)]
pub struct CodeReport {
    pub pages: Vec<AnalysedPage>,
    pub copies: Vec<CopyReport>,
    pub order: Vec<Slot>,
    pub in_order: bool,
}

/// Résumé envoyé au front (`details` du statut `finished`, objet `pagecodes`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Summary {
    pub pages: usize,
    pub copies: usize,
    pub complete: usize,
    pub incomplete: Vec<CopyReport>,
    pub upside_down: Vec<u32>,
    pub duplicates: Vec<u32>,
    pub foreign: Vec<u32>,
    pub unreadable: Vec<u32>,
    pub inferred: Vec<u32>,
    pub blanks: usize,
    pub reordered: bool,
}

/// Lit `CE1|tirage|copie|page` ; `None` pour tout autre texte.
pub fn parse_page_code(text: &str) -> Option<PageCode> {
    let parts: Vec<&str> = text.trim().split('|').collect();
    if parts.len() != 4 || parts[0] != "CE1" {
        return None;
    }
    let digits = |s: &str, max: usize| !s.is_empty() && s.len() <= max && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(parts[1], 9) || !digits(parts[2], 6) || !digits(parts[3], 4) {
        return None;
    }
    Some(PageCode {
        edition: parts[1].parse().ok()?,
        copy: parts[2].parse().ok()?,
        page: parts[3].parse().ok()?,
    })
}

/// Hauteur des bandes lues en bas et en haut de la page (le code est imprimé entre 3,5 et 14,5 mm du bord).
const BAND: f32 = 0.12;

fn decode_band(gray: &GrayImage, top: bool) -> Option<String> {
    let (w, h) = gray.dimensions();
    let band = ((h as f32 * BAND).round() as u32).clamp(1, h);
    let y0 = if top { 0 } else { h - band };
    let mut luma = Vec::with_capacity((w * band) as usize);
    for y in y0..y0 + band {
        for x in 0..w {
            luma.push(gray.get_pixel(x, y)[0]);
        }
    }
    let mut hints = DecodeHints::default();
    rxing::helpers::detect_in_luma_with_hints(luma, w, band, Some(BarcodeFormat::DATA_MATRIX), &mut hints)
        .ok()
        .map(|r| r.getText().to_string())
}

/// Code d'une page : bande du bas, sinon bande du haut (page numérisée à l'envers).
pub fn read_page_code(img: &DynamicImage, scan_page: u32) -> ScannedPage {
    let gray = img.to_luma8();
    if let Some(text) = decode_band(&gray, false) {
        return ScannedPage { scan_page, text: Some(text), upside_down: false };
    }
    match decode_band(&gray, true) {
        // un code lu en haut est celui d'une page à l'envers ; un autre symbole n'est jamais placé
        Some(text) => {
            let upside_down = parse_page_code(&text).is_some();
            ScannedPage { scan_page, text: Some(text), upside_down }
        }
        None => ScannedPage { scan_page, text: None, upside_down: false },
    }
}

/// Analyse les codes lus et calcule l'ordre cible (voir `analyseScan` côté front).
pub fn analyse_scan(scanned: &[ScannedPage], editions: &[EditionInfo]) -> CodeReport {
    let by_id: HashMap<u64, EditionInfo> = editions.iter().map(|e| (e.id, *e)).collect();
    let mut sorted: Vec<&ScannedPage> = scanned.iter().collect();
    sorted.sort_by_key(|p| p.scan_page);
    let mut seen: HashSet<PageCode> = HashSet::new();
    let mut pages: Vec<AnalysedPage> = sorted
        .iter()
        .map(|p| {
            let base = |status, code| AnalysedPage { scan_page: p.scan_page, upside_down: p.upside_down, status, code };
            let Some(text) = p.text.as_deref().filter(|t| !t.is_empty()) else {
                return base(PageStatus::Unreadable, None);
            };
            let code = parse_page_code(text);
            let valid = code.filter(|c| {
                by_id
                    .get(&c.edition)
                    .is_some_and(|e| c.copy >= 1 && c.copy <= e.copies && c.page >= 1 && c.page <= e.pages_per_copy)
            });
            match valid {
                None => base(PageStatus::Foreign, None),
                Some(c) if seen.contains(&c) => base(PageStatus::Duplicate, Some(c)),
                Some(c) => {
                    seen.insert(c);
                    base(PageStatus::Ok, Some(c))
                }
            }
        })
        .collect();

    // pages illisibles placées d'après leurs voisines (code taché ou déchiré)
    for i in 0..pages.len() {
        if pages[i].status != PageStatus::Unreadable {
            continue;
        }
        let mut candidates = Vec::new();
        if i > 0 {
            let before = &pages[i - 1];
            if let (Some(c), PageStatus::Ok | PageStatus::Inferred) = (before.code, before.status) {
                candidates.push(PageCode { page: c.page + 1, ..c });
            }
        }
        if let Some(after) = pages.get(i + 1) {
            if let (Some(c), PageStatus::Ok) = (after.code, after.status) {
                if c.page > 1 {
                    candidates.push(PageCode { page: c.page - 1, ..c });
                }
            }
        }
        let found = candidates.into_iter().find(|c| {
            by_id.get(&c.edition).is_some_and(|e| c.page >= 1 && c.page <= e.pages_per_copy) && !seen.contains(c)
        });
        if let Some(c) = found {
            seen.insert(c);
            pages[i].status = PageStatus::Inferred;
            pages[i].code = Some(c);
        }
    }

    let placed = |p: &AnalysedPage| matches!(p.status, PageStatus::Ok | PageStatus::Inferred);
    let mut copy_keys: Vec<(u64, u32)> = pages
        .iter()
        .filter(|p| placed(p))
        .filter_map(|p| p.code.map(|c| (c.edition, c.copy)))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    copy_keys.sort();
    let located: HashMap<PageCode, &AnalysedPage> =
        pages.iter().filter(|p| placed(p)).filter_map(|p| p.code.map(|c| (c, p))).collect();

    let mut copies = Vec::new();
    let mut order = Vec::new();
    for (edition, copy) in copy_keys {
        let n = by_id[&edition].pages_per_copy;
        let mut missing = Vec::new();
        for page in 1..=n {
            match located.get(&PageCode { edition, copy, page }) {
                Some(p) => order.push(Slot::Scan { scan_page: p.scan_page, rotate: p.upside_down }),
                None => {
                    missing.push(page);
                    order.push(Slot::Blank { edition, copy, page });
                }
            }
        }
        copies.push(CopyReport { edition, copy, missing });
    }
    for p in &pages {
        if matches!(p.status, PageStatus::Unreadable | PageStatus::Foreign | PageStatus::Duplicate) {
            order.push(Slot::Scan { scan_page: p.scan_page, rotate: p.upside_down });
        }
    }
    let in_order = order.iter().enumerate().all(|(i, slot)| {
        matches!(slot, Slot::Scan { scan_page, rotate: false } if Some(*scan_page) == pages.get(i).map(|p| p.scan_page))
    });
    CodeReport { pages, copies, order, in_order }
}

impl CodeReport {
    pub fn summary(&self) -> Summary {
        let select = |s: PageStatus| self.pages.iter().filter(|p| p.status == s).map(|p| p.scan_page).collect::<Vec<_>>();
        let incomplete: Vec<CopyReport> = self.copies.iter().filter(|c| !c.missing.is_empty()).cloned().collect();
        Summary {
            pages: self.pages.len(),
            copies: self.copies.len(),
            complete: self.copies.len() - incomplete.len(),
            incomplete,
            upside_down: self.pages.iter().filter(|p| p.upside_down).map(|p| p.scan_page).collect(),
            duplicates: select(PageStatus::Duplicate),
            foreign: select(PageStatus::Foreign),
            unreadable: select(PageStatus::Unreadable),
            inferred: select(PageStatus::Inferred),
            blanks: self.order.iter().filter(|s| matches!(s, Slot::Blank { .. })).count(),
            reordered: !self.in_order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};
    use rxing::Writer;

    const EDITIONS: [EditionInfo; 1] = [EditionInfo { id: 12, copies: 30, pages_per_copy: 2 }];

    fn scan(texts: &[Option<&str>]) -> Vec<ScannedPage> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| ScannedPage { scan_page: i as u32 + 1, text: t.map(String::from), upside_down: false })
            .collect()
    }
    fn code(copy: u32, page: u32) -> String {
        format!("CE1|12|{}|{}", copy, page)
    }
    fn scan_pages(order: &[Slot]) -> Vec<i64> {
        order
            .iter()
            .map(|s| match s {
                Slot::Scan { scan_page, .. } => *scan_page as i64,
                Slot::Blank { .. } => 0,
            })
            .collect()
    }

    #[test]
    fn lit_seulement_les_codes_du_contrat() {
        assert_eq!(parse_page_code("CE1|12|3|4"), Some(PageCode { edition: 12, copy: 3, page: 4 }));
        assert_eq!(parse_page_code(" CE1|12|3|4\n"), Some(PageCode { edition: 12, copy: 3, page: 4 }));
        for t in ["", "CE2|12|3|4", "CE1|12|3", "CE1|a|3|4", "https://example.org", "CE1|12|3|4|5", "CE1|12|-3|4"] {
            assert_eq!(parse_page_code(t), None, "{}", t);
        }
    }

    #[test]
    fn rien_a_faire_sur_un_scan_dans_l_ordre() {
        let (a, b, c, d) = (code(1, 1), code(1, 2), code(4, 1), code(4, 2));
        let r = analyse_scan(&scan(&[Some(&a), Some(&b), Some(&c), Some(&d)]), &EDITIONS);
        assert!(r.in_order);
        assert!(!r.summary().reordered);
        assert_eq!(r.summary().complete, 2);
    }

    #[test]
    fn remet_en_ordre_et_redresse() {
        let (a, b, c, d) = (code(2, 2), code(1, 1), code(2, 1), code(1, 2));
        let mut s = scan(&[Some(&a), Some(&b), Some(&c), Some(&d)]);
        s[0].upside_down = true;
        let r = analyse_scan(&s, &EDITIONS);
        assert!(!r.in_order);
        assert_eq!(
            r.order,
            vec![
                Slot::Scan { scan_page: 2, rotate: false },
                Slot::Scan { scan_page: 4, rotate: false },
                Slot::Scan { scan_page: 3, rotate: false },
                Slot::Scan { scan_page: 1, rotate: true },
            ]
        );
        assert_eq!(r.summary().upside_down, vec![1]);
    }

    #[test]
    fn page_blanche_a_la_place_d_une_page_manquante() {
        let (a, b, c) = (code(1, 1), code(2, 1), code(2, 2));
        let r = analyse_scan(&scan(&[Some(&a), Some(&b), Some(&c)]), &EDITIONS);
        assert_eq!(r.order[1], Slot::Blank { edition: 12, copy: 1, page: 2 });
        let s = r.summary();
        assert_eq!(s.incomplete, vec![CopyReport { edition: 12, copy: 1, missing: vec![2] }]);
        assert_eq!(s.blanks, 1);
    }

    #[test]
    fn doubles_etrangeres_et_illisibles_en_fin_de_lot() {
        let (a, f, c, d) = (code(1, 1), code(1, 3), code(3, 1), code(3, 2));
        let r = analyse_scan(
            &scan(&[Some(&a), Some(&a), Some("CE1|99|1|1"), Some(&f), Some("autre"), Some(&c), None, None, Some(&d)]),
            &EDITIONS,
        );
        let s = r.summary();
        assert_eq!(s.duplicates, vec![2]);
        assert_eq!(s.foreign, vec![3, 4, 5]);
        assert_eq!(s.unreadable, vec![7, 8]);
        assert_eq!(scan_pages(&r.order)[r.order.len() - 6..], [2, 3, 4, 5, 7, 8]);
    }

    #[test]
    fn deduit_une_page_illisible_de_ses_voisines() {
        let (a, c, d) = (code(5, 1), code(6, 1), code(6, 2));
        let r = analyse_scan(&scan(&[Some(&a), None, Some(&c), Some(&d)]), &EDITIONS);
        assert_eq!(r.summary().inferred, vec![2]);
        assert_eq!(r.pages[1].code, Some(PageCode { edition: 12, copy: 5, page: 2 }));
        assert!(r.in_order);
        let n = code(7, 2);
        let next = analyse_scan(&scan(&[None, Some(&n)]), &EDITIONS);
        assert_eq!(next.pages[0].code, Some(PageCode { edition: 12, copy: 7, page: 1 }));
        // jamais une page qui existe ailleurs
        let (e, g) = (code(5, 1), code(5, 2));
        let r = analyse_scan(&scan(&[Some(&e), None, Some(&g)]), &EDITIONS);
        assert_eq!(r.summary().unreadable, vec![2]);
    }

    /// Page blanche A4 à 2000 px de haut avec le code en bas au centre (11 mm, centre à 9 mm du bord).
    fn printed_page(text: &str) -> DynamicImage {
        let (w, h) = (1414u32, 2000u32);
        let px_per_mm = h as f32 / 297.0;
        let size = (11.0 * px_per_mm) as u32;
        let matrix = rxing::datamatrix::DataMatrixWriter
            .encode(text, &BarcodeFormat::DATA_MATRIX, 0, 0)
            .unwrap();
        let module = size as f32 / matrix.getWidth() as f32;
        let (x0, y0) = (w / 2 - size / 2, h - (9.0 * px_per_mm) as u32 - size / 2);
        let img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
            if x >= x0 && x < x0 + size && y >= y0 && y < y0 + size {
                let mx = ((x - x0) as f32 / module) as u32;
                let my = ((y - y0) as f32 / module) as u32;
                if mx < matrix.getWidth() && my < matrix.getHeight() && matrix.get(mx, my) {
                    return Luma([0u8]);
                }
            }
            Luma([255u8])
        });
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn lit_le_code_d_une_page_droite_et_d_une_page_a_l_envers() {
        let page = printed_page("CE1|41|3|2");
        assert_eq!(
            read_page_code(&page, 7),
            ScannedPage { scan_page: 7, text: Some("CE1|41|3|2".into()), upside_down: false }
        );
        assert_eq!(
            read_page_code(&page.rotate180(), 8),
            ScannedPage { scan_page: 8, text: Some("CE1|41|3|2".into()), upside_down: true }
        );
        let blank = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(1414, 2000, Luma([255u8])));
        assert_eq!(read_page_code(&blank, 9), ScannedPage { scan_page: 9, text: None, upside_down: false });
    }

    #[test]
    fn resume_serialise_pour_le_front() {
        let (a, b) = (code(1, 1), code(2, 1));
        let r = analyse_scan(&scan(&[Some(&a), Some(&b)]), &EDITIONS);
        let json = serde_json::to_value(r.summary()).unwrap();
        for field in ["pages", "copies", "complete", "incomplete", "upside_down", "duplicates", "foreign", "unreadable", "inferred", "blanks", "reordered"] {
            assert!(json.get(field).is_some(), "champ {} absent", field);
        }
    }
}
