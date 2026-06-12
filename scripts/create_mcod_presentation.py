from __future__ import annotations

import csv
import html
import math
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "presentation"
TEMPLATE = ROOT / "related_documents" / "COD_presentation.pptx"
PPTX = OUT / "MCOD_FM_RobustBench_research_presentation.pptx"
NOTES = OUT / "MCOD_FM_RobustBench_speaker_notes.md"

SLIDE_W = 9144000
SLIDE_H = 5143500
EMU_PER_IN = 914400
EMU_PER_PT = 12700

BG = "F7F8FA"
INK = "17202A"
MUTED = "5D6D7E"
ACCENT = "0B6E69"
ACCENT2 = "B75D00"
ACCENT3 = "2E5AAC"
LINE = "D8DEE8"
SOFT = "EEF3F2"
SOFT2 = "FFF4E8"


def emu(inches: float) -> int:
    return int(inches * EMU_PER_IN)


def pt(size: float) -> int:
    return int(size * 100)


def esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def rel_xml(targets: list[tuple[str, str]]) -> str:
    rels = ['<?xml version="1.0" encoding="UTF-8"?>']
    rels.append('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">')
    for rid, (rel_type, target) in enumerate(targets, start=1):
        rels.append(f'<Relationship Id="rId{rid}" Type="{rel_type}" Target="{target}"/>')
    rels.append("</Relationships>")
    return "".join(rels)


def tx_runs(lines: list[str], size: int, color: str, bold: bool = False, bullet: bool = False, line_spacing: int = 108000) -> str:
    out = []
    for line in lines:
        if bullet:
            ppr = (
                f'<a:pPr marL="171450" indent="-171450"><a:lnSpc><a:spcPct val="{line_spacing}"/>'
                '</a:lnSpc><a:buChar char="&#8226;"/></a:pPr>'
            )
        else:
            ppr = f'<a:pPr><a:lnSpc><a:spcPct val="{line_spacing}"/></a:lnSpc></a:pPr>'
        out.append(
            f'<a:p>{ppr}<a:r><a:rPr lang="en-US" sz="{pt(size)}" b="{1 if bold else 0}">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr>'
            f"<a:t>{esc(line)}</a:t></a:r></a:p>"
        )
    return "".join(out)


def shape_rect(shape_id: int, x: int, y: int, w: int, h: int, fill: str, line: str | None = None, radius: bool = False) -> str:
    geom = "roundRect" if radius else "rect"
    line_xml = '<a:ln w="0"><a:noFill/></a:ln>' if line is None else (
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
    )
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Box {shape_id}"/>'
        '<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr>'
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>'
    )


def textbox(shape_id: int, x: int, y: int, w: int, h: int, lines: list[str], size: int = 14, color: str = INK,
            bold: bool = False, bullet: bool = False, anchor: str = "t", align: str = "l") -> str:
    paras = tx_runs(lines, size, color, bold, bullet)
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Text {shape_id}"/>'
        '<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln w="0"><a:noFill/></a:ln></p:spPr>'
        f'<p:txBody><a:bodyPr lIns="45000" tIns="30000" rIns="45000" bIns="30000" anchor="{anchor}">'
        '<a:noAutofit/></a:bodyPr><a:lstStyle/>'
        f'{paras}</p:txBody></p:sp>'
    ).replace("<a:pPr>", f'<a:pPr algn="{align}">')


def picture(shape_id: int, rel_id: int, x: int, y: int, w: int, h: int) -> str:
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="Picture {shape_id}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="rId%d"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>' % rel_id
        + f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def table(shape_start: int, x: int, y: int, w: int, h: int, rows: list[list[str]], widths: list[float] | None = None,
          header_fill: str = ACCENT, body_fill: str = "FFFFFF", font_size: int = 9) -> tuple[str, int]:
    if not widths:
        widths = [1 / len(rows[0])] * len(rows[0])
    total = sum(widths)
    widths = [v / total for v in widths]
    row_h = h // len(rows)
    cur_id = shape_start
    xml = []
    for r, row in enumerate(rows):
        cx = x
        for c, cell in enumerate(row):
            cw = int(w * widths[c])
            fill = header_fill if r == 0 else body_fill
            color = "FFFFFF" if r == 0 else INK
            xml.append(shape_rect(cur_id, cx, y + r * row_h, cw, row_h, fill, LINE))
            cur_id += 1
            xml.append(textbox(cur_id, cx + 15000, y + r * row_h + 5000, cw - 30000, row_h - 10000, [cell], font_size, color, r == 0, anchor="ctr"))
            cur_id += 1
            cx += cw
    return "".join(xml), cur_id


def fit_box(image_path: Path, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    with Image.open(image_path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    return x + (w - nw) // 2, y + (h - nh) // 2, nw, nh


class Slide:
    def __init__(self, title: str, subtitle: str | None = None):
        self.title = title
        self.subtitle = subtitle
        self.items: list[str] = []
        self.images: list[Path] = []
        self.notes: list[str] = []
        self.next_id = 10

    def add(self, xml: str) -> None:
        self.items.append(xml)

    def sid(self) -> int:
        self.next_id += 1
        return self.next_id

    def add_text(self, x, y, w, h, lines, size=14, color=INK, bold=False, bullet=False, anchor="t", align="l"):
        self.add(textbox(self.sid(), emu(x), emu(y), emu(w), emu(h), lines, size, color, bold, bullet, anchor, align))

    def add_box(self, x, y, w, h, fill="FFFFFF", line=LINE, radius=False):
        self.add(shape_rect(self.sid(), emu(x), emu(y), emu(w), emu(h), fill, line, radius))

    def add_image(self, path: Path, x, y, w, h):
        self.images.append(path)
        rid = len(self.images) + 1
        fx, fy, fw, fh = fit_box(path, emu(x), emu(y), emu(w), emu(h))
        self.add(picture(self.sid(), rid, fx, fy, fw, fh))

    def add_table(self, x, y, w, h, rows, widths=None, font_size=9):
        xml, next_id = table(self.sid(), emu(x), emu(y), emu(w), emu(h), rows, widths, font_size=font_size)
        self.next_id = next_id
        self.add(xml)

    def render(self, num: int, total: int) -> str:
        header = [
            shape_rect(2, 0, 0, SLIDE_W, SLIDE_H, BG),
            shape_rect(3, 0, 0, emu(0.10), SLIDE_H, ACCENT),
            textbox(4, emu(0.28), emu(0.18), emu(7.6), emu(0.42), [self.title], 19, INK, True),
            textbox(5, emu(8.25), emu(0.22), emu(1.35), emu(0.28), [f"{num:02d}/{total:02d}"], 8, MUTED, False, align="r"),
            shape_rect(6, emu(0.28), emu(0.66), emu(9.08), emu(0.012), LINE),
        ]
        if self.subtitle:
            header.append(textbox(7, emu(0.30), emu(0.58), emu(8.5), emu(0.25), [self.subtitle], 8, MUTED))
        body = "".join(header + self.items)
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            f'{body}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        )


def make_bar_chart(rows: list[dict[str, str]], out: Path) -> None:
    rows = rows[:11]
    W, H = 1300, 650
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_f = font(32, True)
    label_f = font(21)
    small_f = font(18)
    d.text((40, 28), "Local evaluation of MSIResults on 500 MCOD test masks", font=title_f, fill=(23, 32, 42))
    x0, y0 = 250, 105
    bar_h, gap = 28, 20
    max_dice = max(float(r["Dice_up"]) for r in rows)
    max_iou = max(float(r["mIoU_up"]) for r in rows)
    for i, r in enumerate(rows):
        y = y0 + i * (bar_h + gap)
        name = r["method"]
        dice = float(r["Dice_up"])
        iou = float(r["mIoU_up"])
        mae = float(r["MAE_down"])
        d.text((42, y + 2), name, font=label_f, fill=(23, 32, 42))
        d.rounded_rectangle((x0, y, x0 + int(520 * dice / max_dice), y + 12), radius=5, fill=(11, 110, 105))
        d.rounded_rectangle((x0, y + 16, x0 + int(520 * iou / max_iou), y + 28), radius=5, fill=(46, 90, 172))
        d.text((800, y - 2), f"Dice {dice:.3f}  IoU {iou:.3f}  MAE {mae:.4f}", font=small_f, fill=(70, 80, 95))
    d.rounded_rectangle((1040, 105, 1080, 122), radius=6, fill=(11, 110, 105))
    d.text((1092, 99), "Dice", font=small_f, fill=(23, 32, 42))
    d.rounded_rectangle((1040, 135, 1080, 152), radius=6, fill=(46, 90, 172))
    d.text((1092, 129), "mIoU", font=small_f, fill=(23, 32, 42))
    d.text((40, 604), "Thresholded overlap metrics are local checks; official paper table reports E, S, F_beta, and MAE.", font=small_f, fill=(93, 109, 126))
    img.save(out)


def make_rgb_msi_chart(out: Path) -> None:
    rows = [
        ("SINet", 0.601, 0.616, 0.335, 0.369),
        ("CODCEF", 0.632, 0.677, 0.359, 0.444),
        ("C2FNet-V2", 0.743, 0.810, 0.553, 0.654),
        ("PCNet", 0.788, 0.855, 0.149, 0.386),
    ]
    W, H = 1200, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((35, 25), "Official paper: RGB to MSI improves COD performance", font=font(31, True), fill=(23, 32, 42))
    x0, y0 = 70, 105
    group_w = 255
    maxv = 0.90
    for i, (name, s_rgb, s_msi, f_rgb, f_msi) in enumerate(rows):
        gx = x0 + i * group_w
        d.text((gx, 410), name, font=font(20, True), fill=(23, 32, 42))
        for j, (label, val, col) in enumerate([("S rgb", s_rgb, (185, 93, 0)), ("S msi", s_msi, (11, 110, 105)), ("F rgb", f_rgb, (235, 169, 84)), ("F msi", f_msi, (46, 90, 172))]):
            bw = 36
            bx = gx + j * 43
            bh = int(280 * val / maxv)
            d.rectangle((bx, 380 - bh, bx + bw, 380), fill=col)
            d.text((bx - 2, 384 - bh), f"{val:.2f}", font=font(13), fill=(70, 80, 95))
    legend = [("S rgb", (185, 93, 0)), ("S msi", (11, 110, 105)), ("F rgb", (235, 169, 84)), ("F msi", (46, 90, 172))]
    for i, (lab, col) in enumerate(legend):
        x = 820 + i * 88
        d.rectangle((x, 55, x + 22, 75), fill=col)
        d.text((x + 28, 52), lab, font=font(15), fill=(23, 32, 42))
    d.text((35, 470), "Example from paper: C2FNet-V2 S_alpha 0.743 to 0.810; F_beta 0.553 to 0.654.", font=font(18), fill=(93, 109, 126))
    img.save(out)


def make_qual_grid(out: Path) -> None:
    gt_dir = ROOT / "data" / "MCOD_resized" / "TestDataset" / "GT"
    img_dir = ROOT / "data" / "MCOD_resized" / "TestDataset" / "Pcolor"
    methods = ["PRNet_MSI", "PCNet_MSI", "C2FNetV2_MSI", "SINetV2_MSI"]
    sample = None
    for gt in sorted(gt_dir.glob("*")):
        if (img_dir / (gt.stem + ".png")).exists() and all((ROOT / "MSIResults" / m / (gt.stem + ".png")).exists() for m in methods):
            sample = gt.stem
            break
    if sample is None:
        return
    entries = [("Pcolor", img_dir / f"{sample}.png"), ("GT", gt_dir / f"{sample}.jpg")]
    entries += [(m.replace("_MSI", ""), ROOT / "MSIResults" / m / f"{sample}.png") for m in methods]
    tile_w, tile_h = 210, 158
    W, H = tile_w * len(entries), tile_h + 54
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for i, (label, path) in enumerate(entries):
        im = Image.open(path).convert("RGB").resize((tile_w, tile_h))
        img.paste(im, (i * tile_w, 30))
        d.text((i * tile_w + 8, 6), label, font=font(18, True), fill=(23, 32, 42))
    d.text((8, H - 22), f"Sample: {sample}", font=font(14), fill=(93, 109, 126))
    img.save(out)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_slides() -> list[Slide]:
    metric_rows = read_csv(OUT / "msiresults_eval_summary.csv")
    local_top = metric_rows[:6]
    slides: list[Slide] = []

    s = Slide("MCOD-FM-RobustBench", "Research-level presentation: paper review, dataset audit, preprocessing, baseline results")
    s.add_text(0.38, 0.95, 5.5, 1.0, ["Foundation-Model Promptability and Spectral-Degradation Robustness Benchmark", "for Multispectral Camouflaged Object Detection"], 25, INK, True)
    for i, (k, v) in enumerate([("1,527", "MSI images"), ("8", "spectral bands"), ("500", "test split"), ("11", "COD baselines")]):
        x = 0.45 + i * 2.18
        s.add_box(x, 2.45, 1.78, 0.78, "FFFFFF", LINE, True)
        s.add_text(x + 0.08, 2.48, 0.72, 0.32, [k], 23, ACCENT, True)
        s.add_text(x + 0.08, 2.86, 1.45, 0.22, [v], 9, MUTED)
    s.add_text(0.46, 3.65, 8.5, 0.8, ["Core claim: RGB camouflage failures are partly spectral failures; MCOD lets us test whether non-visible bands make the object separable and robust."], 15, INK)
    s.notes = ["Open by defining the central problem: visually camouflaged objects can still differ spectrally. The presentation moves from MCOD paper review to our repo-backed audit and benchmark pipeline."]
    slides.append(s)

    s = Slide("Paper Review: Problem And Gap", "MCOD: The First Challenging Benchmark for Multispectral Camouflaged Object Detection, ACM MM 2025")
    s.add_box(0.38, 0.92, 2.7, 3.85, SOFT, LINE, True)
    s.add_text(0.50, 1.04, 2.42, 0.35, ["Why RGB COD saturates"], 15, ACCENT, True)
    s.add_text(0.52, 1.50, 2.35, 2.55, [
        "Objects share color and texture with their background.",
        "Small targets and extreme illumination trigger missed detections.",
        "Most prior COD datasets are RGB-only.",
        "Existing COD networks are often architecture-heavy but modality-limited.",
    ], 11, INK, bullet=True)
    s.add_box(3.35, 0.92, 2.7, 3.85, "FFFFFF", LINE, True)
    s.add_text(3.48, 1.04, 2.42, 0.35, ["MCOD intervention"], 15, ACCENT3, True)
    s.add_text(3.50, 1.50, 2.35, 2.55, [
        "First public MSI benchmark for COD.",
        "8 registered spectral channels per sample.",
        "High-resolution 1140 x 860 imagery.",
        "Pixel-level masks plus challenge attributes.",
    ], 11, INK, bullet=True)
    s.add_box(6.33, 0.92, 2.75, 3.85, SOFT2, LINE, True)
    s.add_text(6.46, 1.04, 2.42, 0.35, ["Research implication"], 15, ACCENT2, True)
    s.add_text(6.48, 1.50, 2.35, 2.55, [
        "Foreground-background discrimination can come from material/spectral response, not just RGB appearance.",
        "The benchmark opens a route for spectral fusion, spectral promptability, and degradation robustness.",
    ], 11, INK, bullet=True)
    s.notes = ["The paper's novelty is not another COD architecture; it introduces the missing benchmark that makes multispectral COD measurable."]
    slides.append(s)

    s = Slide("Paper Review: Dataset Positioning", "Compared with CHAMELEON, CAMO-COCO, NC4K, COD10K")
    rows = [
        ["Dataset", "Year", "Modality", "Channels", "Resolution", "Attr.", "Camo imgs"],
        ["CHAMELEON", "2017", "RGB", "3", "450x300 to 2304x34", "-", "76"],
        ["CAMO-COCO", "2019", "RGB", "3", "154x156 to 7360x4912", "7", "1,250"],
        ["NC4K", "2021", "RGB", "3", "354x268 to 1280x960", "-", "4,121"],
        ["COD10K", "2022", "RGB", "3", "300x199 to 2976x3968", "7", "5,066"],
        ["MCOD", "2025", "MSI", "8", "1140x860", "8", "1,527"],
    ]
    s.add_table(0.36, 0.95, 8.8, 2.2, rows, [1.25, .55, .85, .7, 1.9, .55, .85], 8)
    s.add_box(0.38, 3.43, 8.75, 1.08, "FFFFFF", LINE, True)
    s.add_text(0.55, 3.53, 8.2, 0.65, [
        "MCOD trades raw dataset size for richer sensing: calibrated multispectral capture, uniform spatial resolution, pixel masks, and explicit challenge attributes.",
        "Paper statistic to remember: average object area is 0.429% of the image, far below COD10K's reported 8.94%.",
    ], 12, INK)
    s.notes = ["Use this table to show why MCOD is not simply bigger. It is a different benchmark axis: spectral information plus hard object scale."]
    slides.append(s)

    s = Slide("Paper Review: Challenge Attributes", "Eight difficulty labels define targeted evaluation subsets")
    rows = [
        ["Code", "Meaning", "Failure pressure"],
        ["SC", "Shape Complex", "irregular boundaries"],
        ["SO", "Small Object", "tiny object-to-image ratio"],
        ["OC", "Object Occlusion", "partial visibility"],
        ["BS", "Background Similarity", "weak RGB contrast"],
        ["UI/OE/II", "Extreme Illumination", "lighting instability"],
        ["CB", "Complex Background", "distractors and clutter"],
    ]
    s.add_table(0.42, 0.92, 4.35, 3.65, rows, [.65, 1.55, 2.1], 9)
    s.add_image(ROOT / "outputs" / "figures" / "attribute_distribution_bar.png", 5.05, 0.95, 4.05, 2.25)
    s.add_image(ROOT / "outputs" / "figures" / "attribute_cooccurrence_heatmap.png", 5.45, 3.18, 3.25, 1.65)
    s.notes = ["Explain that attribute labels matter because a single aggregate score can hide failures on small objects or bad lighting."]
    slides.append(s)

    s = Slide("Paper Review: Official Baseline Results", "11 COD methods adapted from RGB input to 8-channel MSI input")
    rows = [
        ["Method", "Venue", "E up", "S up", "F up", "MAE down"],
        ["SINet", "CVPR 2020", ".758", ".616", ".369", ".006"],
        ["LSR", "CVPR 2021", ".830", ".625", ".373", ".005"],
        ["CODCEF", "Sensors 2021", ".763", ".677", ".444", ".004"],
        ["C2FNet", "IJCAI 2021", ".726", ".721", ".403", ".010"],
        ["C2FNet-V2", "TCSVT 2022", ".913", ".810", ".654", ".008"],
        ["SINet-V2", "TPAMI 2022", ".849", ".728", ".492", ".004"],
        ["ASBI", "CVIU 2023", ".684", ".675", ".370", ".014"],
        ["FIRNet", "TVC 2024", ".882", ".738", ".537", ".004"],
        ["PRNet", "TCSVT 2024", ".926", ".826", ".698", ".002"],
        ["IdeNet", "TIP 2024", ".846", ".808", ".588", ".004"],
        ["PCNet", "arXiv 2024", ".633", ".855", ".386", ".003"],
    ]
    s.add_table(0.30, 0.88, 8.95, 3.45, rows, [1.15, 1.15, .62, .62, .62, .78], 7)
    s.add_text(0.45, 4.42, 8.55, 0.42, ["Official reading: PRNet leads E/F/MAE, PCNet leads S-measure; strong scores still coexist with hard failure cases under SO, extreme illumination, BS, and CB."], 11, INK)
    s.notes = ["Do not overclaim one winner. The metric split is important: structure, overlap, and absolute error tell different stories."]
    slides.append(s)

    s = Slide("Paper Review: Why Multispectral Helps", "Controlled RGB vs MSI comparison from the official paper")
    s.add_image(OUT / "rgb_msi_chart.png", 0.45, 0.90, 4.8, 2.2)
    rows = [
        ["Method", "Input", "E", "S", "F", "MAE"],
        ["SINet", "RGB", ".695", ".601", ".335", ".005"],
        ["SINet", "MSI", ".758", ".616", ".369", ".006"],
        ["C2FNet-V2", "RGB", ".885", ".743", ".553", ".009"],
        ["C2FNet-V2", "MSI", ".913", ".810", ".654", ".008"],
        ["PCNet", "RGB", ".397", ".788", ".149", ".005"],
        ["PCNet", "MSI", ".633", ".855", ".386", ".003"],
    ]
    s.add_table(5.42, 0.95, 3.65, 2.45, rows, [1.0, .65, .48, .48, .48, .62], 8)
    s.add_text(0.58, 3.65, 8.25, 0.70, [
        "The key evidence is consistency: MSI improves most metrics across representative architectures.",
        "Paper example: C2FNet-V2 S_alpha improves 0.743 to 0.810; F_beta improves 0.553 to 0.654.",
    ], 12, INK, bullet=True)
    s.notes = ["This is the argument that justifies our robustness extension: if spectral cues help, we should test when those cues are missing, noisy, or misaligned."]
    slides.append(s)

    s = Slide("This Repo: Research Objective", "From MCOD benchmark to promptability and robustness benchmark")
    s.add_box(0.45, 0.95, 2.7, 3.25, SOFT, LINE, True)
    s.add_text(0.62, 1.10, 2.3, 0.32, ["Question 1"], 14, ACCENT, True)
    s.add_text(0.62, 1.55, 2.22, 1.65, ["Can foundation or RGB-trained segmentation models use MSI-derived views to localize camouflaged objects?"], 16, INK, True)
    s.add_box(3.45, 0.95, 2.7, 3.25, "FFFFFF", LINE, True)
    s.add_text(3.62, 1.10, 2.3, 0.32, ["Question 2"], 14, ACCENT3, True)
    s.add_text(3.62, 1.55, 2.22, 1.65, ["Which spectral views preserve detection evidence: false color, red-edge, NIR, PCA projections, or all-8 tensors?"], 15, INK, True)
    s.add_box(6.45, 0.95, 2.7, 3.25, SOFT2, LINE, True)
    s.add_text(6.62, 1.10, 2.3, 0.32, ["Question 3"], 14, ACCENT2, True)
    s.add_text(6.62, 1.55, 2.22, 1.65, ["How robust are results when spectral bands are degraded, dropped, perturbed, or misaligned?"], 16, INK, True)
    s.notes = ["Frame the repo as an extension of the paper: not just reproducing MCOD, but stress-testing spectral usefulness."]
    slides.append(s)

    s = Slide("Data Organization And Corrections", "Raw MCOD standardized into a clean working dataset")
    rows = [
        ["Path", "Train", "Test", "Role"],
        ["data/MCOD_raw/*/Pcolor", "1027", "500", "official false-color PNG"],
        ["data/MCOD_raw/*/GT", "1027", "500", "pixel masks"],
        ["data/MCOD_raw/*/Mat", "1027", "500", "8-band gray cubes"],
        ["data/MCOD_resized/*/Pcolor", "1027", "500", "uniform 1140x860"],
        ["data/MCOD_resized/*/GT", "1027", "500", "uniform 1140x860"],
    ]
    s.add_table(0.38, 0.92, 8.75, 2.45, rows, [2.5, .6, .6, 2.25], 8)
    s.add_text(0.52, 3.62, 8.3, 0.68, [
        "Resolution repair performed in a separate folder to preserve raw data.",
        "Final resolution summary: Pcolor 1527 at 1140x860; GT 1527 at 1140x860.",
        "Recent code change: process_mask now copies official masks unchanged rather than binarizing label values.",
    ], 11, INK, bullet=True)
    s.notes = ["Emphasize reproducibility: the raw folder remains intact and the corrected dataset is a derived artifact."]
    slides.append(s)

    s = Slide("Audit Results: Dataset Integrity", "Quality gates before any model-level claim")
    rows = [
        ["Check", "Result", "Interpretation"],
        ["Sample count", "1527 / 1527", "matches official dataset size"],
        ["Official split", "1027 train / 500 test", "split is intact"],
        ["Image-mask-Mat count", "all equal per split", "sample triples align"],
        ["Corrupted files", "0 rows reported", "readability check passed"],
        ["Missing bands", "0 rows reported", "8-band completeness passed"],
        ["Missing masks", "0 rows reported", "mask linkage passed"],
        ["Train-test leakage", "0 rows reported", "no exact leakage reported"],
    ]
    s.add_table(0.40, 0.90, 8.70, 3.25, rows, [1.55, 1.45, 3.1], 9)
    s.add_text(0.55, 4.34, 8.2, 0.35, ["Audit files: MCOD_inspection_report/*.csv and data_audit/MCOD/*.csv"], 10, MUTED)
    s.notes = ["This slide supports credibility. It shows the dataset state is checked, not assumed."]
    slides.append(s)

    s = Slide("Processing Pipeline: Spectral Views", "process_dataset.py generates model-ready views from each 8-band cube")
    s.add_image(ROOT / "outputs" / "figures" / "sample_band_visualization.png", 0.45, 0.92, 4.4, 1.75)
    rows = [
        ["Output view", "Definition"],
        ["official_false_colour", "R=S5, G=S3, B=S2"],
        ["S6_rededge_gray3", "S6 normalized as 3-channel image"],
        ["S7_nir1_gray3", "S7 normalized as 3-channel image"],
        ["S8_nir2_gray3", "S8 normalized as 3-channel image"],
        ["visible_group_projection", "PCA over S1-S5"],
        ["nir_group_projection", "PCA over S6-S8"],
        ["all8_input", "H x W x 8 normalized tensor"],
        ["ground_truth_mask", "official mask copied unchanged"],
    ]
    s.add_table(5.05, 0.92, 4.1, 3.65, rows, [1.55, 2.25], 8)
    s.add_text(0.55, 3.10, 4.05, 1.0, ["Processed counts per view: 1027 train + 500 test for all eight output groups, including all8_input tensors and masks."], 13, INK)
    s.notes = ["Explain that these views make an MSI dataset compatible with RGB COD models, spectral ablations, and tensor-native models."]
    slides.append(s)

    s = Slide("Repo Results: Available Baseline Outputs", "MSIResults contains complete predictions for the 500-image test split")
    rows = [["Method folder", "Pred masks"], *[[r["method"], r["samples"]] for r in metric_rows]]
    s.add_table(0.45, 0.92, 3.0, 3.65, rows, [1.6, .8], 8)
    s.add_image(OUT / "local_metrics_bar.png", 3.80, 0.95, 5.05, 2.65)
    s.add_text(3.95, 3.78, 4.75, 0.62, ["Local check agrees with the paper's broad picture: PRNet has the lowest MAE, PCNet has high overlap/structure behavior, and older baselines degrade strongly."], 11, INK)
    s.notes = ["Clarify that this local table uses simple thresholded Dice and IoU, while official paper metrics include E, S, F_beta, and MAE."]
    slides.append(s)

    s = Slide("Qualitative Snapshot", "One test image across input, mask, and representative prediction maps")
    s.add_image(OUT / "qualitative_grid.png", 0.45, 1.05, 8.7, 2.1)
    s.add_text(0.55, 3.58, 8.2, 0.78, [
        "Dense masks often differ in boundary sharpness and small-object recall even when aggregate MAE is low.",
        "This motivates attribute-wise and failure-mode analysis rather than relying only on mean test-set scores.",
    ], 12, INK, bullet=True)
    s.notes = ["Use this slide to tell the story visually: predictions are not just numbers; boundary quality and missed areas matter."]
    slides.append(s)

    s = Slide("Dataset Statistics From Audit", "Object scale and spectral distribution")
    s.add_image(ROOT / "outputs" / "figures" / "object_area_histogram.png", 0.45, 0.92, 4.0, 2.15)
    s.add_image(ROOT / "outputs" / "figures" / "train_test_size_distribution.png", 4.95, 0.92, 4.0, 2.15)
    s.add_image(ROOT / "outputs" / "figures" / "band_mean_std_plot.png", 0.45, 3.24, 4.0, 1.45)
    s.add_text(4.95, 3.28, 3.95, 1.05, [
        "Train/test area distributions are nearly matched: mean area ratio 0.00538 in both splits.",
        "Band means and variances show that visible, red-edge, and NIR channels have distinct intensity statistics.",
    ], 11, INK, bullet=True)
    s.notes = ["Tie this back to generalization: matched splits reduce a confound, while band statistics justify spectral-specific preprocessing."]
    slides.append(s)

    s = Slide("Evaluation Protocol", "Metrics retained for research-grade reporting")
    rows = [
        ["Metric", "Direction", "Why it matters"],
        ["S-measure", "higher", "structural similarity of object map"],
        ["E-measure", "higher", "alignment between prediction and GT"],
        ["F-measure", "higher", "precision-recall balance"],
        ["MAE", "lower", "pixel-level absolute error"],
        ["Weighted F", "higher", "spatially weighted F-score"],
        ["Dice / mIoU", "higher", "thresholded overlap"],
        ["Boundary F", "higher", "contour quality"],
    ]
    s.add_table(0.45, 0.92, 5.0, 3.5, rows, [1.25, .8, 2.45], 9)
    s.add_box(5.80, 1.00, 3.15, 2.95, SOFT, LINE, True)
    s.add_text(5.98, 1.15, 2.75, 0.45, ["Reporting rule"], 15, ACCENT, True)
    s.add_text(5.98, 1.72, 2.7, 1.65, [
        "Report both aggregate and attribute-wise scores.",
        "State whether masks are official, binarized, resized, or interpolated.",
        "Separate official paper metrics from local verification metrics.",
    ], 11, INK, bullet=True)
    s.notes = ["This slide prevents metric confusion. It also explains why low MAE on tiny objects can be misleading."]
    slides.append(s)

    s = Slide("Failure Modes To Analyze Next", "Where MCOD remains difficult")
    s.add_box(0.42, 0.95, 2.65, 3.45, "FFFFFF", LINE, True)
    s.add_text(0.58, 1.12, 2.2, 0.35, ["Small objects"], 15, ACCENT2, True)
    s.add_text(0.58, 1.65, 2.16, 1.8, ["Low absolute pixel area can yield deceptively good MAE while missing the object. Need Dice, IoU, boundary, and attribute-specific recall."], 12, INK)
    s.add_box(3.47, 0.95, 2.65, 3.45, SOFT, LINE, True)
    s.add_text(3.63, 1.12, 2.2, 0.35, ["Lighting extremes"], 15, ACCENT, True)
    s.add_text(3.63, 1.65, 2.16, 1.8, ["UI, OE, and II can collapse visible contrast. MSI should be tested under band drop, noise, and exposure perturbation."], 12, INK)
    s.add_box(6.52, 0.95, 2.65, 3.45, SOFT2, LINE, True)
    s.add_text(6.68, 1.12, 2.2, 0.35, ["Spectral alignment"], 15, ACCENT3, True)
    s.add_text(6.68, 1.65, 2.16, 1.8, ["Real sensors can misregister bands. Robustness should include spatial shifts between visible, red-edge, and NIR groups."], 12, INK)
    s.notes = ["This sets up future work: the project becomes stronger if it tests why models fail, not only which model wins."]
    slides.append(s)

    s = Slide("Planned Experiment Matrix", "From current repository state to publishable benchmark extension")
    rows = [
        ["Stage", "Experiment", "Main output"],
        ["1", "Manifest and audit finalization", "frozen CSV manifest + QC report"],
        ["2", "Baseline reproduction", "official metric table on 500 test images"],
        ["3", "Spectral view analysis", "false color vs S6/S7/S8 vs PCA views"],
        ["4", "Foundation-model promptability", "SAM/SAM2 prompt sensitivity"],
        ["5", "Spectral degradation robustness", "band drop/noise/misalignment curves"],
        ["6", "Attribute-wise failure analysis", "SO/EI/BS/CB breakdown and examples"],
    ]
    s.add_table(0.42, 0.92, 8.65, 3.55, rows, [.65, 2.45, 3.25], 9)
    s.notes = ["This is the transition from current dataset engineering to the final research paper."]
    slides.append(s)

    s = Slide("Contributions And Takeaway", "What this project can credibly claim now")
    s.add_text(0.55, 1.02, 8.2, 2.0, [
        "1. MCOD paper review establishes the research gap: RGB COD lacks spectral evidence.",
        "2. Repo now contains a corrected MCOD_resized working dataset with verified split, counts, masks, and resolution.",
        "3. process_dataset.py creates multiple MSI-to-RGB and tensor-native views for controlled experiments.",
        "4. Existing MSIResults predictions cover all 500 test images for 11 baseline methods.",
        "5. Next research value comes from attribute-wise and spectral-degradation robustness experiments.",
    ], 14, INK, bullet=False)
    s.add_box(0.60, 3.65, 8.0, 0.68, SOFT, LINE, True)
    s.add_text(0.78, 3.76, 7.55, 0.38, ["One-sentence thesis: MCOD makes camouflage detection a spectral robustness problem, not only an RGB segmentation problem."], 14, ACCENT, True)
    s.notes = ["Close with the thesis. It is compact and defensible."]
    slides.append(s)

    return slides


def content_types(num_slides: int, images: list[str]) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/theme/theme2.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/theme/theme3.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides += [
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, num_slides + 1)
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Default Extension="jpg" ContentType="image/jpeg"/>'
        '<Default Extension="jpeg" ContentType="image/jpeg"/>'
        + "".join(overrides)
        + "</Types>"
    )


def presentation_xml(num_slides: int) -> str:
    ids = "".join([f'<p:sldId id="{255+i}" r:id="rId{4+i}"/>' for i in range(1, num_slides + 1)])
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId2"/><p:sldMasterId id="2147483650" r:id="rId3"/></p:sldMasterIdLst>'
        '<p:notesMasterIdLst><p:notesMasterId r:id="rId4"/></p:notesMasterIdLst>'
        f'<p:sldIdLst>{ids}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/><p:notesSz cx="5143500" cy="9144000"/></p:presentation>'
    )


def presentation_rels(num_slides: int) -> str:
    rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    targets = [
        (f"{rel_type}/theme", "theme/theme1.xml"),
        (f"{rel_type}/slideMaster", "slideMasters/slideMaster1.xml"),
        (f"{rel_type}/slideMaster", "slideMasters/slideMaster2.xml"),
        (f"{rel_type}/notesMaster", "notesMasters/notesMaster1.xml"),
    ]
    targets += [(f"{rel_type}/slide", f"slides/slide{i}.xml") for i in range(1, num_slides + 1)]
    targets += [(f"{rel_type}/presProps", "presProps.xml")]
    return rel_xml(targets)


def build_package(slides: list[Slide]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    media_names: dict[Path, str] = {}
    media_counter = 1
    with zipfile.ZipFile(TEMPLATE, "r") as zin, zipfile.ZipFile(PPTX, "w", zipfile.ZIP_DEFLATED) as zout:
        skip_prefixes = ("ppt/slides/", "ppt/media/", "ppt/notesSlides/")
        skip_names = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "docProps/app.xml"}
        for info in zin.infolist():
            if info.filename in skip_names or info.filename.startswith(skip_prefixes):
                continue
            zout.writestr(info, zin.read(info.filename))
        for i, slide in enumerate(slides, start=1):
            zout.writestr(f"ppt/slides/slide{i}.xml", slide.render(i, len(slides)))
            rel_targets = [("http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
            for img_path in slide.images:
                img_path = img_path.resolve()
                if img_path not in media_names:
                    suffix = img_path.suffix.lower()
                    if suffix not in (".png", ".jpg", ".jpeg"):
                        suffix = ".png"
                    media_names[img_path] = f"image{media_counter}{suffix}"
                    media_counter += 1
                rel_targets.append(("http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_names[img_path]}"))
            zout.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rel_xml(rel_targets))
        for img_path, name in media_names.items():
            zout.write(img_path, f"ppt/media/{name}")
        zout.writestr("[Content_Types].xml", content_types(len(slides), list(media_names.values())))
        zout.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zout.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        app_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat>'
            f'<Slides>{len(slides)}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides>'
            '<MMClips>0</MMClips><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate>'
            '<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>'
        )
        zout.writestr("docProps/app.xml", app_xml)


def write_notes(slides: list[Slide]) -> None:
    lines = ["# MCOD-FM-RobustBench Speaker Notes", ""]
    for i, s in enumerate(slides, start=1):
        lines.append(f"## {i}. {s.title}")
        lines.extend(s.notes or ["Presenter note to add."])
        lines.append("")
    NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows = read_csv(OUT / "msiresults_eval_summary.csv")
    make_bar_chart(metric_rows, OUT / "local_metrics_bar.png")
    make_rgb_msi_chart(OUT / "rgb_msi_chart.png")
    make_qual_grid(OUT / "qualitative_grid.png")
    slides = build_slides()
    build_package(slides)
    write_notes(slides)
    print(PPTX)
    print(NOTES)


if __name__ == "__main__":
    main()
