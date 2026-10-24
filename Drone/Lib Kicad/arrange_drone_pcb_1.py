#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arrange_drone_pcb.py  -  Sắp xếp linh kiện board drone: FC ở giữa + 4 ESC ở 4 góc (2 lớp).
Viết cho KiCad 7/8/9/10 (pcbnew Python API).

CÁCH CHẠY (trong PCB Editor: Tools > Scripting Console, gõ SÁT LỀ TRÁI, không có khoảng trắng đầu dòng):
    exec(open(r"D:\\Drone\\Lib Kicad\\arrange_drone_pcb.py", encoding="utf-8").read())
Hoặc chạy ngoài:  python arrange_drone_pcb.py Drone_ver2.kicad_pcb  ->  Drone_ver2_arranged.kicad_pcb
(Backup file .kicad_pcb trước khi chạy.)

Ý TƯỞNG
  - Đọc OUTLINE thật từ lớp Edge.Cuts (hình X) -> tâm board, hướng 4 nhánh và giới hạn vùng đặt linh kiện.
  - Nhóm ESC1..ESC4 (tách theo tên net ESC1_..ESC4_) đặt ở 4 nhánh, ƯU TIÊN LỚP TOP,
    mọi linh kiện nằm TRONG outline (cách mép >= EDGE_MARGIN_MM). Hết chỗ ở Top mới tràn xuống Bottom.
  - FC (phần còn lại) nằm trong ô vuông giữa, không lấn vào vùng dành cho ESC.
  - Khối ESC1 được bố trí 1 lần rồi xoay 90 độ sao chép sang 3 khối còn lại.
"""
import math
import re
import sys

import pcbnew

# =====================================================================================
#                                   CẤU HÌNH (chỉnh ở đây)
# =====================================================================================
BOARD_CENTER_MM = None                # None = tự lấy tâm bbox của outline; hoặc (x, y) mm
ESC_DISTANCE_MM = None                # None = tự tính: (đỉnh nhánh) - ESC_BLOCK_HALF_LEN_MM
ESC_BLOCK_HALF_LEN_MM = 14.0
ESC_ZONE_START_MM = 33.0              # vùng ESC bắt đầu cách tâm board (dọc trục nhánh) từ đây trở ra
EDGE_MARGIN_MM = 0.5                  # khoảng cách tối thiểu từ linh kiện tới mép board
ALLOW_OVERFLOW_TO_BOTTOM = True       # Top hết chỗ thì cho phép xuống Bottom (ESC ưu tiên Top)

# Hướng (độ, ngược chiều kim đồng hồ nhìn từ trên xuống): ESC1 = phải-trên ...
ESC_ANGLES = {"ESC1": 45, "ESC2": 135, "ESC3": 225, "ESC4": 315}

# Tất cả 4 kênh ESC nằm chung sheet "ESC" -> tách theo tên net ESC1_..ESC4_.
ESC_SHEET_REGEX = r"esc"

# --- FC: toạ độ so với tâm board (mm, x sang phải, y xuống), góc, Bottom?  (ước lượng từ ảnh) ---
# Nếu vị trí nằm ngoài outline, script tự dịch dần về tâm cho tới khi lọt vào trong.
FC_FIXED = {
    "U30": (0.0, -19.0, 0, False),     # ESP32-S3, sát cạnh trên
    "SW1": (-14.0, -28.5, 0, False),   # 2 nút nhấn hai bên ESP32 (đổi SW1/SW2 nếu ngược)
    "SW2": (12.5, -28.5, 0, False),
    "J1": (-24.0, -27.5, 0, False),    # USB-C (ảnh không rõ -> chỉnh tay)
    "D2": (-15.0, -12.0, 0, False),    # LED WS2812B
    "M1": (0.0, -4.0, 0, False),       # GY-91
    "M2": (0.0, 8.0, 0, False),        # module MP1584
    "J5": (-29.5, -9.0, 0, False),     # UART2
    "J4": (-25.0, -11.5, 0, False),    # UART1
    "J3": (-25.0, -0.7, 0, False),     # UART0
    "J6": (-29.5, 8.5, 0, False),      # I2C
    "J2": (-25.0, 8.8, 0, False),      # RC in
    "U31": (23.5, -10.0, 90, False),   # microSD
    "C79": (26.0, 6.0, 0, False),      # tụ 1000uF SMD
    "BZ1": (24.0, 19.0, 0, False),     # buzzer
    "U33": (12.0, 27.0, 0, False),     # INA219
    "Q3": (-3.0, 18.0, 0, False),
    "C76": (-9.0, 26.0, 0, False),     # tụ 1000uF THT
    "J11": (4.0, 29.0, 0, False),      # XT30
    "U32": (-17.0, 27.0, 0, False),    # AMS1117
    "U1": (20.0, 26.0, 0, False),      # LDO TPS72733 (dùng chung cho 4 ESC)
}

# Cặp linh kiện chính được phép chồng nhau (M2 nằm dưới/ngay cạnh M1 như trong ảnh)
OVERLAP_OK = {frozenset(("M1", "M2"))}

# --- ESC: vị trí cục bộ (x hướng ra đỉnh nhánh, y ngang nhánh), tính từ tâm khối ---
ESC_PHASE_PITCH_MM = 6.8
ESC_X = {"mcu": -8.0, "drv": -1.0, "siz": 5.0, "flag": 11.0}

# --- 2 lớp ---
BOTTOM_PREFIXES = {"FC": (), "ESC": ()}   # ESC: mọi thứ ưu tiên Top (đổi thành ("C",) nếu muốn tụ ở Bottom)
FORCE_BOTTOM = set()                  # ví dụ {"C79"}
FORCE_TOP = set()

CLEARANCE_MM = 0.25                   # khoảng hở giữa các linh kiện khi tự xếp
SPIRAL_STEP_MM = 0.6
RING_LIMIT = 140
SKIP_LOCKED = True                    # bỏ qua footprint đã khoá


# =====================================================================================
#                                      HÀM PHỤ
# =====================================================================================
def mm(v):
    return int(pcbnew.FromMM(v))


def vec(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))


def natural_key(ref):
    m = re.match(r"([A-Za-z_]+)(\d+)", ref)
    return (m.group(1), int(m.group(2))) if m else (ref, 0)


def prefix(ref):
    m = re.match(r"[A-Za-z_]+", ref)
    return m.group(0) if m else ""


def expand_refs(spec):
    """'R1-R3, C5' -> {'R1','R2','R3','C5'}"""
    out = set()
    for tok in re.split(r"[,\s;]+", spec.strip()):
        if not tok:
            continue
        m = re.match(r"^([A-Za-z_]+)(\d+)-(?:[A-Za-z_]+)?(\d+)$", tok)
        if m:
            for i in range(int(m.group(2)), int(m.group(3)) + 1):
                out.add(f"{m.group(1)}{i}")
        else:
            out.add(tok)
    return out


def fp_size_mm(fp):
    """Kích thước (rộng, cao) của footprint khi góc = 0 (đo tạm ở góc 0 để không bị sai khi đã xoay)."""
    try:
        ang = fp.GetOrientationDegrees()
    except AttributeError:
        ang = 0.0
    if ang:
        fp.SetOrientationDegrees(0)
    try:
        bb = fp.GetBoundingBox(False)
    except TypeError:
        bb = fp.GetBoundingBox()
    w, h = pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())
    if ang:
        fp.SetOrientationDegrees(ang)
    return w, h


def rect_pts(x, y, hw, hh):
    """8 điểm kiểm tra (4 góc + 4 trung điểm cạnh) của hình chữ nhật tâm (x,y), nửa cạnh hw, hh."""
    return [(x + dx * hw, y + dy * hh) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def is_bottom(fp):
    return fp.GetLayer() == pcbnew.B_Cu


def flip_to(fp, bottom):
    if is_bottom(fp) == bottom:
        return
    pos = fp.GetPosition()
    try:  # KiCad 9+
        fp.Flip(pos, pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
    except (AttributeError, TypeError):  # KiCad 7/8
        fp.Flip(pos, False)


def set_pose(fp, x_mm, y_mm, angle, bottom):
    flip_to(fp, bottom)
    fp.SetPosition(vec(x_mm, y_mm))
    fp.SetOrientationDegrees(angle)


def to_global(lx, ly, lang, origin, phi_deg):
    """local (x hướng ra đỉnh nhánh) -> global; xoay ngược chiều kim đồng hồ phi độ (trục y KiCad hướng xuống)."""
    p = math.radians(phi_deg)
    gx = origin[0] + lx * math.cos(p) + ly * math.sin(p)
    gy = origin[1] - lx * math.sin(p) + ly * math.cos(p)
    return gx, gy, (lang + phi_deg) % 360


# ------------------------------------- OUTLINE (Edge.Cuts) -------------------------------------
class Outline:
    def __init__(self, pts):
        self.pts = pts
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self.center = ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def contains(self, x, y):
        inside = False
        pts = self.pts
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    def edge_dist(self, x, y):
        best = 1e9
        pts = self.pts
        for i in range(len(pts)):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % len(pts)]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
            best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
        return best

    def ok(self, x, y, r):
        return self.contains(x, y) and self.edge_dist(x, y) >= r + EDGE_MARGIN_MM

    def ok_pts(self, pts):
        return all(self.contains(x, y) and self.edge_dist(x, y) >= EDGE_MARGIN_MM for x, y in pts)


def read_outline(board):
    segs = []
    skipped = 0
    for d in board.GetDrawings():
        try:
            if d.GetLayer() != pcbnew.Edge_Cuts:
                continue
            if d.GetShape() != pcbnew.SHAPE_T_SEGMENT:
                skipped += 1
                continue
            a, b = d.GetStart(), d.GetEnd()
            segs.append(((pcbnew.ToMM(a.x), pcbnew.ToMM(a.y)), (pcbnew.ToMM(b.x), pcbnew.ToMM(b.y))))
        except Exception:
            continue
    if skipped:
        print(f"  [!] Edge.Cuts có {skipped} hình không phải đoạn thẳng (cung/rect/poly) -> bỏ qua")
    if len(segs) < 3:
        return None
    key = lambda p: (round(p[0] * 100), round(p[1] * 100))      # lưới 0.01 mm
    ends = {}
    for i, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append(i)
        ends.setdefault(key(b), []).append(i)
    used = {0}
    pts = [segs[0][0]]
    cur = segs[0][1]
    while True:
        nxt = None
        for i in ends.get(key(cur), []):
            if i not in used:
                nxt = i
                break
        if nxt is None:
            break
        used.add(nxt)
        a, b = segs[nxt]
        if key(a) == key(cur):
            pts.append(a)
            cur = b
        else:
            pts.append(b)
            cur = a
    if len(used) != len(segs):
        print(f"  [!] Outline: chỉ nối được {len(used)}/{len(segs)} đoạn Edge.Cuts (outline hở/nhiều vòng?)")
    return Outline(pts) if len(pts) >= 3 else None


def axis_tip(outline, center, phi):
    p = math.radians(phi)
    return max((x - center[0]) * math.cos(p) - (y - center[1]) * math.sin(p) for x, y in outline.pts)


def fpid_name(fp):
    return str(fp.GetFPID().GetLibItemName())


def split_esc(rows, n_blocks=4):
    """
    rows: list of dict(ref, nets:set, sheet:str) của các linh kiện thuộc sheet ESC.
    trả về (blocks {1..n: [ref]}, shared [ref]).
    """
    assigned = {}
    # 1) trực tiếp theo tên net ESC<k>_...
    for r in rows:
        cnt = {}
        for n in r["nets"]:
            for m in re.findall(r"ESC(\d)", n):
                cnt[int(m)] = cnt.get(int(m), 0) + 1
        if cnt:
            assigned[r["ref"]] = max(cnt, key=cnt.get)
    # 2) lan truyền qua net cục bộ (<=3 chân, không phải GND / +rail / ESCx)
    pads = {}
    for r in rows:
        for n in r["nets"]:
            pads[n] = pads.get(n, 0) + 1
    local = lambda n: pads[n] <= 3 and not re.match(r"^(GND|\+)", n) and "ESCx" not in n
    changed = True
    while changed:
        changed = False
        for r in rows:
            if r["ref"] in assigned:
                continue
            for n in r["nets"]:
                if not local(n):
                    continue
                for o in rows:
                    if o["ref"] in assigned and n in o["nets"]:
                        assigned[r["ref"]] = assigned[o["ref"]]
                        changed = True
                        break
                if r["ref"] in assigned:
                    break
    # 3) còn lại: chia đều theo nhóm giống nhau nếu chia hết cho n_blocks, không thì dùng chung
    rest = [r for r in rows if r["ref"] not in assigned]
    buckets = {}
    for r in rest:
        buckets.setdefault((r["fpid"], r["val"], tuple(sorted(r["nets"]))), []).append(r)
    shared = []
    for lst in buckets.values():
        lst.sort(key=lambda r: natural_key(r["ref"]))
        if len(lst) >= n_blocks and len(lst) % n_blocks == 0:
            per = len(lst) // n_blocks
            for i, r in enumerate(lst):
                assigned[r["ref"]] = i // per + 1
        else:
            shared += [r["ref"] for r in lst]
    blocks = {k: [] for k in range(1, n_blocks + 1)}
    for ref, k in assigned.items():
        blocks.setdefault(k, []).append(ref)
    return blocks, shared


def group_fps(board):
    fps = [f for f in board.GetFootprints() if not (SKIP_LOCKED and f.IsLocked())]
    by_ref = {f.GetReference(): f for f in fps}
    rows = []
    for f in fps:
        try:
            sheet = f.GetSheetname() or ""
        except AttributeError:
            sheet = ""
        if re.search(ESC_SHEET_REGEX, sheet, re.I):
            rows.append(dict(ref=f.GetReference(), fpid=fpid_name(f), val=f.GetValue(), sheet=sheet,
                             nets={p.GetNetname() for p in f.Pads() if p.GetNetname()}))
    blocks, shared = split_esc(rows, len(ESC_ANGLES))
    groups = {name: [] for name in ESC_ANGLES}
    used = set()
    for name in sorted(ESC_ANGLES):
        k = int(name[3:])
        groups[name] = [by_ref[r] for r in blocks.get(k, [])]
        used.update(blocks.get(k, []))
    groups["FC"] = [f for f in fps if f.GetReference() not in used]
    print(f"  ESC dùng chung (đưa về FC): {sorted(shared, key=natural_key)}")
    for k in groups:
        groups[k].sort(key=lambda f: natural_key(f.GetReference()))
    return groups


# ---------------------------- kết nối mạng để gom linh kiện ----------------------------
def build_net_maps(board):
    net_pads = {}
    fp_nets = {}
    for fp in board.GetFootprints():
        nets = set()
        for pad in fp.Pads():
            n = pad.GetNetname()
            if n:
                nets.add(n)
                net_pads[n] = net_pads.get(n, 0) + 1
        fp_nets[fp.GetReference()] = nets
    return net_pads, fp_nets


def affinity(ref_a, ref_b, net_pads, fp_nets):
    s = 0.0
    for n in fp_nets[ref_a] & fp_nets[ref_b]:
        s += 1.0 / max(1, net_pads[n] - 1)   # mạng càng ít chân càng "gần" nhau
    return s


# ------------------------------------- bố trí cục bộ -------------------------------------
def overlaps(x, y, hw, hh, occupied):
    for ox, oy, ohw, ohh in occupied:
        if abs(x - ox) < hw + ohw + CLEARANCE_MM and abs(y - oy) < hh + ohh + CLEARANCE_MM:
            return True
    return False


def find_free_spot(cx, cy, hw, hh, occupied, start_r, seed, ok):
    """Xoắn ốc quanh (cx,cy); trả về (x,y) đầu tiên hợp lệ (ok) và không chạm 'occupied' = [(x,y,hw,hh)]."""
    phase = (seed * 2.399963) % (2 * math.pi)     # góc vàng -> rải đều
    for ring in range(RING_LIMIT):
        rad = start_r + ring * SPIRAL_STEP_MM
        n = max(8, int(2 * math.pi * rad / SPIRAL_STEP_MM))
        for i in range(n):
            a = phase + 2 * math.pi * i / n
            x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
            if ok(x, y, hw, hh) and not overlaps(x, y, hw, hh, occupied):
                return x, y
    return None


def layout_group(fps, fixed, kind, board_maps, ok, shift):
    """
    fixed : {ref: (x,y,angle,bottom)} toạ độ cục bộ của linh kiện chính
    ok    : ok(x, y, hw, hh) -> True nếu hình chữ nhật (tâm x,y; nửa cạnh hw,hh) nằm trong vùng cho phép
    shift : (x,y) -> (dx,dy) hướng dịch linh kiện chính vào trong khi nó lọt ra ngoài
    trả về (poses {ref: (x,y,angle,bottom)}, danh sách ref không xếp được trong vùng)
    """
    net_pads, fp_nets = board_maps
    poses = {}
    occ = {False: [], True: []}
    half = {}
    for f in fps:
        w, h = fp_size_mm(f)
        half[f.GetReference()] = (w / 2.0, h / 2.0)
    failed = []
    placed_fixed = []

    def half_at(ref, ang):
        hw, hh = half[ref]
        return (hh, hw) if int(round(ang)) % 180 == 90 else (hw, hh)

    for ref, (x, y, ang, bot) in fixed.items():
        hw, hh = half_at(ref, ang)
        x0, y0 = x, y
        for _ in range(160):
            if ok(x, y, hw, hh):
                break
            dx, dy = shift(x, y)
            x, y = x + 0.5 * dx, y + 0.5 * dy
        else:
            x, y = x0, y0
            failed.append(ref)
        # linh kiện chính chồng lên linh kiện chính khác -> tìm chỗ trống gần nhất
        blockers = [(ox, oy, ohw, ohh) for r2, ox, oy, ohw, ohh, ob in placed_fixed
                    if ob == bot and frozenset((ref, r2)) not in OVERLAP_OK]
        if overlaps(x, y, hw, hh, blockers):
            spot = find_free_spot(x, y, hw, hh, blockers, 0.0, 0, ok)
            if spot:
                x, y = spot
            else:
                failed.append(ref)
        poses[ref] = (x, y, ang, bot)
        placed_fixed.append((ref, x, y, hw, hh, bot))
        occ[bot].append((x, y, hw, hh))

    anchors = set(fixed)
    todo = [f for f in fps if f.GetReference() not in poses]
    todo.sort(key=lambda f: (-max(half[f.GetReference()]), natural_key(f.GetReference())))

    for idx, fp in enumerate(todo):
        ref = fp.GetReference()
        best, best_s = None, -1.0
        for other in poses:
            s = affinity(ref, other, net_pads, fp_nets) * (2.0 if other in anchors else 1.0)
            if s > best_s:
                best, best_s = other, s
        if best is None:
            ax, ay, ar = 0.0, 0.0, 0.0
        else:
            ax, ay, aang, _ = poses[best]
            ar = max(half_at(best, aang))

        hw, hh = half[ref]
        pref_bottom = (ref in FORCE_BOTTOM) or (
            ref not in FORCE_TOP and prefix(ref) in BOTTOM_PREFIXES.get(kind, ()))
        sides = [pref_bottom] + ([not pref_bottom] if ALLOW_OVERFLOW_TO_BOTTOM else [])
        start = ar + max(hw, hh) + CLEARANCE_MM
        spot, side = None, pref_bottom
        for side in sides:
            spot = find_free_spot(ax, ay, hw, hh, occ[side], start, idx, ok)
            if spot:
                break
        if spot is None:                       # không còn chỗ trong khung -> đặt tạm, báo lỗi
            side = pref_bottom
            spot = find_free_spot(ax, ay, hw, hh, occ[side], start, idx, lambda *a: True)
            failed.append(ref)
        poses[ref] = (spot[0], spot[1], 0, side)
        occ[side].append((spot[0], spot[1], hw, hh))
    return poses, failed


def esc_anchor_poses(fps):
    """MCU gần tâm board; mỗi pha A/B/C một hàng: driver -> MOSFET -> pad ra motor (ngoài cùng)."""
    out = {}
    for fp in fps:
        ref, lib = fp.GetReference(), fpid_name(fp)
        nets = {p.GetNetname() for p in fp.Pads()}
        ph = sorted({m for n in nets for m in re.findall(r"ESC\d_([ABC])(?:_|$)", n)})
        if "UFQFPN" in lib:
            out[ref] = (ESC_X["mcu"], 0.0, 0, False)
        elif ph:
            role = ("drv" if "DFN-8" in lib else "siz" if "PowerPAIR" in lib
                    else "flag" if "esc_pad" in lib else None)
            if role:
                i = "ABC".index(ph[0])
                out[ref] = (ESC_X[role], (i - 1) * ESC_PHASE_PITCH_MM, 0, False)
    return out


def role_key(fp, fp_nets):
    nets = []
    for n in fp_nets[fp.GetReference()]:
        if n.startswith("unconnected-"):
            continue
        n = re.sub(r"ESC\d", "ESC#", n)
        n = re.sub(r"Net-\(.*\)", "Net-(*)", n)
        nets.append(n)
    return (fpid_name(fp).replace("_dense", ""), tuple(sorted(nets)))   # không dùng value: vài RN khác giá trị giữa các kênh


# =====================================================================================
#                                          MAIN
# =====================================================================================
def run(board):
    outline = read_outline(board)
    if outline is None:
        print("  [!] Không đọc được outline Edge.Cuts -> KHÔNG kiểm tra được việc nằm trong board")
        outline = Outline([(-1e4, -1e4), (1e4, -1e4), (1e4, 1e4), (-1e4, 1e4)])
        auto_center = (0.0, 0.0)
    else:
        auto_center = outline.center
        print(f"  Outline: {len(outline.pts)} đỉnh, bbox "
              f"{outline.bbox[2] - outline.bbox[0]:.1f} x {outline.bbox[3] - outline.bbox[1]:.1f} mm")
    cx, cy = BOARD_CENTER_MM or auto_center

    groups = group_fps(board)
    maps = build_net_maps(board)
    fp_nets = maps[1]
    print("=== Số linh kiện mỗi nhóm ===")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    # ------------------------------ FC ở giữa ------------------------------
    def fc_ok(x, y, hw, hh):
        pts = rect_pts(x, y, hw, hh)
        for phi in ESC_ANGLES.values():            # chừa vùng cho ESC
            p = math.radians(phi)
            if max(px * math.cos(p) - py * math.sin(p) for px, py in pts) > ESC_ZONE_START_MM:
                return False
        return outline.ok_pts([(cx + px, cy + py) for px, py in pts])

    def to_center(x, y):
        d = math.hypot(x, y) or 1.0
        return -x / d, -y / d

    fc = groups["FC"]
    fc_refs = {f.GetReference() for f in fc}
    fixed = {}
    for ref, pose in FC_FIXED.items():
        if ref in fc_refs:
            fixed[ref] = pose
        else:
            print(f"  [!] FC_FIXED: không thấy {ref} trong nhóm FC -> bỏ qua")
    fc_local, fc_failed = layout_group(fc, fixed, "FC", maps, fc_ok, to_center)
    by_ref = {f.GetReference(): f for f in fc}
    for ref, (x, y, a, b) in fc_local.items():
        set_pose(by_ref[ref], cx + x, cy + y, a, b)
    if fc_failed:
        print(f"  [!] FC: {len(fc_failed)} linh kiện không vừa trong khung: {sorted(fc_failed, key=natural_key)}")

    # ------------------------------ 4 khối ESC ------------------------------
    template, template_keys = None, None
    n_bottom = 0
    esc_failed_all = []
    for name, phi in ESC_ANGLES.items():
        fps = groups[name]
        if not fps:
            print(f"  [!] {name}: không có linh kiện")
            continue
        d0 = ESC_DISTANCE_MM
        if d0 is None:
            d0 = axis_tip(outline, (cx, cy), phi) - ESC_BLOCK_HALF_LEN_MM
        origin = (cx + d0 * math.cos(math.radians(phi)), cy - d0 * math.sin(math.radians(phi)))

        ordered = sorted(fps, key=lambda f: (role_key(f, fp_nets), natural_key(f.GetReference())))
        keys = [role_key(f, fp_nets) for f in ordered]
        if template is not None and keys == template_keys:
            local = {f.GetReference(): template[i] for i, f in enumerate(ordered)}
        else:
            if template is not None:
                print(f"  [!] {name}: thành phần khác ESC1 -> tự xếp lại riêng")

            def esc_ok(lx, ly, hw, hh, origin=origin, phi=phi, d0=d0):
                if lx - hw < ESC_ZONE_START_MM - d0:
                    return False
                return outline.ok_pts([to_global(px, py, 0, origin, phi)[:2]
                                       for px, py in rect_pts(lx, ly, hw, hh)])

            local, failed = layout_group(fps, esc_anchor_poses(fps), "ESC", maps, esc_ok,
                                         lambda x, y: (-1.0, 0.0))
            esc_failed_all += failed
            if template is None:
                template = [local[f.GetReference()] for f in ordered]
                template_keys = keys

        by_ref = {f.GetReference(): f for f in fps}
        for ref, (x, y, a, b) in local.items():
            gx, gy, ga = to_global(x, y, a, origin, phi)
            set_pose(by_ref[ref], gx, gy, ga, b)
            n_bottom += 1 if b else 0

    # ------------------------------ kiểm tra cuối ------------------------------
    outside = []
    for f in board.GetFootprints():
        p = f.GetPosition()
        x, y = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
        if not outline.ok(x, y, 0.0):
            outside.append(f.GetReference())
    print(f"  ESC phải xuống Bottom vì Top hết chỗ: {n_bottom} linh kiện")
    if esc_failed_all:
        print(f"  [!] ESC1 không đủ chỗ trong khung: {sorted(esc_failed_all, key=natural_key)}")
    if outside:
        print(f"  [!] {len(outside)} linh kiện có tâm ngoài/sát mép board: {sorted(outside, key=natural_key)}")
    else:
        print("  Tất cả linh kiện nằm trong outline.")

    pcbnew.Refresh()
    print("Xong. Hãy kiểm tra, chỉnh tay rồi chạy DRC.")


def main():
    try:
        board = pcbnew.GetBoard()
        standalone = board is None
    except Exception:
        standalone = True

    if standalone:
        if len(sys.argv) < 2:
            print("Usage: python arrange_drone_pcb.py board.kicad_pcb")
            return
        src = sys.argv[1]
        board = pcbnew.LoadBoard(src)
        run(board)
        dst = src.replace(".kicad_pcb", "_arranged.kicad_pcb")
        pcbnew.SaveBoard(dst, board)
        print("Đã lưu:", dst)
    else:
        run(board)


main()
