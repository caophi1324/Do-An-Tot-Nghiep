#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arrange_drone_pcb.py  -  Sắp xếp linh kiện board drone (FC giữa + 4 ESC ở 4 góc), 2 lớp.
Viết cho KiCad 7 / 8 / 9 (file của bạn là KiCad 10, chưa test trên bản này) (pcbnew Python API).

CÁCH CHẠY
  1) Trong KiCad PCB Editor:  Tools > Scripting Console, gõ:
        exec(open(r"D:\\duong_dan\\arrange_drone_pcb.py", encoding="utf-8").read())
     (nhớ "Update PCB from Schematic" trước để footprint đã có đủ, và LƯU/backup file .kicad_pcb)
  2) Hoặc chạy ngoài (dùng python của KiCad):
        python arrange_drone_pcb.py Drone_ver2.kicad_pcb      ->  Drone_ver2_arranged.kicad_pcb

Ý TƯỞNG
  - Linh kiện chia nhóm: ESC1..ESC4 (tách theo tên net ESC1_..ESC4_ trong sheet ESC) và FC (phần còn lại).
  - FC: các linh kiện chính (ESP32, header, IMU, nguồn...) đặt theo toạ độ cố định (FC_FIXED),
        linh kiện nhỏ tự xếp quanh linh kiện chính mà nó nối mạng tới.
  - ESC: bố trí 1 khối "mẫu" trong hệ toạ độ cục bộ (x hướng ra góc), rồi xoay 45/135/225/315 độ
        và đặt tại 4 góc -> 4 khối giống hệt nhau như trong ảnh.
  - 2 lớp: mặc định linh kiện chính ở Top, tụ (C) của ESC đặt ở Bottom (chỉnh ở BOTTOM_PREFIXES).
"""
import math
import re
import sys

import pcbnew

# =====================================================================================
#                                   CẤU HÌNH (chỉnh ở đây)
# =====================================================================================
BOARD_CENTER_MM = (150.0, 100.0)      # tâm board trong KiCad (mm)
ESC_DISTANCE_MM = 52.0                # từ tâm board tới tâm mỗi khối ESC (theo đường chéo)

# Hướng (độ, ngược chiều kim đồng hồ nhìn từ trên xuống): ESC1 = góc phải-trên ...
ESC_ANGLES = {"ESC1": 45, "ESC2": 135, "ESC3": 225, "ESC4": 315}

# Trong file của bạn 4 kênh ESC nằm chung 1 sheet "ESC" -> tách theo tên net ESC1_..ESC4_.
ESC_SHEET_REGEX = r"esc"
# Linh kiện ESC chỉ nối GND/+BATT/+5V (tụ lọc...) được chia đều cho 4 khối nếu số lượng chia hết cho 4,
# phần còn lại (LDO TPS72733, LED, tụ của LDO...) coi là dùng chung -> đưa về nhóm FC.

# --- FC: toạ độ so với tâm board (mm, x sang phải, y xuống), góc, mặt Bottom?  (ước lượng từ ảnh) ---
FC_FIXED = {
    "U30": (0.0, -22.0, 0, False),     # ESP32-S3, sát cạnh trên
    "SW1": (-15.0, -25.0, 0, False),   # 2 nút nhấn hai bên ESP32 (đổi SW1/SW2 nếu ngược)
    "SW2": (14.5, -25.0, 0, False),
    "J1": (-27.0, -25.0, 0, False),    # USB-C  (ảnh không rõ vị trí -> chỉnh tay)
    "D2": (-17.0, -6.0, 0, False),     # LED WS2812B
    "M1": (0.0, 3.0, 0, False),        # GY-91
    "M2": (0.0, 19.0, 0, False),       # module MP1584
    "J5": (-36.5, -2.0, 0, False),     # header trái (UART2)
    "J4": (-31.3, -3.0, 0, False),     # UART1
    "J3": (-31.3, 7.0, 0, False),      # UART0
    "J6": (-36.5, 19.0, 0, False),     # I2C
    "J2": (-31.3, 19.0, 0, False),     # RC in
    "U31": (32.0, -4.0, 90, False),    # microSD bên phải
    "C79": (36.5, 10.0, 0, False),     # tụ 1000uF SMD
    "BZ1": (34.5, 23.0, 0, False),     # buzzer
    "U33": (13.5, 30.0, 0, False),     # INA219
    "Q3": (-2.5, 21.0, 0, False),
    "C76": (-2.0, 35.0, 0, False),     # tụ 1000uF THT
    "U32": (-14.0, 32.0, 0, False),    # AMS1117
    "J11": (0.0, 43.0, 0, False),      # XT30
    "U1": (22.0, 28.0, 0, False),      # LDO TPS72733 (dùng chung cho 4 ESC)
}

# --- ESC: vị trí cục bộ (x hướng ra góc, y ngang nhánh) ---
ESC_PHASE_PITCH_MM = 9.0
ESC_X = {"mcu": -7.0, "drv": 1.5, "siz": 8.5, "flag": 16.0}

# --- 2 lớp: ref bắt đầu bằng chữ nào (không phải linh kiện đã cố định) thì đưa xuống Bottom ---
BOTTOM_PREFIXES = {"FC": (), "ESC": ("C",)}
FORCE_BOTTOM = set()                  # ví dụ {"C79"}
FORCE_TOP = set()

CLEARANCE_MM = 0.35                   # khoảng hở giữa các linh kiện khi tự xếp
SPIRAL_STEP_MM = 0.7
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


def fp_radius_mm(fp):
    try:
        bb = fp.GetBoundingBox(False)
    except TypeError:
        bb = fp.GetBoundingBox()
    return 0.5 * max(pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight()))


def is_bottom(fp):
    return fp.GetLayer() == pcbnew.B_Cu


def flip_to(fp, bottom):
    if is_bottom(fp) == bottom:
        return
    pos = fp.GetPosition()
    try:  # KiCad 9
        fp.Flip(pos, pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
    except (AttributeError, TypeError):  # KiCad 7/8
        fp.Flip(pos, False)


def set_pose(fp, x_mm, y_mm, angle, bottom):
    flip_to(fp, bottom)
    fp.SetPosition(vec(x_mm, y_mm))
    fp.SetOrientationDegrees(angle)


def to_global(lx, ly, lang, origin, phi_deg):
    """local (x hướng ra góc) -> global; xoay ngược chiều kim đồng hồ phi độ (trục y của KiCad hướng xuống)."""
    p = math.radians(phi_deg)
    gx = origin[0] + lx * math.cos(p) + ly * math.sin(p)
    gy = origin[1] - lx * math.sin(p) + ly * math.cos(p)
    return gx, gy, (lang + phi_deg) % 360


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
def find_free_spot(cx, cy, r, occupied, start_r, seed):
    """Xoắn ốc quanh (cx,cy), trả về (x,y) đầu tiên không chạm 'occupied' = [(x,y,r)]."""
    ring = 0
    while ring < 400:
        rad = start_r + ring * SPIRAL_STEP_MM
        n = max(8, int(2 * math.pi * rad / SPIRAL_STEP_MM))
        phase = (seed * 2.399963) % (2 * math.pi)     # góc vàng -> rải đều
        for i in range(n):
            a = phase + 2 * math.pi * i / n
            x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
            if all(math.hypot(x - ox, y - oy) >= r + orr + CLEARANCE_MM for ox, oy, orr in occupied):
                return x, y
        ring += 1
    return cx, cy


def layout_group(fps, fixed, kind, board_maps):
    """
    fps   : footprints của nhóm (đã sort)
    fixed : {ref: (x,y,angle,bottom)} toạ độ cục bộ của linh kiện chính
    trả về: {ref: (x,y,angle,bottom)} cục bộ cho TẤT CẢ footprint trong nhóm
    """
    net_pads, fp_nets = board_maps
    poses = dict(fixed)
    occ = {False: [], True: []}
    radius = {f.GetReference(): fp_radius_mm(f) for f in fps}
    by_ref = {f.GetReference(): f for f in fps}

    for ref, (x, y, ang, bot) in fixed.items():
        occ[bot].append((x, y, radius[ref]))

    anchors = set(fixed)
    todo = [f for f in fps if f.GetReference() not in poses]
    # linh kiện lớn/kết nối nhiều đặt trước
    todo.sort(key=lambda f: (-radius[f.GetReference()], natural_key(f.GetReference())))

    for idx, fp in enumerate(todo):
        ref = fp.GetReference()
        best, best_s = None, -1.0
        for other, (ox, oy, _, _) in poses.items():
            s = affinity(ref, other, net_pads, fp_nets) * (2.0 if other in anchors else 1.0)
            if s > best_s:
                best, best_s = other, s
        if best is None:
            ax, ay, ar = 0.0, 0.0, 0.0
        else:
            ax, ay, _, _ = poses[best]
            ar = radius[best]

        pre = prefix(ref)
        bottom = (ref in FORCE_BOTTOM) or (
            ref not in FORCE_TOP and pre in BOTTOM_PREFIXES.get(kind, ())
        )
        r = radius[ref]
        x, y = find_free_spot(ax, ay, r, occ[bottom], ar + r + CLEARANCE_MM, idx)
        poses[ref] = (x, y, 0, bottom)
        occ[bottom].append((x, y, r))
    return poses


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
    groups = group_fps(board)
    maps = build_net_maps(board)
    fp_nets = maps[1]
    cx, cy = BOARD_CENTER_MM

    print("=== Số linh kiện mỗi nhóm ===")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    # ------------------------------ FC ở giữa ------------------------------
    fc = groups["FC"]
    fc_refs = {f.GetReference() for f in fc}
    fixed = {}
    for ref, pose in FC_FIXED.items():
        if ref in fc_refs:
            fixed[ref] = pose
        else:
            print(f"  [!] FC_FIXED: không thấy {ref} trong nhóm FC -> bỏ qua")
    fc_local = layout_group(fc, fixed, "FC", maps)
    by_ref = {f.GetReference(): f for f in fc}
    for ref, (x, y, a, b) in fc_local.items():
        gx, gy, ga = to_global(x, y, a, (cx, cy), 0)
        set_pose(by_ref[ref], gx, gy, ga, b)

    # ------------------------------ 4 khối ESC ------------------------------
    template, template_keys = None, None
    for name, phi in ESC_ANGLES.items():
        fps = groups[name]
        if not fps:
            print(f"  [!] {name}: không có linh kiện")
            continue
        origin = (cx + ESC_DISTANCE_MM * math.cos(math.radians(phi)),
                  cy - ESC_DISTANCE_MM * math.sin(math.radians(phi)))

        # sắp theo "vai trò" (footprint, value, tên net chuẩn hoá) để 4 khối khớp nhau
        ordered = sorted(fps, key=lambda f: (role_key(f, fp_nets), natural_key(f.GetReference())))
        keys = [role_key(f, fp_nets) for f in ordered]
        if template is not None and keys == template_keys:
            local = {f.GetReference(): template[i] for i, f in enumerate(ordered)}
        else:
            if template is not None:
                print(f"  [!] {name}: thành phần khác ESC1 -> tự xếp lại riêng")
            local = layout_group(fps, esc_anchor_poses(fps), "ESC", maps)
            if template is None:
                template = [local[f.GetReference()] for f in ordered]
                template_keys = keys

        by_ref = {f.GetReference(): f for f in fps}
        for ref, (x, y, a, b) in local.items():
            gx, gy, ga = to_global(x, y, a, origin, phi)
            set_pose(by_ref[ref], gx, gy, ga, b)

    pcbnew.Refresh()
    print("Xong. Hãy kiểm tra, chỉnh tay, rồi vẽ outline + chạy DRC.")


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
