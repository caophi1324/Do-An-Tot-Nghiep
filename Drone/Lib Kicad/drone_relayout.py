# -*- coding: utf-8 -*-
"""
drone_relayout.py  -  Sap xep lai linh kien board Drone (X-frame, ESP32-S3 + 4 ESC)
=====================================================================================
Cach chay (KiCad PCB Editor):
  1. Luu board, tao ban sao du phong (script cung tu copy file *.kicad_pcb ra *_backup_*.kicad_pcb).
  2. Tools > Scripting Console, go:
         exec(open(r"D:/duong_dan/drone_relayout.py", encoding="utf-8").read())
  3. Xem ket qua, chinh THAM SO o phan CONFIG neu can, chay lai (Ctrl+Z khong hoan tac duoc
     script -> hay dung ban backup neu muon quay lai).

Script lam gi:
  * 4 khoi ESC (linh kien nam trong 4 canh cua X): GIU NGUYEN cau truc / huong xoay,
    chi co/gian khoang cach quanh tam canh (ESC_SPACING), tu dong ha he so neu cham vien/chong lan.
  * Phan than (octagon o giua): dat lai theo bo cuc trong anh:
        ESP32 o canh tren (anten thong ra ngoai board), 2 nut EN/BOOT hai ben,
        cot header UART/I2C/RC o canh trai, GY-91 (M1) + module nguon MP1584 (M2) o giua,
        microSD + buzzer + tu 1000uF o canh phai, USB-C + XT30 o canh duoi.
    Linh kien nho duoc dat sat linh kien "chu" cua chung (theo net) bang thuat toan xep
    luoi, khong chong lan, nam trong vien board.
  * Khong dong vao track/via/zone, khong doi mat (F/B) cua linh kien.
"""
import math
import os
import shutil
import time

try:
    import pcbnew
except ImportError:          # chay ngoai KiCad (de test thuat toan)
    pcbnew = None

# ----------------------------------------------------------------------------------
# CONFIG  (don vi mm)
# ----------------------------------------------------------------------------------
ESC_SPACING_ALONG  = 1.15   # he so gian DOC truc canh (tu tam than ra dau canh). 1.0 = giu nguyen. Tu ha neu khong vua.
ESC_SPACING_ACROSS = 1.06   # he so gian NGANG canh. Tu ha neu khong vua.
ESC_EDGE_MARGIN    = 0.45   # le toi thieu tu courtyard ESC den vien board
ESC_INNER_MARGIN   = 0.40   # le toi thieu tu ESC den duong noi canh-than
ESC_GAP            = 0.12   # khoang cach toi thieu giua 2 linh kien ESC

BODY_EDGE_MARGIN   = 0.60   # le tu linh kien phan than den vien board
BODY_GAP           = 0.25   # khoang cach toi thieu giua 2 linh kien phan than
UNDER_MODULE_OK    = True   # cho phep linh kien nho nam DUOI module M1/M2 (giong anh: module chi la pad)
ANTENNA_OVERHANG   = 6.5    # ESP32: phan anten thong ra ngoai canh tren board (mm)
USBC_OVERHANG      = 0.30   # mieng USB-C nhô ra khoi mep board
GRID               = 0.25   # buoc luoi tim cho trong
SEARCH_RADIUS      = 30.0   # ban kinh tim cho toi da

# Vi tri co dinh cua linh kien lon, tinh bang (dx, dy, rot) so voi TAM THAN board
# (dx,dy la toa do tam hop bao cua linh kien; +x sang phai, +y xuong duoi; rot theo KiCad).
FIXED = {
    # --- canh tren: ESP32 (anten thong ra ngoai) + 2 nut: EN ben trai, BOOT ben phai nhu anh ---
    'U30': (0.0,  None, 0),        # dy=None -> tinh tu ANTENNA_OVERHANG
    'SW2': (-14.6, -28.9, 0),      # EN
    'SW1': ( 14.6, -28.9, 0),      # GPIO0 / BOOT
    # --- canh trai: 2 cot header (J5,J6 sat mep; J4,J3,J2 cot trong) ---
    'J5':  (-30.1, -10.2, 0),      # UART2 (6 chan)
    'J6':  (-30.1,   5.9, 0),      # I2C
    'J4':  (-24.9, -15.2, 0),      # UART1
    'J3':  (-24.9,  -3.2, 0),      # UART0
    'J2':  (-24.9,   7.8, 0),      # RC in
    # --- giua: GY-91 va MP1584 dung canh nhau (board nho -> khong xep chong nhu anh duoc) ---
    'M1':  (-13.3,   0.7, 0),
    'M2':  (  4.4,   2.2, 0),
}

# Vi tri mong muon (tim cho trong gan nhat) cho linh kien lon con lai: (dx, dy, rot)
PREFERRED = {
    'U31': ( 23.1,  -8.5, 90),     # microSD sat canh phai
    'C79': ( 26.0,   6.0, 0),      # 1000uF SMD canh phai
    'BZ1': ( 21.0,  19.0, 0),      # buzzer goc duoi phai
    'J11': (  5.5,  25.0, 180),    # XT30 sat canh duoi
    'C76': (-16.5,  17.7, 0),      # 1000uF THT phia duoi GY-91
    'J1':  ( -9.9,  28.3, 0),      # USB-C sat canh duoi
    'D2':  (-19.5, -22.0, 0),      # LED WS2812
    'U32': (-13.0,  20.0, 0),
    'U33': ( 17.0,  14.0, 0),
    'R62': ( 12.0,  12.0, 0),
}
# Linh kien duoc phep nho ra ngoai mep board: ref -> (mep, mm). mep: 'B' duoi, 'R' phai, 'L' trai, 'T' tren
EDGE_ALLOW = {'J1': ('B', 1.0), 'U31': ('R', 0.5), 'J11': ('B', 0.5)}
# Linh kien CAO (khong duoc nam duoi than module M1/M2)
TALL_REFS = ('C76', 'C79', 'BZ1', 'J11', 'J1', 'U31', 'J2', 'J3', 'J4', 'J5', 'J6', 'U30')
# Linh kien chi noi toi GND/+5V/+BATT -> dat canh linh kien chu nay
DEFAULT_ANCHOR = {
    'C70': 'U30', 'C71': 'U30', 'C72': 'U30', 'C73': 'U30', 'C80': 'U30', 'C82': 'U30',
    'C77': 'U30', 'R65': 'U30', 'R66': 'U30',
    'C68': 'J1',  'C69': 'U32', 'C1': 'U32', 'D3': 'U32', 'D5': 'U32', 'C76': 'J11',
}
# Hop bao "tay" cho linh kien ma courtyard khong dung (toa do local, rot 0): (x0, x1, y0, y1)
BOX_OVERRIDE = {
    'U30': (-9.2, 9.2, -12.9, 12.9),
    'M1':  (-1.8, 13.5, -10.7, 10.7),
    'M2':  (-9.1, 9.1, -12.2, 12.2),
    'C76': (-2.8, 7.8, -5.2, 5.2),
    'BZ1': (-2.5, 10.1, -6.3, 6.3),
}
# Linh kien "module": chi coi pad la vat can (khi UNDER_MODULE_OK)
MODULE_REFS = ('M1', 'M2')

# Du phong neu khong doc duoc Edge.Cuts: dinh bat (octagon) cua than board
FALLBACK_OCTAGON = [(115.348, 64.849), (150.638, 64.849), (165.637, 79.867), (165.637, 115.157),
                    (150.619, 130.157), (115.329, 130.157), (100.330, 115.139), (100.330, 79.849)]
FALLBACK_OUTLINE = [(80.177, 59.696), (95.188, 44.704), (115.348, 64.849), (150.638, 64.849),
                    (170.791, 44.697), (185.783, 59.708), (165.637, 79.867), (165.637, 115.157),
                    (185.790, 135.310), (170.779, 150.302), (150.619, 130.157), (115.329, 130.157),
                    (95.177, 150.309), (80.184, 135.298), (100.330, 115.139), (100.330, 79.849)]


# ----------------------------------------------------------------------------------
# HINH HOC
# ----------------------------------------------------------------------------------
def rot_xy(x, y, deg):
    """Quay diem (x,y) local -> toa do man hinh theo quy uoc KiCad (CCW duong, y xuong)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (x * c + y * s, -x * s + y * c)


def unrot_xy(x, y, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (x * c - y * s, x * s + y * c)


def rect_corners(b):
    x0, x1, y0, y1 = b
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def world_poly(box, ax, ay, rot):
    """Hop local -> da giac 4 dinh trong toa do board (anchor tai (ax,ay))."""
    pts = []
    for (x, y) in rect_corners(box):
        rx, ry = rot_xy(x, y, rot)
        pts.append((ax + rx, ay + ry))
    return pts


def aabb(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), max(xs), min(ys), max(ys))


def grow(b, g):
    return (b[0] - g, b[1] + g, b[2] - g, b[3] + g)


def aabb_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def poly_overlap(p, q, gap=0.0):
    """SAT cho 2 da giac loi (co the xoay 45 do). gap > 0: yeu cau cach nhau it nhat gap."""
    for poly in (p, q):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            nx, ny = (y2 - y1), -(x2 - x1)
            l = math.hypot(nx, ny)
            if l == 0:
                continue
            nx, ny = nx / l, ny / l
            pa = [nx * x + ny * y for x, y in p]
            pb = [nx * x + ny * y for x, y in q]
            if max(pa) + gap <= min(pb) or max(pb) + gap <= min(pa):
                return False
    return True


def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def dist_pt_seg(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def dist_to_poly_edges(p, poly):
    return min(dist_pt_seg(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def poly_inside(poly, outline, margin):
    """Tat ca dinh nam trong outline va cach vien >= margin."""
    for p in poly:
        if not point_in_poly(p, outline):
            return False
        if dist_to_poly_edges(p, outline) < margin - 1e-9:
            return False
    # kiem tra them diem giua canh (bat truong hop outline lom)
    n = len(poly)
    for i in range(n):
        m = ((poly[i][0] + poly[(i + 1) % n][0]) / 2, (poly[i][1] + poly[(i + 1) % n][1]) / 2)
        if not point_in_poly(m, outline) or dist_to_poly_edges(m, outline) < margin - 1e-9:
            return False
    return True


def signed_area(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return a / 2


def reflex_vertices(outline):
    """Cac dinh lom cua outline hinh chu X = 8 dinh cua octagon than."""
    sgn = 1 if signed_area(outline) > 0 else -1
    res = []
    n = len(outline)
    for i in range(n):
        p0, p1, p2 = outline[i - 1], outline[i], outline[(i + 1) % n]
        cr = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cr * sgn < 0:
            res.append(p1)
    return res


# ----------------------------------------------------------------------------------
# MO HINH LINH KIEN
# ----------------------------------------------------------------------------------
class Item(object):
    def __init__(self, ref, side, x, y, rot, box, pad_boxes, tht, nets, value=''):
        self.ref, self.side, self.x, self.y, self.rot = ref, side, x, y, rot
        self.box = box                  # hop bao local (rot 0) quanh anchor
        self.pad_boxes = pad_boxes      # list hop pad local (rot 0)
        self.tht = tht                  # co pad xuyen lo
        self.nets = nets                # set ten net
        self.value = value
        self.new = None                 # (x, y, rot) ket qua
        self.group = None               # 'ESC_xx' hoac 'BODY'

    @property
    def local_center(self):
        return ((self.box[0] + self.box[1]) / 2, (self.box[2] + self.box[3]) / 2)

    def anchor_for_center(self, cx, cy, rot):
        lx, ly = self.local_center
        rx, ry = rot_xy(lx, ly, rot)
        return (cx - rx, cy - ry)

    def center_at(self, ax, ay, rot):
        lx, ly = self.local_center
        rx, ry = rot_xy(lx, ly, rot)
        return (ax + rx, ay + ry)

    def obstacles(self, ax, ay, rot):
        """Danh sach (aabb, sides) vat can trong toa do board. 'T' = vung cam linh kien cao."""
        res = []
        sides = 'FB' if self.tht else self.side
        if self.ref in MODULE_REFS and UNDER_MODULE_OK:
            for pb in self.pad_boxes:
                res.append((aabb(world_poly(grow(pb, 0.35), ax, ay, rot)), 'FB'))
            res.append((aabb(world_poly(self.box, ax, ay, rot)), 'T'))
        else:
            if self.ref in TALL_REFS:
                sides = sides + 'T'
            res.append((aabb(world_poly(self.box, ax, ay, rot)), sides))
        return res


class Grid(object):
    """Luoi bam khong gian de kiem tra va cham nhanh."""
    def __init__(self, cell=4.0):
        self.cell = cell
        self.cells = {}
        self.n = 0

    def _rng(self, b):
        c = self.cell
        return (int(math.floor(b[0] / c)), int(math.floor(b[1] / c)),
                int(math.floor(b[2] / c)), int(math.floor(b[3] / c)))

    def add(self, b, sides, tag):
        x0, x1, y0, y1 = self._rng((b[0], b[1], b[2], b[3]))
        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                self.cells.setdefault((i, j), []).append((b, sides, tag))
        self.n += 1

    def hit(self, b, sides, gap):
        gb = grow(b, gap)
        x0, x1, y0, y1 = self._rng(gb)
        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                for (ob, osides, tag) in self.cells.get((i, j), ()):
                    if any(s in osides for s in sides) and aabb_overlap(gb, ob):
                        return tag
        return None


# ----------------------------------------------------------------------------------
# THUAT TOAN
# ----------------------------------------------------------------------------------
class Planner(object):
    def __init__(self, items, outline, octagon, log=print):
        self.items = items
        self.by_ref = dict((it.ref, it) for it in items)
        self.outline = outline
        self.octagon = octagon
        xs = [p[0] for p in octagon]
        ys = [p[1] for p in octagon]
        self.cx = (min(xs) + max(xs)) / 2
        self.cy = (min(ys) + max(ys)) / 2
        self.half = (max(xs) - min(xs)) / 2
        self.log = log
        self.warn = []

    # ---- phan loai ESC / THAN ----
    def classify(self):
        for it in self.items:
            c = it.center_at(it.x, it.y, it.rot)
            if point_in_poly(c, self.octagon):
                it.group = 'BODY'
            else:
                q = ('T' if c[1] < self.cy else 'B') + ('L' if c[0] < self.cx else 'R')
                it.group = 'ESC_' + q

    def chamfer_edge(self, quad):
        """Canh cheo cua octagon doi dien voi canh quad ('TL','TR','BL','BR')."""
        sx = -1 if quad[1] == 'L' else 1
        sy = -1 if quad[0] == 'T' else 1
        best, bd = None, 1e9
        n = len(self.octagon)
        for i in range(n):
            a, b = self.octagon[i], self.octagon[(i + 1) % n]
            if abs(a[0] - b[0]) < 0.05 or abs(a[1] - b[1]) < 0.05:
                continue
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            d = math.hypot(mx - (self.cx + sx * 1e3), my - (self.cy + sy * 1e3))
            if d < bd:
                best, bd = (a, b), d
        return best

    # ---- ESC: gian/co khoang cach, giu cau truc ----
    def esc_pivot(self, quad, members):
        """Tam canh = trung diem tam cac linh kien (theo hop bao toan khoi)."""
        xs, ys = [], []
        for it in members:
            c = it.center_at(it.x, it.y, it.rot)
            xs.append(c[0]); ys.append(c[1])
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    def esc_map(self, quad, it, k, pivot):
        ku, kv = k
        sx = -1 if quad[1] == 'L' else 1
        sy = -1 if quad[0] == 'T' else 1
        ux, uy = sx / math.sqrt(2), sy / math.sqrt(2)      # truc doc canh
        vx, vy = -uy, ux                                    # truc ngang canh
        c = it.center_at(it.x, it.y, it.rot)
        dx, dy = c[0] - pivot[0], c[1] - pivot[1]
        u = dx * ux + dy * uy
        v = dx * vx + dy * vy
        u *= ku
        v *= kv
        return (pivot[0] + u * ux + v * vx, pivot[1] + u * uy + v * vy)

    def esc_violations(self, quad, members, k, pivot):
        """Tra ve tap cac vi pham (linh kien cham vien / lan vao than, cap chong lan) tai he so k."""
        edge = self.chamfer_edge(quad)
        a, b = edge

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        body_sign = 1 if side((self.cx, self.cy)) > 0 else -1
        ln = math.hypot(b[0] - a[0], b[1] - a[1])
        bad = set()
        polys = []
        for it in members:
            nc = self.esc_map(quad, it, k, pivot)
            ax, ay = it.anchor_for_center(nc[0], nc[1], it.rot)
            poly = world_poly(it.box, ax, ay, it.rot)
            if not poly_inside(poly, self.outline, ESC_EDGE_MARGIN):
                bad.add(it.ref)
            elif any(side(p) * body_sign / ln > -ESC_INNER_MARGIN for p in poly):
                bad.add(it.ref)
            polys.append((it, poly))
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                a_, b_ = polys[i], polys[j]
                if a_[0].side != b_[0].side and not (a_[0].tht or b_[0].tht):
                    continue
                if poly_overlap(a_[1], b_[1], ESC_GAP):
                    bad.add((a_[0].ref, b_[0].ref))
        return bad

    def esc_valid(self, quad, members, k, pivot, base=None):
        bad = self.esc_violations(quad, members, k, pivot)
        if base is not None:
            bad = bad - base          # vi pham da co tu truoc (bo cuc goc) thi khong tinh
        return (not bad), bad

    def do_esc(self):
        for quad in ('TL', 'TR', 'BL', 'BR'):
            members = [it for it in self.items if it.group == 'ESC_' + quad]
            if not members:
                continue
            pivot = self.esc_pivot(quad, members)
            base = self.esc_violations(quad, members, (1.0, 1.0), pivot)
            best = None
            ku_list = [round(ESC_SPACING_ALONG - 0.01 * i, 3) for i in range(int((ESC_SPACING_ALONG - 0.90) / 0.01) + 1)]
            kv_list = [round(ESC_SPACING_ACROSS - 0.01 * i, 3) for i in range(int((ESC_SPACING_ACROSS - 0.90) / 0.01) + 1)]
            for kv in kv_list:
                for ku in ku_list:
                    if best and ku * kv <= best[0]:
                        break
                    ok, _ = self.esc_valid(quad, members, (ku, kv), pivot, base)
                    if ok:
                        best = (ku * kv, ku, kv)
                        break
            if not best:
                self.warn.append('ESC %s: khong tim duoc he so hop le, giu nguyen vi tri goc.' % quad)
                for it in members:
                    it.new = (it.x, it.y, it.rot)
                continue
            _, ku, kv = best
            for it in members:
                nc = self.esc_map(quad, it, (ku, kv), pivot)
                ax, ay = it.anchor_for_center(nc[0], nc[1], it.rot)
                it.new = (ax, ay, it.rot)
            self.log('ESC %s: %d linh kien, gian doc canh x%.2f, ngang canh x%.2f' % (quad, len(members), ku, kv))

    # ---- THAN ----
    def body_ok(self, poly, margin):
        return poly_inside(poly, self.octagon, margin)

    def build_nets(self, body):
        self.net_members = {}
        for it in self.items:
            for n in it.nets:
                self.net_members.setdefault(n, set()).add(it.ref)

    def weight_to(self, ref, placed):
        it = self.by_ref[ref]
        w = {}
        for n in it.nets:
            mem = self.net_members.get(n, ())
            if len(mem) > 12 or len(mem) < 2:
                continue
            for o in mem:
                if o != ref and o in placed:
                    w[o] = w.get(o, 0.0) + 1.0 / (len(mem) - 1)
        return w

    def do_body(self):
        body = [it for it in self.items if it.group == 'BODY']
        self.build_nets(body)
        grid = Grid()
        placed = {}
        cx, cy = self.cx, self.cy

        # ESC la vat can (de than khong de len ESC)
        for it in self.items:
            if it.group.startswith('ESC_') and it.new:
                for (b, sd) in it.obstacles(it.new[0], it.new[1], it.new[2]):
                    grid.add(b, sd, it.ref)

        def put(it, ax, ay, rot):
            it.new = (ax, ay, rot)
            placed[it.ref] = it
            for (b, sd) in it.obstacles(ax, ay, rot):
                grid.add(b, sd, it.ref)

        # 1) vi tri co dinh
        for ref, (dx, dy, rot) in FIXED.items():
            it = self.by_ref.get(ref)
            if it is None or it.group != 'BODY':
                continue
            if ref == 'U30' and dy is None:
                top = self.cy - self.half
                h = (it.box[3] - it.box[2])
                dy = (top - ANTENNA_OVERHANG + h / 2) - self.cy
            ax, ay = it.anchor_for_center(cx + dx, cy + dy, rot)
            put(it, ax, ay, rot)

        # 2) linh kien lon uu tien: tim cho gan vi tri mong muon
        big = [r for r in PREFERRED if r in self.by_ref and self.by_ref[r].group == 'BODY']
        big.sort(key=lambda r: -(self.by_ref[r].box[1] - self.by_ref[r].box[0]) * (self.by_ref[r].box[3] - self.by_ref[r].box[2]))
        for ref in big:
            it = self.by_ref[ref]
            dx, dy, rot = PREFERRED[ref]
            self.search_place(it, (cx + dx, cy + dy), [rot], grid, placed, put)

        # 3) linh kien nho theo ket noi
        rest = [it for it in body if it.ref not in placed]
        remaining = set(it.ref for it in rest)
        while remaining:
            best, bw = None, -1.0
            for r in sorted(remaining):
                w = sum(self.weight_to(r, placed).values())
                it = self.by_ref[r]
                score = w + 0.001 * (it.box[1] - it.box[0]) * (it.box[3] - it.box[2])
                if r in DEFAULT_ANCHOR and DEFAULT_ANCHOR[r] in placed:
                    score += 0.2
                if score > bw:
                    best, bw = r, score
            it = self.by_ref[best]
            remaining.discard(best)
            w = self.weight_to(best, placed)
            if best in DEFAULT_ANCHOR and DEFAULT_ANCHOR[best] in placed:
                w[DEFAULT_ANCHOR[best]] = w.get(DEFAULT_ANCHOR[best], 0.0) + 0.3
            if w:
                tot = sum(w.values())
                px = sum(placed[o].center_at(*placed[o].new)[0] * v for o, v in w.items()) / tot
                py = sum(placed[o].center_at(*placed[o].new)[1] * v for o, v in w.items()) / tot
            else:
                c = it.center_at(it.x, it.y, it.rot)
                px, py = cx + (c[0] - cx) * 0.4, cy + (c[1] - cy) * 0.4
            rots = [it.rot % 360, (it.rot + 90) % 360]
            self.search_place(it, (px, py), rots, grid, placed, put)

    def search_place(self, it, want, rots, grid, placed, put):
        offs = self._offsets()
        for (ox, oy) in offs:
            cx_, cy_ = want[0] + ox, want[1] + oy
            for rot in rots:
                ax, ay = it.anchor_for_center(cx_, cy_, rot)
                poly = world_poly(it.box, ax, ay, rot)
                if it.ref in EDGE_ALLOW:
                    side_, mmv = EDGE_ALLOW[it.ref]
                    bb = list(aabb(poly))
                    if side_ == 'B': bb[3] -= mmv
                    elif side_ == 'T': bb[2] += mmv
                    elif side_ == 'R': bb[1] -= mmv
                    elif side_ == 'L': bb[0] += mmv
                    poly = rect_corners(tuple(bb))
                if not self.body_ok(poly, BODY_EDGE_MARGIN):
                    continue
                ok = True
                for (b, sd) in it.obstacles(ax, ay, rot):
                    if grid.hit(b, sd, BODY_GAP):
                        ok = False
                        break
                if ok:
                    put(it, ax, ay, rot)
                    return True
        self.warn.append('KHONG tim duoc cho cho %s (%s)' % (it.ref, it.value))
        it.new = (it.x, it.y, it.rot)
        return False

    _off_cache = None

    def _offsets(self):
        if Planner._off_cache is None:
            n = int(SEARCH_RADIUS / GRID)
            lst = []
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    d = math.hypot(i, j) * GRID
                    if d <= SEARCH_RADIUS:
                        lst.append((d, i * GRID, j * GRID))
            lst.sort()
            Planner._off_cache = [(x, y) for (_, x, y) in lst]
        return Planner._off_cache

    def run(self):
        self.classify()
        self.do_esc()
        self.do_body()
        return self.warn


# ----------------------------------------------------------------------------------
# LOP NOI PCBNEW
# ----------------------------------------------------------------------------------
def _mm(v):
    return pcbnew.ToMM(v)


def read_outline(board):
    """Doc Edge.Cuts (chi doan thang) va noi thanh da giac."""
    segs = []
    try:
        for d in board.GetDrawings():
            if d.GetLayer() == pcbnew.Edge_Cuts and d.GetShape() == pcbnew.SHAPE_T_SEGMENT:
                a, b = d.GetStart(), d.GetEnd()
                segs.append(((round(_mm(a.x), 3), round(_mm(a.y), 3)), (round(_mm(b.x), 3), round(_mm(b.y), 3))))
    except Exception:
        return None
    if len(segs) < 8:
        return None

    def key(p):
        return (round(p[0], 1), round(p[1], 1))
    adj = {}
    for a, b in segs:
        adj.setdefault(key(a), []).append((a, b))
        adj.setdefault(key(b), []).append((b, a))
    start = segs[0][0]
    poly = [start]
    cur = start
    prev_key = None
    for _ in range(len(segs) + 2):
        nxt = None
        for (p, q) in adj.get(key(cur), []):
            if key(q) != prev_key and key(q) != key(cur):
                nxt = q
                break
        if nxt is None:
            return None
        prev_key = key(cur)
        cur = nxt
        if key(cur) == key(start):
            break
        poly.append(cur)
    return poly if len(poly) >= 8 else None


def _pads(fp):
    try:
        return list(fp.Pads())
    except Exception:
        return list(fp.GetPads())


def load_items(board):
    items = []
    for fp in board.GetFootprints():
        ref = fp.GetReference()
        pos = fp.GetPosition()
        x, y = _mm(pos.x), _mm(pos.y)
        rot = fp.GetOrientationDegrees()
        side = 'B' if fp.GetLayer() == pcbnew.B_Cu else 'F'
        pad_boxes, nets, tht = [], set(), False
        for pad in _pads(fp):
            pc = pad.GetPosition()
            lx, ly = unrot_xy(_mm(pc.x) - x, _mm(pc.y) - y, rot)
            sz = pad.GetSize()
            w, h = _mm(sz.x), _mm(sz.y)
            rel = (pad.GetOrientationDegrees() - rot) % 180.0
            if abs(rel - 90.0) < 1.0:
                w, h = h, w
            elif rel > 1.0 and rel < 179.0:
                w = h = max(w, h)
            pad_boxes.append((lx - w / 2, lx + w / 2, ly - h / 2, ly + h / 2))
            nn = pad.GetNetname()
            if nn:
                nets.add(nn)
            try:
                if pad.GetAttribute() in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH):
                    tht = True
            except Exception:
                pass
        box = None
        try:      # courtyard -> hop bao LOCAL chinh xac (ke ca khi linh kien xoay 45 do)
            cy_layer = pcbnew.B_CrtYd if side == 'B' else pcbnew.F_CrtYd
            cyd = fp.GetCourtyard(cy_layer)
            xs, ys = [], []
            for oi in range(cyd.OutlineCount()):
                ol = cyd.Outline(oi)
                for k in range(ol.PointCount()):
                    pt = ol.CPoint(k)
                    px, py = unrot_xy(_mm(pt.x) - x, _mm(pt.y) - y, rot)
                    xs.append(px)
                    ys.append(py)
            if xs and (max(xs) - min(xs)) < 30 and (max(ys) - min(ys)) < 30:
                box = (min(xs), max(xs), min(ys), max(ys))
        except Exception:
            box = None
        if pad_boxes:
            pb = (min(b[0] for b in pad_boxes), max(b[1] for b in pad_boxes),
                  min(b[2] for b in pad_boxes), max(b[3] for b in pad_boxes))
            pb = grow(pb, 0.25)
            if box is None:
                box = pb
            else:
                box = (min(box[0], pb[0]), max(box[1], pb[1]), min(box[2], pb[2]), max(box[3], pb[3]))
        if box is None:
            box = (-1.0, 1.0, -1.0, 1.0)
        if ref in BOX_OVERRIDE:
            box = BOX_OVERRIDE[ref]
        it = Item(ref, side, x, y, rot, box, pad_boxes, tht, nets, fp.GetValue())
        it.fp = fp
        items.append(it)
    return items


def apply(board, items):
    n = 0
    for it in items:
        if not it.new or getattr(it, 'fp', None) is None:
            continue
        ax, ay, rot = it.new
        fp = it.fp
        if abs(rot - fp.GetOrientationDegrees()) > 1e-6:
            fp.SetOrientationDegrees(rot)
        fp.SetPosition(pcbnew.VECTOR2I(int(round(pcbnew.FromMM(ax))), int(round(pcbnew.FromMM(ay)))))
        n += 1
    return n


def main():
    board = pcbnew.GetBoard()
    fname = board.GetFileName()
    if fname and os.path.exists(fname):
        bk = fname.replace('.kicad_pcb', '_backup_%s.kicad_pcb' % time.strftime('%Y%m%d_%H%M%S'))
        shutil.copy2(fname, bk)
        print('Da backup file goc ->', bk)
    outline = read_outline(board) or FALLBACK_OUTLINE
    octagon = reflex_vertices(outline)
    if len(octagon) != 8:
        octagon = FALLBACK_OCTAGON
    xs = [p[0] for p in octagon]
    size = max(xs) - min(xs)
    print('Than board (octagon): %.1f mm' % size)
    items = load_items(board)
    pl = Planner(items, outline, octagon)
    warns = pl.run()
    n = apply(board, items)
    pcbnew.Refresh()
    print('Hoan tat. Da dat lai %d linh kien.' % n)
    for w in warns:
        print('CANH BAO:', w)
    print('Kiem tra bang DRC (courtyard) roi di lai track. Nho luu file.')


if pcbnew is not None:
    main()
