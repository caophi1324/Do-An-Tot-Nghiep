"""Video viewport widget with aspect ratio preservation, side-margin live face tracking tags, and standby screen."""
from __future__ import annotations

import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QPainterPath
from PySide6.QtWidgets import QWidget


class VideoWidget(QWidget):
    """Renders processed video frames with aspect ratio preservation, click interaction,
    and live recognition tags (face avatars & IDs) utilizing the viewport pillarbox margins.
    """

    frame_clicked = Signal(float, float)  # Emits (frame_x, frame_y) in original frame coordinates

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background-color: #0d0e14;")
        self.setCursor(Qt.CrossCursor)

        self._pixmap: Optional[QPixmap] = None
        self._latest_results: List[Any] = []
        self._face_thumbnails: Dict[int, Tuple[QPixmap, float]] = {}
        self._standby_text = "No Video Source Active\nClick [Start Pipeline] to begin"

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        results: Optional[List[Any]] = None,
        raw_frame: Optional[np.ndarray] = None,
    ) -> None:
        """Receives a new BGR frame, extracts/updates face thumbnails for active tracks, and triggers repaint."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self._pixmap = QPixmap.fromImage(q_img)

        now = time.time()
        self._latest_results = results or []

        # Extract & cache live face crop avatars for tracked persons
        for res in self._latest_results:
            fb = getattr(res, "face_bbox", None)
            tid = getattr(res, "track_id", None)
            if fb is not None and tid is not None and raw_frame is not None:
                try:
                    fx1, fy1, fx2, fy2 = map(int, fb[:4])
                    fw, fh = fx2 - fx1, fy2 - fy1
                    if fw > 12 and fh > 12:
                        h_raw, w_raw = raw_frame.shape[:2]
                        mx = int(0.20 * fw)
                        my = int(0.25 * fh)
                        cx1 = max(0, fx1 - mx)
                        cy1 = max(0, fy1 - my)
                        cx2 = min(w_raw, fx2 + mx)
                        cy2 = min(h_raw, fy2 + my)
                        crop = raw_frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            thumb = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
                            q_thumb = QImage(thumb.data, 96, 96, 3 * 96, QImage.Format_BGR888)
                            self._face_thumbnails[tid] = (QPixmap.fromImage(q_thumb.copy()), now)
                except Exception:
                    pass

        # Prune stale thumbnails older than 5.0 seconds
        stale_tids = [tid for tid, (_, t) in self._face_thumbnails.items() if (now - t) > 5.0]
        for tid in stale_tids:
            self._face_thumbnails.pop(tid, None)

        self.update()

    def clear_frame(self) -> None:
        self._pixmap = None
        self._latest_results.clear()
        self._face_thumbnails.clear()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d0e14"))

        if self._pixmap is not None and not self._pixmap.isNull():
            widget_w, widget_h = self.width(), self.height()
            pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
            if pix_w > 0 and pix_h > 0:
                scale = min(widget_w / float(pix_w), widget_h / float(pix_h))
                dest_w = int(pix_w * scale)
                dest_h = int(pix_h * scale)
                dest_x = (widget_w - dest_w) // 2
                dest_y = (widget_h - dest_h) // 2
                target_rect = QRect(dest_x, dest_y, dest_w, dest_h)

                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.drawPixmap(target_rect, self._pixmap)

                # Render Recognition Tags & Face Avatars in side margins or floating HUD
                self._draw_tracking_tags(painter, widget_w, widget_h, dest_x, dest_y, dest_w, dest_h)
        else:
            # Standby screen with subtle OBS grid
            pen = QPen(QColor("#1a1b24"))
            pen.setWidth(1)
            painter.setPen(pen)

            grid_size = 40
            for gx in range(0, self.width(), grid_size):
                painter.drawLine(gx, 0, gx, self.height())
            for gy in range(0, self.height(), grid_size):
                painter.drawLine(0, gy, self.width(), gy)

            # Standby text
            painter.setPen(QColor("#777b8f"))
            font = QFont("Segoe UI", 11, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                self._standby_text,
            )

    def _draw_tracking_tags(
        self,
        painter: QPainter,
        widget_w: int,
        widget_h: int,
        dest_x: int,
        dest_y: int,
        dest_w: int,
        dest_h: int,
    ) -> None:
        """Renders live identification tags with face avatars utilizing viewport margins or floating HUD."""
        if not self._latest_results:
            return

        # 1. Identify Primary Target (Priority: Locked Target > Known Person > First Track)
        locked = [r for r in self._latest_results if getattr(r, "is_locked", False)]
        if locked:
            primary_res = locked[0]
        else:
            known = [r for r in self._latest_results if not getattr(r, "is_unknown", True)]
            primary_res = known[0] if known else self._latest_results[0]

        left_margin_w = dest_x
        right_margin_x = dest_x + dest_w
        right_margin_w = widget_w - right_margin_x

        # Primary Target Card Dimensions
        card_w = min(170, max(130, left_margin_w - 14)) if left_margin_w >= 110 else 165
        card_h = 245

        # Position: Left Pillarbox margin if space permits, otherwise floating top-left inside video
        if left_margin_w >= 110:
            card_x = (left_margin_w - card_w) // 2
            card_y = max(16, dest_y + 12)
        else:
            card_x = dest_x + 12
            card_y = dest_y + 12

        self._draw_target_card(painter, primary_res, card_x, card_y, card_w, card_h, is_primary=True)

        # 2. Right Pillarbox Margin: Scene Tracks Roster (other detected persons in scene)
        if right_margin_w >= 110:
            secondary_tracks = [r for r in self._latest_results if r.track_id != primary_res.track_id]
            if secondary_tracks:
                sec_card_w = min(160, right_margin_w - 14)
                sec_x = right_margin_x + (right_margin_w - sec_card_w) // 2
                sec_y = max(16, dest_y + 12)

                # Header label
                painter.save()
                hdr_rect = QRect(sec_x, sec_y, sec_card_w, 20)
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.setPen(QColor("#8b949e"))
                painter.drawText(hdr_rect, Qt.AlignLeft | Qt.AlignVCenter, f"ĐANG THEO DÕI ({len(self._latest_results)})")
                painter.restore()

                sec_y += 24
                for r in secondary_tracks[:3]:  # Show up to 3 other tracks
                    self._draw_secondary_track_card(painter, r, sec_x, sec_y, sec_card_w, 68)
                    sec_y += 74

    def _draw_target_card(
        self,
        painter: QPainter,
        res: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        is_primary: bool = True,
    ) -> None:
        """Renders tactical primary target card with face avatar, ID, status badge, and quality meter."""
        painter.save()
        card_rect = QRect(x, y, w, h)

        is_locked = getattr(res, "is_locked", False)
        is_unknown = getattr(res, "is_unknown", True)
        person_name = getattr(res, "person_name", "Unknown")
        person_id = getattr(res, "person_id", None)
        track_id = getattr(res, "track_id", 0)
        similarity = getattr(res, "similarity", 0.0)
        quality = getattr(res, "quality", 0.0)
        is_complete = getattr(res, "is_learning_complete", False)
        learn_cnt = getattr(res, "learning_count", 1)
        learn_tgt = getattr(res, "learning_target", 5)

        # Border & Accent Colors
        if is_locked:
            accent_color = QColor(255, 193, 7)  # Gold
            bg_color = QColor(18, 22, 32, 235)
        elif not is_unknown:
            accent_color = QColor(46, 160, 67)  # Emerald Green
            bg_color = QColor(16, 24, 20, 235)
        else:
            accent_color = QColor(70, 110, 160)  # Slate Blue
            bg_color = QColor(16, 20, 28, 235)

        # 1. Background Card
        painter.setPen(QPen(accent_color, 1.5))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(card_rect, 8.0, 8.0)

        # 2. Header Pill
        pill_rect = QRect(x + 8, y + 8, w - 16, 20)
        if is_locked:
            pill_bg = QColor(255, 193, 7, 45)
            pill_fg = QColor(255, 215, 0)
            pill_text = "🎯 TARGET LOCKED"
        elif not is_unknown:
            pill_bg = QColor(46, 160, 67, 45)
            pill_fg = QColor(80, 220, 120)
            pill_text = "👁️ RECOGNIZED"
        else:
            pill_bg = QColor(70, 110, 160, 45)
            pill_fg = QColor(140, 190, 255)
            pill_text = "🔍 TRACKING"

        painter.setPen(Qt.NoPen)
        painter.setBrush(pill_bg)
        painter.drawRoundedRect(pill_rect, 4.0, 4.0)
        painter.setPen(pill_fg)
        painter.setFont(QFont("Segoe UI", 7.5, QFont.Weight.Bold))
        painter.drawText(pill_rect, Qt.AlignCenter, pill_text)

        # 3. Face Avatar / Thumbnail
        avatar_size = min(84, w - 24)
        avatar_x = x + (w - avatar_size) // 2
        avatar_y = y + 34
        avatar_rect = QRect(avatar_x, avatar_y, avatar_size, avatar_size)

        thumb_entry = self._face_thumbnails.get(track_id)
        if thumb_entry is not None and not thumb_entry[0].isNull():
            painter.save()
            clip_path = QPainterPath()
            clip_path.addRoundedRect(avatar_rect, 6.0, 6.0)
            painter.setClipPath(clip_path)
            painter.drawPixmap(avatar_rect, thumb_entry[0])
            painter.restore()

            painter.setPen(QPen(accent_color, 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(avatar_rect, 6.0, 6.0)
        else:
            painter.setPen(QPen(QColor("#2d333b"), 1.0))
            painter.setBrush(QColor("#161b22"))
            painter.drawRoundedRect(avatar_rect, 6.0, 6.0)
            painter.setPen(QColor("#8b949e"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(avatar_rect, Qt.AlignCenter, "Chưa rõ mặt\n(Searching)")

        # 4. Assigned ID & Person Name
        text_y = avatar_y + avatar_size + 8
        name_rect = QRect(x + 6, text_y, w - 12, 20)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        display_name = person_name if (not is_unknown or is_locked) else f"Track #{track_id}"
        painter.drawText(name_rect, Qt.AlignCenter, display_name)

        id_rect = QRect(x + 6, text_y + 18, w - 12, 16)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#8b949e"))
        id_str = f"ID: {person_id if person_id is not None else '—'} | Track #{track_id}"
        painter.drawText(id_rect, Qt.AlignCenter, id_str)

        # 5. Status / Learning / Similarity Sub-badge
        status_y = text_y + 36
        status_rect = QRect(x + 8, status_y, w - 16, 16)
        if is_locked:
            if is_complete:
                sub_badge = f"Ready ({learn_cnt}/{learn_tgt})"
                badge_fg = QColor(46, 160, 67)
            else:
                sub_badge = f"Adapting ({learn_cnt}/{learn_tgt})"
                badge_fg = QColor(255, 193, 7)
        elif not is_unknown:
            sub_badge = f"Match: {similarity * 100:.0f}%"
            badge_fg = QColor(46, 160, 67)
        else:
            sub_badge = "Chưa nhận diện"
            badge_fg = QColor("#8b949e")

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(badge_fg)
        painter.drawText(status_rect, Qt.AlignCenter, sub_badge)

        # 6. Mini Progress / Quality Meter Bar
        bar_y = status_y + 18
        bar_w = w - 24
        bar_rect = QRect(x + 12, bar_y, bar_w, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#21262d"))
        painter.drawRoundedRect(bar_rect, 2.5, 2.5)

        if is_locked:
            ratio = min(1.0, max(0.0, learn_cnt / float(max(1, learn_tgt))))
            meter_color = QColor(46, 160, 67) if is_complete else QColor(255, 193, 7)
        elif not is_unknown:
            ratio = min(1.0, max(0.0, similarity))
            meter_color = QColor(46, 160, 67)
        else:
            ratio = min(1.0, max(0.0, quality))
            meter_color = QColor(70, 110, 160)

        fill_w = int(bar_w * ratio)
        if fill_w > 0:
            painter.setBrush(meter_color)
            painter.drawRoundedRect(QRect(x + 12, bar_y, fill_w, 5), 2.5, 2.5)

        # 7. Quality Text Footer
        q_rect = QRect(x + 8, bar_y + 7, w - 16, 14)
        painter.setFont(QFont("Segoe UI", 7.5))
        painter.setPen(QColor("#8b949e"))
        painter.drawText(q_rect, Qt.AlignCenter, f"Quality: {quality * 100:.0f}%")

        painter.restore()

    def _draw_secondary_track_card(
        self,
        painter: QPainter,
        res: Any,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        """Renders compact track card for other persons visible in the scene."""
        painter.save()
        card_rect = QRect(x, y, w, h)
        is_unknown = getattr(res, "is_unknown", True)
        person_name = getattr(res, "person_name", "Unknown")
        track_id = getattr(res, "track_id", 0)
        similarity = getattr(res, "similarity", 0.0)

        border_col = QColor(46, 160, 67) if not is_unknown else QColor("#30363d")
        painter.setPen(QPen(border_col, 1))
        painter.setBrush(QColor(18, 22, 30, 220))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        # Thumbnail 48x48
        av_rect = QRect(x + 8, y + (h - 48) // 2, 48, 48)
        thumb_entry = self._face_thumbnails.get(track_id)
        if thumb_entry is not None and not thumb_entry[0].isNull():
            painter.save()
            clip_path = QPainterPath()
            clip_path.addRoundedRect(av_rect, 4.0, 4.0)
            painter.setClipPath(clip_path)
            painter.drawPixmap(av_rect, thumb_entry[0])
            painter.restore()
            painter.setPen(QPen(border_col, 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(av_rect, 4.0, 4.0)
        else:
            painter.setPen(QPen(QColor("#2d333b"), 1))
            painter.setBrush(QColor("#161b22"))
            painter.drawRoundedRect(av_rect, 4.0, 4.0)

        # Details
        tx_x = x + 62
        tx_w = w - 68
        painter.setFont(QFont("Segoe UI", 8.5, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff") if not is_unknown else QColor("#c9d1d9"))
        painter.drawText(QRect(tx_x, y + 10, tx_w, 16), Qt.AlignLeft, f"Track #{track_id}")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#8b949e"))
        painter.drawText(QRect(tx_x, y + 27, tx_w, 14), Qt.AlignLeft, person_name)

        if not is_unknown:
            painter.setPen(QColor(46, 160, 67))
            status_lbl = f"{similarity * 100:.0f}%"
        else:
            painter.setPen(QColor("#8b949e"))
            status_lbl = "Unknown"
        painter.drawText(QRect(tx_x, y + 43, tx_w, 14), Qt.AlignLeft, status_lbl)

        painter.restore()

    def mousePressEvent(self, event) -> None:
        """Translates viewport click to original frame coordinates and emits frame_clicked."""
        if event.button() == Qt.LeftButton:
            if self._pixmap is not None and not self._pixmap.isNull():
                widget_w, widget_h = self.width(), self.height()
                pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
                if pix_w > 0 and pix_h > 0:
                    scale = min(widget_w / float(pix_w), widget_h / float(pix_h))
                    dest_w = int(pix_w * scale)
                    dest_h = int(pix_h * scale)
                    offset_x = (widget_w - dest_w) // 2
                    offset_y = (widget_h - dest_h) // 2

                    pos = event.position()
                    click_x = pos.x()
                    click_y = pos.y()

                    if (
                        offset_x <= click_x < offset_x + dest_w
                        and offset_y <= click_y < offset_y + dest_h
                    ):
                        frame_x = (click_x - offset_x) / scale
                        frame_y = (click_y - offset_y) / scale
                        self.frame_clicked.emit(float(frame_x), float(frame_y))

        super().mousePressEvent(event)
