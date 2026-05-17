import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class BodyMeasurer:

    def measure(self, mask, contour, bbox, image_shape,
                reference=None, reference_cm=100.0, breed='default'):
        h_img, w_img = image_shape

        # Medir en píxeles
        body_length_px = self._measure_body_length(contour, bbox)
        chest_width_px = self._measure_chest_width(mask, contour, bbox)
        height_px = self._measure_height(contour, bbox)

        has_reference = reference is not None
        px_per_cm = None

        if has_reference:
            px_per_cm = self._calculate_scale(reference, reference_cm)

        if has_reference and px_per_cm and px_per_cm > 0:
            # === CON REFERENCIA: medidas reales ===
            body_length_cm = body_length_px / px_per_cm
            height_cm = height_px / px_per_cm

            # El ancho visible del pecho en vista lateral ≈ diámetro dorso-ventral
            # El perímetro torácico real ≈ diámetro × π × factor_elipse
            # Para ganado bovino, la sección es elíptica: ~0.85 del círculo perfecto
            chest_visible_cm = chest_width_px / px_per_cm
            chest_girth_cm = chest_visible_cm * np.pi * 0.85

            logger.info(f"Medidas CON referencia: px/cm={px_per_cm:.2f}, "
                         f"pecho_visible={chest_visible_cm:.1f}cm → PT={chest_girth_cm:.1f}cm, "
                         f"L={body_length_cm:.1f}cm, H={height_cm:.1f}cm")
        else:
            # === SIN REFERENCIA: no se puede estimar con precisión ===
            has_reference = False
            body_length_cm = 0
            chest_girth_cm = 0
            height_cm = 0
            logger.info("Sin referencia detectada, no se pueden calcular medidas reales")

        measurements = {
            'body_length_px': round(body_length_px, 1),
            'chest_girth_px': round(chest_width_px, 1),
            'height_px': round(height_px, 1),
            'body_length_cm': round(body_length_cm, 1),
            'chest_girth_cm': round(chest_girth_cm, 1),
            'height_cm': round(height_cm, 1),
            'has_reference': has_reference,
            'px_per_cm': round(px_per_cm, 3) if px_per_cm else None,
            'bbox_area_ratio': self._bbox_area_ratio(bbox, image_shape),
        }

        return measurements

    def _calculate_scale(self, reference, reference_cm):
        """Calcular píxeles por centímetro usando el objeto de referencia."""
        if reference is None:
            return None

        if reference['type'] == 'line':
            px_length = reference['length_px']
        else:
            px_length = max(reference.get('width_px', 0), reference.get('height_px', 0))

        if px_length < 10:
            return None

        return px_length / reference_cm

    def _measure_body_length(self, contour, bbox):
        """Medir largo del cuerpo usando eje mayor de elipse ajustada."""
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            return major_axis

        x1, y1, x2, y2 = bbox
        return x2 - x1

    def _measure_chest_width(self, mask, contour, bbox):
        """
        Medir ancho visible del pecho (vista lateral).
        Toma cortes verticales en la zona del barril (25%-40% del cuerpo)
        y usa el más ancho.
        """
        x1, y1, x2, y2 = [int(b) for b in bbox]
        body_width = x2 - x1

        max_width = 0
        for pct in [0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]:
            chest_x = int(x1 + body_width * pct)
            if 0 <= chest_x < mask.shape[1]:
                column = mask[:, chest_x]
                pixels = np.where(column > 0)[0]
                if len(pixels) > 0:
                    width = pixels[-1] - pixels[0]
                    max_width = max(max_width, width)

        return max_width if max_width > 0 else (y2 - y1)

    def _measure_height(self, contour, bbox):
        """Medir altura del animal."""
        x1, y1, x2, y2 = bbox
        return y2 - y1

    def _bbox_area_ratio(self, bbox, image_shape):
        x1, y1, x2, y2 = bbox
        h, w = image_shape
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = h * w
        return round(bbox_area / img_area, 4)
