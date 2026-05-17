import numpy as np
import cv2
from ultralytics import YOLO
import logging
import os

logger = logging.getLogger(__name__)

CATTLE_CLASSES = {'cow'}


class CattleDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.environ.get('MODEL_PATH', 'models/yolov8m-seg.pt')

        try:
            self.model = YOLO(model_path)
            self._loaded = True
            logger.info(f"Modelo cargado: {model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self._loaded = False
            self.model = YOLO('yolov8m-seg.pt')
            self._loaded = True
            logger.info("Modelo por defecto descargado y cargado")

    def is_loaded(self):
        return self._loaded

    def _bytes_to_cv2(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def detect(self, image_bytes):
        """Detectar ganado bovino con segmentacion."""
        img = self._bytes_to_cv2(image_bytes)
        if img is None:
            logger.error("No se pudo decodificar la imagen")
            return None

        h, w = img.shape[:2]
        results = self.model(img, conf=0.4, verbose=False)

        best_detection = None
        best_score = 0

        for result in results:
            if result.masks is None:
                continue

            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                if class_name in CATTLE_CLASSES and confidence > best_score:
                    mask = result.masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    main_contour = max(contours, key=cv2.contourArea)
                    bbox = box.xyxy[0].tolist()

                    best_detection = {
                        'mask': mask,
                        'contour': main_contour,
                        'bbox': [round(b, 1) for b in bbox],
                        'score': round(confidence, 3),
                        'image_shape': (h, w),
                    }
                    best_score = confidence

        if best_detection:
            logger.info(f"Ganado detectado con confianza {best_score:.2f}")
        else:
            logger.info("No se detecto ganado en la imagen")

        return best_detection

    def detect_reference(self, image_bytes, cattle_bbox=None, cattle_mask=None):
        """
        Detectar poste/palo de referencia combinando:
        1. Deteccion de contornos rectangulares verticales
        2. Hough Lines agrupadas como respaldo
        
        Usa la mascara de segmentacion del animal (no el bbox) para
        excluir solo la silueta del animal, no el area completa.
        Asi se detectan postes que estan junto al animal.
        """
        img = self._bytes_to_cv2(image_bytes)
        if img is None:
            return None

        h, w = img.shape[:2]

        if cattle_bbox:
            bx1, by1, bx2, by2 = [int(b) for b in cattle_bbox]
            animal_height_px = by2 - by1
            animal_center_x = (bx1 + bx2) / 2
            animal_bottom_y = by2
        else:
            animal_height_px = h * 0.5
            animal_center_x = w / 2
            animal_bottom_y = h * 0.8

        # Mascara de exclusion: usar la silueta del animal, no el bbox completo
        exclude_mask = np.ones((h, w), dtype=np.uint8) * 255
        if cattle_mask is not None:
            # Dilatar la mascara un poco para dar margen
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate(cattle_mask, kernel, iterations=1)
            exclude_mask[dilated_mask > 0] = 0
        elif cattle_bbox:
            # Fallback: usar bbox si no hay mascara
            margin = int(max(bx2 - bx1, by2 - by1) * 0.03)
            cx1 = max(0, bx1 - margin)
            cy1 = max(0, by1 - margin)
            cx2 = min(w, bx2 + margin)
            cy2 = min(h, by2 + margin)
            exclude_mask[cy1:cy2, cx1:cx2] = 0

        candidates = []

        # Metodo 1: Contornos rectangulares
        c1 = self._find_pole_contours(img, exclude_mask, h, w,
                                       animal_height_px, animal_center_x, animal_bottom_y)
        candidates.extend(c1)

        # Metodo 2: Hough Lines agrupadas
        c2 = self._find_pole_lines(img, exclude_mask, h, w,
                                    animal_height_px, animal_center_x, animal_bottom_y)
        candidates.extend(c2)

        if not candidates:
            logger.info("No se encontro referencia valida")
            return None

        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]

        logger.info(f"Referencia [{best['method']}]: largo={best['length_px']:.0f}px, "
                     f"score={best['score']:.3f}, size_ratio={best.get('size_ratio', 0):.2f}")

        return {
            'type': 'line',
            'points': best.get('points', [0, 0, 0, 0]),
            'length_px': best['length_px'],
        }

    def _find_pole_contours(self, img, exclude_mask, h, w,
                             animal_height_px, animal_center_x, animal_bottom_y):
        """Buscar poste como contorno rectangular vertical delgado."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        candidates = []

        for method in ['adaptive', 'edges', 'otsu']:
            if method == 'adaptive':
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 15, 2)
            elif method == 'otsu':
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                binary = cv2.Canny(gray, 30, 100)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary = cv2.dilate(binary, kernel, iterations=2)

            binary = cv2.bitwise_and(binary, exclude_mask)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect

                if rw > rh:
                    rw, rh = rh, rw

                if rw < 1:
                    continue

                # Aspect ratio: poste es alto y delgado (ratio > 2.5)
                aspect_ratio = rh / rw
                if aspect_ratio < 2.5 or aspect_ratio > 40:
                    continue

                # Tamano vs animal
                size_ratio = rh / animal_height_px
                if size_ratio < 0.20 or size_ratio > 1.3:
                    continue

                # Verticalidad
                norm_angle = abs(angle) % 180
                if norm_angle > 90:
                    norm_angle = 180 - norm_angle
                if norm_angle > 35:
                    continue

                # Debe llegar cerca del suelo
                box_points = cv2.boxPoints(rect)
                max_y = max(box_points[:, 1])
                min_y = min(box_points[:, 1])
                if max_y < h * 0.4:
                    continue

                # Ancho razonable
                if rw < 3 or rw > w * 0.08:
                    continue

                pole_height = max_y - min_y

                score = self._score_candidate(
                    cx, pole_height, size_ratio, norm_angle, max_y,
                    w, animal_center_x, animal_bottom_y
                )

                candidates.append({
                    'length_px': float(pole_height),
                    'score': float(score),
                    'size_ratio': float(size_ratio),
                    'method': f'contour_{method}',
                    'points': [int(cx), int(min_y), int(cx), int(max_y)],
                })

        return candidates

    def _find_pole_lines(self, img, exclude_mask, h, w,
                          animal_height_px, animal_center_x, animal_bottom_y):
        """Buscar poste agrupando lineas verticales cercanas."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, exclude_mask)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                 threshold=50, minLineLength=int(h * 0.05), maxLineGap=25)
        if lines is None:
            return []

        # Recoger segmentos verticales
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy < 1:
                continue
            angle = np.degrees(np.arctan2(dx, dy))
            if angle < 20:
                segments.append({
                    'x': (x1 + x2) / 2,
                    'y_min': min(y1, y2),
                    'y_max': max(y1, y2),
                })

        if not segments:
            return []

        # Agrupar segmentos por posicion X (±15px = mismo poste)
        segments.sort(key=lambda s: s['x'])
        groups = []
        current = [segments[0]]

        for seg in segments[1:]:
            if abs(seg['x'] - current[-1]['x']) < 15:
                current.append(seg)
            else:
                groups.append(current)
                current = [seg]
        groups.append(current)

        candidates = []
        for group in groups:
            total_y_min = min(s['y_min'] for s in group)
            total_y_max = max(s['y_max'] for s in group)
            total_length = total_y_max - total_y_min
            avg_x = np.mean([s['x'] for s in group])

            size_ratio = total_length / animal_height_px
            if size_ratio < 0.25 or size_ratio > 1.2:
                continue
            if total_length < h * 0.05 or total_length > h * 0.75:
                continue
            if total_y_max < h * 0.4:
                continue

            score = self._score_candidate(
                avg_x, total_length, size_ratio, 0, total_y_max,
                w, animal_center_x, animal_bottom_y
            ) * 0.85  # Penalizar ligeramente vs contornos

            candidates.append({
                'length_px': float(total_length),
                'score': float(score),
                'size_ratio': float(size_ratio),
                'method': 'hough_grouped',
                'points': [int(avg_x), int(total_y_min), int(avg_x), int(total_y_max)],
            })

        return candidates

    def _score_candidate(self, center_x, length, size_ratio, angle,
                          bottom_y, img_w, animal_center_x, animal_bottom_y):
        """Puntaje unificado para candidatos a referencia."""
        # Cercania al animal (relajado - el poste puede estar en los bordes)
        dist = abs(center_x - animal_center_x)
        closeness = 1.0 / (1.0 + (dist / img_w) * 2)

        # Tamano ideal: 45-85% de la altura del animal (1m palo vs 1.2-1.6m vaca)
        size_ideal = max(0, 1.0 - abs(size_ratio - 0.65) * 1.8)

        # Verticalidad
        vert = 1.0 - (angle / 30) if angle < 30 else 0

        # Base alineada con el suelo del animal (relajado)
        bottom_diff = abs(bottom_y - animal_bottom_y) / max(animal_bottom_y, 1)
        bottom_match = max(0, 1.0 - bottom_diff * 2)

        score = (closeness * 0.20 +
                 size_ideal * 0.35 +
                 vert * 0.15 +
                 bottom_match * 0.30)

        return score
