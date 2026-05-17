from flask import Flask, request, jsonify
from flask_cors import CORS
from services.detector import CattleDetector
from services.measurement import BodyMeasurer
from services.weight_estimator import WeightEstimator
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = CattleDetector()
measurer = BodyMeasurer()
estimator = WeightEstimator()


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': detector.is_loaded()})


@app.route('/api/estimate', methods=['POST'])
def estimate_weight():
    """
    Estimar peso de ganado bovino a partir de una foto.

    Requiere:
    - image: foto del animal (vista lateral)
    - reference_length_cm: largo real del objeto de referencia (default: 100 cm)
    - breed: raza del animal (brahman, cebu, criollo, default)

    El objeto de referencia (palo/tabla de tamaño conocido) debe estar
    visible en la foto al lado del animal para poder calcular medidas reales.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No se envio imagen', 'code': 'NO_IMAGE'}), 400

    image_file = request.files['image']
    reference_cm = float(request.form.get('reference_length_cm', 100.0))
    breed = request.form.get('breed', 'default')

    try:
        image_bytes = image_file.read()

        # Paso 1: Detectar ganado
        detection = detector.detect(image_bytes)

        if detection is None:
            return jsonify({
                'error': 'No se detecto ganado bovino en la imagen',
                'code': 'NO_CATTLE',
                'sugerencia': 'Asegurese de que el animal este completamente visible, '
                              'de perfil (vista lateral) y bien iluminado'
            }), 422

        # Paso 2: Detectar objeto de referencia (fuera del área del animal)
        reference = detector.detect_reference(
            image_bytes,
            cattle_bbox=detection['bbox'],
            cattle_mask=detection['mask']
        )

        # Paso 3: Medir
        measurements = measurer.measure(
            mask=detection['mask'],
            contour=detection['contour'],
            bbox=detection['bbox'],
            image_shape=detection['image_shape'],
            reference=reference,
            reference_cm=reference_cm,
            breed=breed
        )

        # Paso 4: Estimar peso
        result = estimator.estimate(measurements, breed=breed)

        # Respuesta diferente según si hay referencia o no
        response = {
            'peso_estimado_kg': result['weight_kg'],
            'rango_min_kg': result['range_min'],
            'rango_max_kg': result['range_max'],
            'confianza': result['confidence'],
            'metodo': result['method'],
            'medidas': {
                'perimetro_toracico_cm': measurements.get('chest_girth_cm'),
                'largo_cuerpo_cm': measurements.get('body_length_cm'),
                'altura_cm': measurements.get('height_cm'),
            },
            'referencia_detectada': reference is not None,
            'deteccion': {
                'bbox': detection['bbox'],
                'score': detection['score'],
            },
        }

        if result['method'] == 'no_reference':
            response['advertencia'] = (
                'No se detecto objeto de referencia en la imagen. '
                'Para obtener una estimacion precisa, coloque un palo o tabla '
                f'de {reference_cm:.0f} cm verticalmente al lado del animal '
                'y vuelva a tomar la foto.'
            )
            response['requiere_referencia'] = True
        else:
            response['advertencia'] = (
                'Estimacion aproximada (margen +-10%). '
                'No sustituye bascula oficial para transacciones comerciales.'
            )
            response['requiere_referencia'] = False

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error en estimacion: {e}", exc_info=True)
        return jsonify({'error': 'Error interno en el procesamiento', 'code': 'INTERNAL_ERROR'}), 500


@app.route('/api/estimate/batch', methods=['POST'])
def estimate_batch():
    """Estimar con múltiples fotos del mismo animal (promedia resultados)."""
    images = request.files.getlist('images')
    reference_cm = float(request.form.get('reference_length_cm', 100.0))
    breed = request.form.get('breed', 'default')

    if not images:
        return jsonify({'error': 'No se enviaron imagenes'}), 400

    results = []
    for img in images:
        image_bytes = img.read()
        detection = detector.detect(image_bytes)
        if detection:
            reference = detector.detect_reference(image_bytes, cattle_bbox=detection['bbox'], cattle_mask=detection['mask'])
            measurements = measurer.measure(
                mask=detection['mask'],
                contour=detection['contour'],
                bbox=detection['bbox'],
                image_shape=detection['image_shape'],
                reference=reference,
                reference_cm=reference_cm,
                breed=breed
            )
            result = estimator.estimate(measurements, breed=breed)
            if result['weight_kg'] > 0:
                results.append(result['weight_kg'])

    if not results:
        return jsonify({
            'error': 'No se pudo estimar peso en ninguna imagen',
            'sugerencia': 'Asegurese de incluir el objeto de referencia en cada foto'
        }), 422

    avg_weight = sum(results) / len(results)
    return jsonify({
        'peso_estimado_kg': round(avg_weight, 1),
        'pesos_individuales': results,
        'num_imagenes_procesadas': len(results),
        'advertencia': 'Estimacion aproximada. No sustituye bascula oficial.'
    })


@app.route('/api/debug', methods=['POST'])
def debug_detection():
    """
    Endpoint de debug: devuelve la imagen con anotaciones dibujadas.
    Muestra: bbox del animal, mascara, referencia detectada, medidas.
    """
    import cv2
    import numpy as np
    from flask import send_file
    import io

    if 'image' not in request.files:
        return jsonify({'error': 'No se envio imagen'}), 400

    image_file = request.files['image']
    breed = request.form.get('breed', 'default')
    image_bytes = image_file.read()

    # Decodificar imagen
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    debug_img = img.copy()
    h, w = img.shape[:2]

    # Detectar animal
    detection = detector.detect(image_bytes)
    if detection is None:
        cv2.putText(debug_img, "NO SE DETECTO ANIMAL", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        _, buffer = cv2.imencode('.jpg', debug_img)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')

    # Dibujar bbox del animal (verde)
    bx1, by1, bx2, by2 = [int(b) for b in detection['bbox']]
    cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
    cv2.putText(debug_img, f"Animal ({detection['score']:.0%})", (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Dibujar mascara semi-transparente (azul)
    mask_overlay = debug_img.copy()
    mask_overlay[detection['mask'] > 0] = [255, 150, 0]
    debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)

    # Detectar referencia
    reference = detector.detect_reference(image_bytes,
                                           cattle_bbox=detection['bbox'],
                                           cattle_mask=detection['mask'])

    if reference:
        pts = reference['points']
        # Dibujar referencia (rojo, grueso)
        cv2.line(debug_img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 0, 255), 4)
        cv2.putText(debug_img, f"REF: {reference['length_px']:.0f}px",
                    (pts[0] - 80, pts[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calcular medidas
        reference_cm = float(request.form.get('reference_length_cm', 100.0))
        measurements = measurer.measure(
            mask=detection['mask'],
            contour=detection['contour'],
            bbox=detection['bbox'],
            image_shape=detection['image_shape'],
            reference=reference,
            reference_cm=reference_cm,
            breed=breed
        )
        result = estimator.estimate(measurements, breed=breed)

        # Dibujar info
        info_y = 40
        infos = [
            f"Peso: {result['weight_kg']:.1f} kg",
            f"PT: {measurements['chest_girth_cm']:.1f} cm",
            f"Largo: {measurements['body_length_cm']:.1f} cm",
            f"Altura: {measurements['height_cm']:.1f} cm",
            f"px/cm: {measurements.get('px_per_cm', 0):.2f}",
            f"Ref: {reference['length_px']:.0f} px = {reference_cm} cm",
        ]
        for info in infos:
            cv2.putText(debug_img, info, (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
    else:
        cv2.putText(debug_img, "NO SE DETECTO REFERENCIA", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', debug_img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
