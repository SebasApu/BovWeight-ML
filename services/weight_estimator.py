import numpy as np
import logging

logger = logging.getLogger(__name__)

# Fórmula barrimétrica calibrada para ganado tropical de Costa Rica
# Peso = (PT² × L) / K
BREED_COEFFICIENTS = {
    'brahman': {
        'formula_divisor': 11000,
        'crevat_coef': 80,
    },
    'cebu': {
        'formula_divisor': 10700,
        'crevat_coef': 78,
    },
    'criollo': {
        'formula_divisor': 11500,
        'crevat_coef': 76,
    },
    'default': {
        'formula_divisor': 11000,
        'crevat_coef': 80,
    }
}


class WeightEstimator:

    def estimate(self, measurements, breed='default'):
        pt = measurements.get('chest_girth_cm', 0)
        l = measurements.get('body_length_cm', 0)
        h = measurements.get('height_cm', 0)
        has_ref = measurements.get('has_reference', False)

        if not has_ref or pt <= 0 or l <= 0:
            return self._no_reference_response(measurements)

        coefs = BREED_COEFFICIENTS.get(breed, BREED_COEFFICIENTS['default'])

        # Método 1: Fórmula barrimétrica clásica
        weight_barimetric = (pt ** 2 * l) / coefs['formula_divisor']

        # Método 2: Fórmula Crevat-Quetelet
        weight_crevat = coefs['crevat_coef'] * (pt / 100) ** 3

        # Combinar
        estimates = []
        if 80 < weight_barimetric < 1500:
            estimates.append(weight_barimetric)
        if 80 < weight_crevat < 1500:
            estimates.append(weight_crevat)

        if not estimates:
            return self._no_reference_response(measurements)

        if len(estimates) == 2:
            # Barrimétrica tiene más peso porque usa largo + perímetro
            final_weight = estimates[0] * 0.60 + estimates[1] * 0.40
        else:
            final_weight = estimates[0]

        # Confianza y rango
        confidence = self._calculate_confidence(estimates, measurements)
        error_margin = 0.10  # ±10% con referencia
        range_min = final_weight * (1 - error_margin)
        range_max = final_weight * (1 + error_margin)

        result = {
            'weight_kg': round(final_weight, 1),
            'range_min': round(range_min, 1),
            'range_max': round(range_max, 1),
            'confidence': round(confidence, 2),
            'method': 'barimetric_with_reference',
            'detail': {
                'barimetric': round(weight_barimetric, 1),
                'crevat': round(weight_crevat, 1),
            }
        }

        logger.info(f"Peso estimado: {final_weight:.1f} kg "
                     f"(bar={weight_barimetric:.1f}, crev={weight_crevat:.1f}), "
                     f"confianza: {confidence:.0%}")

        return result

    def _calculate_confidence(self, weights, measurements):
        confidence = 0.65  # Base con referencia

        pt = measurements.get('chest_girth_cm', 0)
        l = measurements.get('body_length_cm', 0)

        # Medidas en rangos realistas
        if 120 < pt < 280 and 100 < l < 250:
            confidence += 0.15

        # Poco desacuerdo entre métodos
        if len(weights) >= 2:
            cv = np.std(weights) / np.mean(weights)
            if cv < 0.15:
                confidence += 0.10
            elif cv < 0.25:
                confidence += 0.05

        return min(confidence, 0.95)

    def _no_reference_response(self, measurements):
        """
        Cuando no hay referencia, NO inventar un peso.
        Informar al usuario que necesita el objeto de referencia.
        """
        return {
            'weight_kg': 0,
            'range_min': 0,
            'range_max': 0,
            'confidence': 0,
            'method': 'no_reference',
            'detail': {
                'message': 'No se detectó objeto de referencia. '
                           'Coloque un palo de 1 metro al lado del animal '
                           'y vuelva a tomar la foto.',
                'animal_detected': True,
            }
        }
