import numpy as np
import skfuzzy as fuzz
import sys

def main(argv):
    print("Algoritmo de lógica difusa")
    print("Sistema de propinas para restaurantes basados en atributos de calidad de comida")

    if len(argv) < 3:
        print("Longitud del ARGV muy corta. Se esperan dos argumentos.")
        exit(1)
    else:
        cal_comida = float(argv[1])  # Convertir a float ya que argv son cadenas
        cal_servicio = float(argv[2])

        x_qual = np.arange(0, 11, 1)
        x_serv = np.arange(0, 11, 1)
        x_tip = np.arange(5, 26, 1)

        qual_lo = fuzz.trimf(x_qual, [0, 0, 5])  # 0-1 5-0
        qual_md = fuzz.trimf(x_qual, [0, 5, 10])  # 0-0 5-1 10-0
        qual_hi = fuzz.trimf(x_qual, [5, 10, 10])  # 5-0 10-1

        serv_lo = fuzz.trimf(x_serv, [0, 0, 5])  # 0-1 5-0
        serv_md = fuzz.trimf(x_serv, [0, 5, 10])  # 0-0 5-1 10-0
        serv_hi = fuzz.trimf(x_serv, [5, 10, 10])  # 5-0 10-1

        tip_lo = fuzz.trimf(x_tip, [5, 5, 13])  # 5-1 13-0
        tip_md = fuzz.trimf(x_tip, [5, 13, 25])  # 5-0 13-1 25-0
        tip_hi = fuzz.trimf(x_tip, [13, 25, 25])  # 13-0 25-1

        qual_level_lo = fuzz.interp_membership(x_qual, qual_lo, cal_comida)
        qual_level_md = fuzz.interp_membership(x_qual, qual_md, cal_comida)
        qual_level_hi = fuzz.interp_membership(x_qual, qual_hi, cal_comida)

        serv_level_lo = fuzz.interp_membership(x_serv, serv_lo, cal_servicio)
        serv_level_md = fuzz.interp_membership(x_serv, serv_md, cal_servicio)
        serv_level_hi = fuzz.interp_membership(x_serv, serv_hi, cal_servicio)

        active_rule1 = np.fmax(qual_level_lo, serv_level_lo)
        tip_activation_lo = np.fmin(active_rule1, tip_lo)

        active_rule2 = serv_level_md
        tip_activation_md = np.fmin(active_rule2, tip_md)

        active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
        tip_activation_hi = np.fmin(active_rule3, tip_hi)

        aggregated = np.fmax(tip_activation_lo, np.fmax(tip_activation_md, tip_activation_hi))

        tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
        print(tip)

if __name__ == "__main__":
    main(sys.argv)
