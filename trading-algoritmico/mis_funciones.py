def calcular_media_movil(precios, periodo):
    return sum(precios[-periodo:]) / periodo

def calcular_maximo(precios):
    return max(precios)