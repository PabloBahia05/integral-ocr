#!/usr/bin/env python3
"""
ocr_worker.py  —  Flask OCR worker para facturas argentinas
Corre en localhost:5001
"""

import re
import unicodedata
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
CORS(app)

# ── Patrones cabecera ─────────────────────────────────────────────────────────
PATRONES = {
    "numero":         r"(?i)nro\.?\s*:?\s*(\d{4}-\d{5,8})",
    "fecha":          r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
    "tipo_factura":   r"(?i)factura\s+([ABCME])\b",
    "condicion_pago": r"(?i)(contado|30\s*d[ií]as?|60\s*d[ií]as?|90\s*d[ií]as?|cuenta\s+corriente|1\s*d[ií]a\s*ff)",
    "subtotal":       r"(?i)subtotal\s*:?\s*\$?\s*([\d\.,]+)",
    "iva_pct":        r"(?i)iva\s+(?:insc\.?\s+)?(10[,\.]?5|21|27)\s*%",
    "iva":            r"(?i)iva\s+(?:insc\.?\s+)?(?:10[,\.]?5|21|27)[,\.]?0*\s*%\s*:?\s*([\d\.,]+)",
    "total":          r"(?i)total\s*:?\s*\$?\s*([\d\.,]+)",
    "moneda":         r"(?i)\b(USD|ARS|EUR)\b",
}

# Monto en formato argentino: 1.234,56
NUM_ARG = r'\d{1,3}(?:[.]\d{3})*[,]\d{2}'
# Cantidad: 4,00 o 4,5
CANT_PAT = r'\d+[,.]\d{1,3}'

# Palabras que indican fin de la tabla de ítems
STOP_RE = re.compile(
    r'(?i)^(subtotal|total|gravado|descuento|flete|perc|exento|'
    r'c\.a\.e|entrega|transporte|atendido|nota|original|n\.p\.|'
    r'dto\.|vto\.|iva\s+insc)'
)

# ── Helpers numéricos ─────────────────────────────────────────────────────────
def limpiar_numero(s):
    if not s:
        return None
    s = s.strip().replace(" ", "")
    if re.search(r'\d[.]\d{3}', s) or (s.count(',') == 1 and '.' not in s):
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None

def normalizar_fecha(s):
    if not s:
        return None
    partes = re.split(r'[/\-\.]', s)
    if len(partes) != 3:
        return s
    d, m, a = partes
    if len(a) == 2:
        a = '20' + a
    try:
        return f"{int(a):04d}-{int(m):02d}-{int(d):02d}"
    except ValueError:
        return s

def extraer_campo(texto, patron):
    m = re.search(patron, texto)
    if not m:
        return None
    return m.group(m.lastindex).strip() if m.lastindex else m.group(0).strip()

# ── Limpieza de líneas ────────────────────────────────────────────────────────
def es_espurio(c):
    """Caracteres que Tesseract genera al leer bordes de celdas."""
    if unicodedata.category(c) in ('Pd', 'Ps', 'Pe'):
        return True
    # Solo símbolos claramente no alfanuméricos
    if c in (chr(92), chr(39), chr(96), chr(34), '~'):
        return True
    return False

def limpiar_linea(linea):
    i = 0
    while i < len(linea) and es_espurio(linea[i]):
        i += 1
    return linea[i:]

def corregir_codigo(codigo):
    """
    Tesseract confunde S con $ al inicio de códigos de AGLOLAM.
    Ej: $14018F → S14018F, $11118S → S11118S
    """
    if not codigo:
        return codigo
    # $ al inicio seguido de dígito o letra → era una S
    return re.sub(r'^\$(?=[A-Za-z0-9])', 'S', codigo)

# ── Parser de ítem línea por línea ───────────────────────────────────────────
def parsear_linea_item(linea):
    """
    Parsea una línea como ítem de factura buscando desde el final:
      [CODIGO] CANTIDAD DESCRIPCION PRECIO_UNIT [(DESC%)] SUBTOTAL
    Devuelve dict o None si la línea no es un ítem válido.
    """
    linea = linea.strip()
    if not linea:
        return None

    # Buscar PRECIO y SUBTOTAL al final de la línea
    m_fin = re.search(
        r'(' + NUM_ARG + r')\s+(?:[(]\d+[,.]\d+%[)]\s+)?(' + NUM_ARG + r')\s*$',
        linea
    )
    if not m_fin:
        return None

    precio_str   = m_fin.group(1)
    subtotal_str = m_fin.group(2)
    resto        = linea[:m_fin.start()].strip()

    # Buscar CANTIDAD en el resto (primera ocurrencia de patrón cantidad)
    m_cant = re.search(r'(?<!\d)(' + CANT_PAT + r')(?!\d)\s+', resto)
    if not m_cant:
        return None

    cant_str   = m_cant.group(1)
    antes_cant = resto[:m_cant.start()].strip()   # posible código
    desc       = resto[m_cant.end():].strip()      # descripción

    # Limpiar porcentajes residuales de la descripción
    desc = re.sub(r'\s*[(]\d+[,.]\d+%[)]\s*$', '', desc).strip()

    # Código: alfanumérico, permitir $ al inicio (Tesseract confunde S con $)
    codigo = antes_cant if re.match(r'^[\$A-Za-z0-9][A-Za-z0-9]{1,19}$', antes_cant) else None
    if codigo:
        codigo = corregir_codigo(codigo)

    # Si no hay código antes de la cantidad, buscar si la descripción empieza
    # con un token que Tesseract pegó a ella
    if not codigo and desc:
        tokens = desc.split()
        if tokens and re.match(r'^[\$A-Za-z0-9][A-Za-z0-9]{1,19}$', tokens[0]) and len(tokens[0]) <= 12:
            codigo = corregir_codigo(tokens[0])
            desc = ' '.join(tokens[1:]).strip()

    # Descartar si no hay descripción ni código
    if not desc and not codigo:
        return None

    return {
        "codigo":       codigo,
        "descripcion":  desc or None,
        "cantidad":     limpiar_numero(cant_str),
        "precio_unit":  limpiar_numero(precio_str),
        "subtotalprod": limpiar_numero(subtotal_str),
    }

def extraer_items(texto):
    items = []
    print("[OCR] === LÍNEAS ANALIZADAS COMO ÍTEMS ===")
    for linea_raw in texto.split('\n'):
        linea = limpiar_linea(linea_raw)
        if STOP_RE.search(linea):
            continue
        item = parsear_linea_item(linea)
        if item:
            print(f"[OCR]  raw: {repr(linea_raw)}")
            print(f"[OCR]  → codigo={item['codigo']} desc={item['descripcion']}")
            items.append(item)
    print("[OCR] === FIN ÍTEMS ===")
    return items

# ── Preprocesamiento imagen ───────────────────────────────────────────────────
def preparar_imagen(file_obj):
    img = Image.open(file_obj).convert('RGB')
    w, h = img.size
    if w < 2400:
        factor = 2400 / w
        img = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img, cutoff=2)
    return img

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post('/ocr')
def ocr():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se recibió imagen'}), 400

    img   = preparar_imagen(request.files['imagen'])
    texto = pytesseract.image_to_string(img, config='--psm 6 -l spa+eng')

    m_tipo = re.search(PATRONES['tipo_factura'], texto)
    tipo_factura = m_tipo.group(1).upper() if m_tipo else None

    m_iva_pct = re.search(PATRONES['iva_pct'], texto)
    iva_pct = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else None

    factura = {
        'numero':         extraer_campo(texto, PATRONES['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES['fecha'])),
        'tipo_factura':   tipo_factura,
        'condicion_pago': extraer_campo(texto, PATRONES['condicion_pago']),
        'subtotal':       limpiar_numero(extraer_campo(texto, PATRONES['subtotal'])),
        'iva_pct':        iva_pct,
        'iva':            limpiar_numero(extraer_campo(texto, PATRONES['iva'])),
        'total':          limpiar_numero(extraer_campo(texto, PATRONES['total'])),
        'moneda':         extraer_campo(texto, PATRONES['moneda']) or 'ARS',
        'texto_raw':      texto,
    }

    items = extraer_items(texto)
    return jsonify({'factura': factura, 'items': items})


@app.get('/ping')
def ping():
    return jsonify({'ok': True})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
