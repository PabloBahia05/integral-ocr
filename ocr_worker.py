#!/usr/bin/env python3
"""
ocr_worker.py  —  Flask OCR worker para facturas argentinas
"""

import re
import unicodedata
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import pytesseract
import pdfplumber
import os

app = Flask(__name__)
CORS(app)

# ── Patrones cabecera ─────────────────────────────────────────────────────────
PATRONES = {
    "numero":         r"(?i)nro\.?\s*:?\s*(\d{4,5}\s*[-–]\s*\d{5,8})",
    "fecha":          r"(?i)(?:fecha[^:]*?:?\s*)?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    "tipo_factura":   r"(?i)factura\s+([ABCME])\b",
    "condicion_pago": r"(?i)(contado|30\s*d[ií]as?|60\s*d[ií]as?|90\s*d[ií]as?|cuenta\s+corriente|1\s*d[ií]a\s*ff)",
    # subtotal: formato Aglolam "Subtotal: 613.226,88" y Cantochap tabla "561.376,08"
    "subtotal":       r"(?i)subtotal\s*[:\|]?\s*\$?\s*([\d\.,]+)",
    "iva_pct":        r"(?i)iva\s+(?:insc\.?\s+)?(10[,\.]?5|21|27)\s*%",
    # iva: formato Aglolam "Iva Insc. 21,00 %: 128.777,64" y Cantochap "21,00% 117.888,98"
    "iva":            r"(?i)iva\s+(?:insc\.?\s+)?(?:total\s*)?(?:10[,\.]?5|21|27)[,\.]?0*\s*%\s*:?\s*([\d\.,]+)",
    # total: formato Aglolam "TOTAL: 743.230,97" y Cantochap "$\s*679.265,06"
    "total":          r"(?is)(?:^|\s)\$?\s*([\d]{1,3}(?:[.][\d]{3})+[,][\d]{2})\s*$",
    "moneda":         r"(?i)\b(USD|ARS|EUR)\b",
    "pers_IIBB":      r"(?i)perc[./]?\s*i{1,2}b{1,2}\s*:?\s*(?:[A-Za-z 0-9]+\s+)?([\d]+(?:[.][\d]{3})*[,][\d]{2})",
}

NUM_ARG = r'\d{1,3}(?:[.]\d{3})*[,]\d{2}'
CANT_PAT = r'\d+[,.]\d{1,3}'

STOP_RE = re.compile(
    r'(?i)^(subtotal|total|gravado|descuento|flete|perc|exento|'
    r'c\.a\.e|entrega|transporte|atendido|nota|original|n\.p\.|'
    r'dto\.|vto\.|iva\s+insc|bultos|son\s+pesos|pagos|cta\.|fecha\s+vto|'
    r'[\d.,]+\s+[\d.,]+\s+\d+[,.]\d+%)'  # fila de totales formato Cantochap
)

UMED_RE = r'(?:UN|M2|KG|MT|ML|JGO|SET|PAR|LT|CM|MM|GL|U)'

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def es_espurio(c):
    if unicodedata.category(c) in ('Pd', 'Ps', 'Pe'):
        return True
    if c in (chr(92), chr(39), chr(96), chr(34), '~'):
        return True
    return False

def limpiar_linea(linea):
    i = 0
    while i < len(linea) and es_espurio(linea[i]):
        i += 1
    return linea[i:]

def corregir_codigo(codigo):
    if not codigo:
        return codigo
    return re.sub(r'^\$(?=[A-Za-z0-9])', 'S', codigo)

def parsear_linea_item(linea):
    linea = linea.strip()
    if not linea:
        return None

    # Formato Cantochap: CODIGO  CANTIDAD(NUM_ARG)  UMED  DESCRIPCION  PRECIO  IMPORTE
    # Ej: "400122 2.400,00 UN PVC BLANCO TX 22MM - 040MM SIN COLA 101,95 244.677,60"
    m_cant = re.match(
        r'^(\S+)\s+(' + NUM_ARG + r')\s+' + UMED_RE + r'\s+(.+?)\s+(' + NUM_ARG + r')\s+(' + NUM_ARG + r')\s*$',
        linea, re.IGNORECASE
    )
    if m_cant:
        return {
            "codigo":       corregir_codigo(m_cant.group(1)),
            "descripcion":  m_cant.group(3).strip(),
            "cantidad":     limpiar_numero(m_cant.group(2)),
            "precio_unit":  limpiar_numero(m_cant.group(4)),
            "subtotalprod": limpiar_numero(m_cant.group(5)),
        }

    # Formato Aglolam: CODIGO  CANTIDAD  DESCRIPCION  (DESC%)  PRECIO  IMPORTE
    m_fin = re.search(
        r'(' + NUM_ARG + r')\s+(?:[(]\d+[,.]\d+%[)]\s+)?(' + NUM_ARG + r')\s*$',
        linea
    )
    if not m_fin:
        return None
    precio_str   = m_fin.group(1)
    subtotal_str = m_fin.group(2)
    resto        = linea[:m_fin.start()].strip()
    m_cant = re.search(r'(?<!\d)(' + CANT_PAT + r')(?!\d)\s+', resto)
    if not m_cant:
        return None
    cant_str   = m_cant.group(1)
    antes_cant = resto[:m_cant.start()].strip()
    desc       = resto[m_cant.end():].strip()
    desc = re.sub(r'\s*[(]\d+[,.]\d+%[)]\s*$', '', desc).strip()
    codigo = antes_cant if re.match(r'^[\$A-Za-z0-9][A-Za-z0-9]{1,19}$', antes_cant) else None
    if codigo:
        codigo = corregir_codigo(codigo)
    if not codigo and desc:
        tokens = desc.split()
        if tokens and re.match(r'^[\$A-Za-z0-9][A-Za-z0-9]{1,19}$', tokens[0]) and len(tokens[0]) <= 12:
            codigo = corregir_codigo(tokens[0])
            desc = ' '.join(tokens[1:]).strip()
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
    for linea_raw in texto.split('\n'):
        linea = limpiar_linea(linea_raw)
        if STOP_RE.search(linea):
            continue
        item = parsear_linea_item(linea)
        if item:
            items.append(item)
    return items

def extraer_totales(texto):
    """
    Extrae subtotal, iva, pers_IIBB y total soportando múltiples formatos:
    - Aglolam: "Subtotal: 613.226,88" / "Iva Insc. 21,00 %: 128.777,64" / "TOTAL: 743.230,97"
    - Cantochap: tabla con headers "Subtotal ... TOTAL" y valores en línea siguiente + "$ 679.265,06"
    """
    # ── Formato Cantochap: tabla con headers ──────────────────────────────────
    # Línea: "Subtotal Descuentos Neto IVA No Gravado Percepciones TOTAL"
    # Siguiente: "561.376,08 561.376,08 21,00% 117.888,98 0,00 0,00"
    # Línea total: "$ 679.265,06"
    subtotal_val = 0
    iva_val      = 0
    pers_val     = 0
    total_leido  = None

    m_tabla = re.search(
        r'(?i)subtotal[^\n]*total\s*\n'
        r'\s*([\d.]+,\d{2})'
        r'[^\n]*?([\d.,]+)%\s*([\d.]+,\d{2})',
        texto
    )
    if m_tabla:
        subtotal_val = limpiar_numero(m_tabla.group(1)) or 0
        iva_val      = limpiar_numero(m_tabla.group(3)) or 0
        # Total con $ en línea aparte
        m_total_pesos = re.search(r'\$\s*([\d]{1,3}(?:[.][\d]{3})+[,][\d]{2})', texto)
        if m_total_pesos:
            total_leido = limpiar_numero(m_total_pesos.group(1))
    else:
        # ── Formato Aglolam: etiquetas explícitas ─────────────────────────────
        subtotal_val = limpiar_numero(extraer_campo(texto, PATRONES['subtotal'])) or 0
        iva_val      = limpiar_numero(extraer_campo(texto, PATRONES['iva'])) or 0
        pers_val     = limpiar_numero(extraer_campo(texto, PATRONES['pers_IIBB'])) or 0
        m_total_label = re.search(
            r'(?i)(?<!sub)total\s*:?\s*\$?\s*([\d]{1,3}(?:[.][\d]{3})+[,][\d]{2})', texto
        )
        if m_total_label:
            total_leido = limpiar_numero(m_total_label.group(1))

    if total_leido:
        total = total_leido
    else:
        total = round(subtotal_val + iva_val + pers_val, 2) if subtotal_val else None

    return subtotal_val, iva_val, pers_val, total_leido, total


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
    app.logger.warning("OCR TEXTO:\n%s", texto)
    
    m_tipo = re.search(PATRONES['tipo_factura'], texto)
    tipo_factura = m_tipo.group(1).upper() if m_tipo else None

    m_iva_pct = re.search(PATRONES['iva_pct'], texto)
    iva_pct = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else None

    subtotal_val, iva_val, pers_val, total_leido, total = extraer_totales(texto)

    app.logger.warning(
        "[OCR] total_leido=%s sub=%s iva=%s pers=%s total_final=%s",
        total_leido, subtotal_val, iva_val, pers_val, total)

    factura = {
        'numero':         extraer_campo(texto, PATRONES['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES['fecha'])),
        'tipo_factura':   tipo_factura,
        'condicion_pago': extraer_campo(texto, PATRONES['condicion_pago']),
        'subtotal':       subtotal_val or None,
        'iva_pct':        iva_pct,
        'iva':            iva_val or None,
        'total':          total,
        'moneda':         extraer_campo(texto, PATRONES['moneda']) or 'ARS',
        'pers_IIBB':      pers_val or None,
        'texto_raw':      texto,
    }

    items = extraer_items(texto)
    return jsonify({'factura': factura, 'items': items})


@app.post('/ocr-pdf')
@app.post('/ocr-pdf-preview')
def ocr_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No se recibió archivo PDF'}), 400

    archivo = request.files['pdf']
    if not archivo.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'El archivo no es un PDF'}), 400

    # Guardar temporalmente y extraer texto con pdfplumber
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        archivo.save(tmp.name)
        tmp_path = tmp.name

    try:
        texto = ''
        with pdfplumber.open(tmp_path) as pdf:
            for pagina in pdf.pages:
                t = pagina.extract_text()
                if t:
                    texto += t + '\n'

        # Si pdfplumber no extrajo nada (PDF escaneado), hacer OCR página por página
        if not texto.strip():
            import fitz  # PyMuPDF
            doc = fitz.open(tmp_path)
            for num_pag in range(len(doc)):
                pix = doc[num_pag].get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                from io import BytesIO
                img = Image.open(BytesIO(img_data)).convert('RGB')
                img = ImageOps.grayscale(img)
                img = ImageOps.autocontrast(img, cutoff=2)
                texto += pytesseract.image_to_string(img, config='--psm 6 -l spa+eng') + '\n'
    finally:
        os.unlink(tmp_path)

    app.logger.warning("OCR-PDF TEXTO:\n%s", texto)

    m_tipo   = re.search(PATRONES['tipo_factura'], texto)
    tipo_factura = m_tipo.group(1).upper() if m_tipo else None

    m_iva_pct = re.search(PATRONES['iva_pct'], texto)
    iva_pct   = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else None

    subtotal_val, iva_val, pers_val, total_leido, total = extraer_totales(texto)

    app.logger.warning(
        "[OCR-PDF] total_leido=%s sub=%s iva=%s pers=%s total_final=%s",
        total_leido, subtotal_val, iva_val, pers_val, total)

    factura = {
        'numero':         extraer_campo(texto, PATRONES['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES['fecha'])),
        'tipo_factura':   tipo_factura,
        'condicion_pago': extraer_campo(texto, PATRONES['condicion_pago']),
        'subtotal':       subtotal_val or None,
        'iva_pct':        iva_pct,
        'iva':            iva_val or None,
        'total':          total,
        'moneda':         extraer_campo(texto, PATRONES['moneda']) or 'ARS',
        'pers_IIBB':      pers_val or None,
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
