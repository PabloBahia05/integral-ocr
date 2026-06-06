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

# ── Detección de proveedor ────────────────────────────────────────────────────
PROVEEDORES = {
    "placasur":  re.compile(r'(?i)placasur|DISTRIBUIDORA PLACASUR'),
    "cantochap": re.compile(r'(?i)cantochap'),
    "aglolam":   re.compile(r'(?i)aglolam'),
    "bonzini":   re.compile(r'(?i)bonzini|HERRAJES BONZINI'),
}

def detectar_proveedor(texto):
    for nombre, patron in PROVEEDORES.items():
        if patron.search(texto):
            return nombre
    return "generico"

# ── Patrones específicos PlacaSur ─────────────────────────────────────────────
PATRONES_PLACASUR = {
    # N°: 00018-00018355
    "numero":         r"N[°º]:\s*([\d]{5}-[\d]{8})",
    "fecha":          r"Fecha:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "vencimiento":    r"Vencimiento:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "tipo_factura":   r"Factura\s+([A-Z])\b",
    "cae":            r"CAE:\s*(\d+)",
    "cae_vto":        r"VTO:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "condicion_pago": r"Condicion de Venta:\s*(.+?)(?:\n|Cnel|$)",
    "cliente_nombre": r"Señor\(es\):\s*(.+?)(?:\n|$)",
    "cliente_cuit":   r"CUIT:\s*([\d-]+)",
    # Totales: "Gravado ARS 1.561.659,29"
    "gravado":        r"(?i)Gravado\s+ARS\s+([\d.]+,\d{2})",
    "subtotal":       r"(?i)Subtotal\s+ARS\s+([\d.]+,\d{2})",
    "iva":            r"(?i)IVA:\s+ARS\s+([\d.]+,\d{2})",
    "iva_pct":        r"(?i)IVA\s+(21|10[,.]?5|27)\s*%",
    "pers_IIBB":      r"(?i)Total Percep\.\s*:\s*ARS\s+([\d.]+,\d{2})",
    "total":          r"(?i)Total:\s*ARS\s+([\d.]+,\d{2})",
}

# Items PlacaSur: CODIGO  DESCRIPCION  CANTIDAD  UN  (UNxART)  PRECIO_ARS  PRECIO_DESC_ARS  DESC%  TOTAL_ARS
# Ej: "EHB35C9 BISAGRAS EUROHARD 35 C 9 250,00 UN (1) 607,09 394,76 34,98% 98.690,82"
_NUM_ARG_PS = r'[\d]{1,3}(?:[.][\d]{3})*[,][\d]{2}'
_CANT_PS    = r'[\d]+[,][\d]{2}'
_DESC_PS    = r'[\d]+[,][\d]{2}%'

ITEM_RE_PLACASUR = re.compile(
    r'^(\S+)\s+'           # codigo
    r'(.+?)\s+'            # descripcion
    r'(' + _CANT_PS + r')\s+'   # cantidad
    r'UN\s+\(\d+\)\s+'    # unidad
    r'(' + _NUM_ARG_PS + r')\s+'  # precio ARS
    r'(' + _NUM_ARG_PS + r')\s+'  # precio desc ARS
    r'' + _DESC_PS + r'\s+'       # descuento %
    r'(' + _NUM_ARG_PS + r')'     # total ARS
    r'\s*$',
    re.IGNORECASE
)

LOTE_RE_PLACASUR = re.compile(r'(?i)^Lote\s+Nro\s+Lote:', re.MULTILINE)

def extraer_items_placasur(texto):
    items = []
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea:
            continue
        if LOTE_RE_PLACASUR.match(linea):
            continue
        m = ITEM_RE_PLACASUR.match(linea)
        if m:
            items.append({
                "codigo":       m.group(1),
                "descripcion":  m.group(2).strip(),
                "cantidad":     limpiar_numero(m.group(3)),
                "precio_unit":  limpiar_numero(m.group(5)),  # precio con descuento aplicado
                "subtotalprod": limpiar_numero(m.group(6)),
            })
    return items

def extraer_totales_placasur(texto):
    subtotal = limpiar_numero(extraer_campo(texto, PATRONES_PLACASUR['subtotal']))
    iva      = limpiar_numero(extraer_campo(texto, PATRONES_PLACASUR['iva']))
    pers     = limpiar_numero(extraer_campo(texto, PATRONES_PLACASUR['pers_IIBB']))
    total    = limpiar_numero(extraer_campo(texto, PATRONES_PLACASUR['total']))
    if not total and subtotal:
        total = round((subtotal or 0) + (iva or 0) + (pers or 0), 2)
    return subtotal or 0, iva or 0, pers or 0, total, total

def parsear_placasur(texto):
    subtotal_val, iva_val, pers_val, total_leido, total = extraer_totales_placasur(texto)
    m_iva_pct = re.search(PATRONES_PLACASUR['iva_pct'], texto)
    iva_pct = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else 21.0
    factura = {
        'proveedor':      'placasur',
        'numero':         extraer_campo(texto, PATRONES_PLACASUR['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_PLACASUR['fecha'])),
        'vencimiento':    normalizar_fecha(extraer_campo(texto, PATRONES_PLACASUR['vencimiento'])),
        'tipo_factura':   extraer_campo(texto, PATRONES_PLACASUR['tipo_factura']),
        'cae':            extraer_campo(texto, PATRONES_PLACASUR['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_PLACASUR['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_PLACASUR['condicion_pago']) or '').strip(),
        'cliente_nombre': extraer_campo(texto, PATRONES_PLACASUR['cliente_nombre']),
        'cliente_cuit':   extraer_campo(texto, PATRONES_PLACASUR['cliente_cuit']),
        'subtotal':       subtotal_val or None,
        'iva_pct':        iva_pct,
        'iva':            iva_val or None,
        'pers_IIBB':      pers_val or None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    items = extraer_items_placasur(texto)
    return factura, items

NUM_ARG = r'\d{1,3}(?:[.]\d{3})*[,]\d{2}'

# ── Patrones específicos Bonzini ──────────────────────────────────────────────
PATRONES_BONZINI = {
    # Factura: "Fecha de Emision: 04/05/2026"
    "fecha_factura":  r"Fecha de Emision[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
    # Presupuesto: "FECHA: 05/05/2026"
    "fecha_presup":   r"FECHA[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
    "numero":         r"(?:N[°º\.:]+\s*)?(\d{4}-\d{8})",
    "tipo_factura":   r"(?:COD\.?\s*\d+\s*)?\n?\s*([A-Z])\s*\n",
    "cae":            r"N[°º]?\s*de\s*CAE[:\s]*([\d]+)",
    "cae_vto":        r"Fecha de Vto de CAE[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
    # Factura: "Condicion de Venta: ..."  | Presupuesto: "Condiciones de Venta: ..."
    "condicion_pago": r"Condici[oó]n(?:es)? de Venta[:\s]*(.+?)(?:\n|$)",
    # Factura: "Apellido Nombre/Razón Social: ..."  | Presupuesto: "SEÑORES: ..."
    "cliente_factura": r"Apellido Nombre/Raz[oó]n Social[:\s]*(.+?)(?:\n|$)",
    "cliente_presup":  r"SE[ÑN]ORES[:\s]*(.+?)(?:\n|$)",
    "cliente_cuit":   r"CUIT[:\s]*([\d]+-[\d]+-[\d])",
    # Totales factura (formato AR con coma decimal)
    "subtotal":       r"Importe Neto Gravado:\$?\s*([\d.]+,\d{2})",
    "iva":            r"Total Iva:\s*\$?\s*([\d.]+,\d{2})",
    "total_factura":  r"Importe Total:\s*\$?\s*([\d.]+,\d{2})",
    # Total presupuesto (formato con punto decimal: 119227.36)
    "total_presup":   r"TOTAL[:\s]*([\d]+[.,]\d{2})",
    "iva_pct":        r"IVA\s+(21|10[,.]?5|27)[,.]?0*\s*%",
}

def _es_presupuesto_bonzini(texto):
    """Detecta si el documento Bonzini es un presupuesto (no factura AFIP)."""
    return bool(re.search(r'(?i)presupuesto', texto)) or \
           bool(re.search(r'\b9997-\d{8}\b', texto))

def _limpiar_num_bonzini(s, es_presup=False):
    """
    Presupuesto Bonzini usa punto como decimal (65279.98).
    Factura Bonzini usa formato AR (65.279,98).
    """
    if not s:
        return None
    s = s.strip().replace(' ', '')
    if es_presup:
        # Punto es decimal, coma es miles (poco probable pero por si acaso)
        s = s.replace(',', '')
        try:
            return float(s)
        except ValueError:
            return None
    else:
        # Formato argentino: punto = miles, coma = decimal
        if re.search(r'\d[.]\d{3}', s) or (s.count(',') == 1 and '.' not in s):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
        try:
            return float(s)
        except ValueError:
            return None

def extraer_items_bonzini(texto, es_presup=False):
    """
    Factura:     DESCRIPCION  CANTIDAD  PRECIO_UNIT  FINAL_C_IVA   (números AR)
    Presupuesto: CANTIDAD  DESCRIPCION  PRECIO  IMPORTE            (punto decimal)
    Descuentos: línea con valor negativo.
    """
    items = []

    if es_presup:
        # Número con punto decimal (ej: 65279.98 o -11332.61)
        NUM_P = r'-?[\d]+[.][\d]{2}'
        # Formato: CANTIDAD  DESCRIPCION  PRECIO  IMPORTE
        item_re = re.compile(
            r'^(\d+(?:[,.]\d+)?)\s+'          # cantidad
            r'(.+?)\s+'                        # descripcion
            r'(' + NUM_P + r')\s+'             # precio
            r'(' + NUM_P + r')\s*$',
        )
        # Descuento general en presupuesto: "Descuento General X%  -precio  -importe"
        desc_re = re.compile(
            r'^(Descuento\s+General[^\n]*?)\s+(' + NUM_P + r')\s+(' + NUM_P + r')\s*$',
            re.IGNORECASE
        )
        en_items = False
        for linea_raw in texto.split('\n'):
            linea = linea_raw.strip()
            if not linea:
                continue
            if re.search(r'(?i)^(cantidad|descripci[oó]n)', linea):
                en_items = True
                continue
            if not en_items:
                continue
            if re.search(r'(?i)^(observaciones|total\b)', linea):
                break
            m_desc = desc_re.match(linea)
            if m_desc:
                precio = _limpiar_num_bonzini(m_desc.group(2), True)
                importe = _limpiar_num_bonzini(m_desc.group(3), True)
                items.append({
                    "codigo":       None,
                    "descripcion":  m_desc.group(1).strip(),
                    "cantidad":     1,
                    "precio_unit":  precio,
                    "subtotalprod": importe,
                })
                continue
            m = item_re.match(linea)
            if m:
                items.append({
                    "codigo":       None,
                    "descripcion":  m.group(2).strip(),
                    "cantidad":     _limpiar_num_bonzini(m.group(1), True),
                    "precio_unit":  _limpiar_num_bonzini(m.group(3), True),
                    "subtotalprod": _limpiar_num_bonzini(m.group(4), True),
                })
    else:
        # Formato factura AFIP: números AR con coma decimal
        NUM = r'[\d]{1,3}(?:[.][\d]{3})*[,][\d]{2}'
        item_re = re.compile(
            r'^(.+?)\s+'
            r'(\d+(?:[,.]\d+)?)\s+'
            r'(' + NUM + r')\s+'
            r'(' + NUM + r')\s*$',
            re.IGNORECASE
        )
        desc_neg_re = re.compile(
            r'^(Desc[^\n]*?)\s+(\d+)\s+(-[\d.]+,\d{2})\s+(-[\d.]+,\d{2})\s*$',
            re.IGNORECASE
        )
        en_items = False
        for linea_raw in texto.split('\n'):
            linea = linea_raw.strip()
            if not linea:
                continue
            if re.search(r'(?i)producto[/\s]*servicio', linea):
                en_items = True
                continue
            if not en_items:
                continue
            if re.search(r'(?i)^(observaciones|otros\s+tributos|importe\s+neto|descuento\s+general\b)', linea):
                break
            m_neg = desc_neg_re.match(linea)
            if m_neg:
                items.append({
                    "codigo":       None,
                    "descripcion":  m_neg.group(1).strip(),
                    "cantidad":     _limpiar_num_bonzini(m_neg.group(2)),
                    "precio_unit":  _limpiar_num_bonzini(m_neg.group(3)),
                    "subtotalprod": _limpiar_num_bonzini(m_neg.group(4)),
                })
                continue
            m = item_re.match(linea)
            if m:
                items.append({
                    "codigo":       None,
                    "descripcion":  m.group(1).strip(),
                    "cantidad":     _limpiar_num_bonzini(m.group(2)),
                    "precio_unit":  _limpiar_num_bonzini(m.group(3)),
                    "subtotalprod": _limpiar_num_bonzini(m.group(4)),
                })
    return items

def parsear_bonzini(texto):
    es_presup = _es_presupuesto_bonzini(texto)

    # Fecha
    fecha_raw = extraer_campo(texto, PATRONES_BONZINI['fecha_factura']) \
                or extraer_campo(texto, PATRONES_BONZINI['fecha_presup'])

    # Cliente
    cliente = extraer_campo(texto, PATRONES_BONZINI['cliente_factura']) \
              or extraer_campo(texto, PATRONES_BONZINI['cliente_presup'])

    # Tipo de documento
    if es_presup:
        tipo_doc = "Presupuesto"
    else:
        tipo_doc = extraer_campo(texto, PATRONES_BONZINI['tipo_factura'])

    # Totales
    if es_presup:
        total_raw = extraer_campo(texto, PATRONES_BONZINI['total_presup'])
        total    = _limpiar_num_bonzini(total_raw, True)
        subtotal = total  # sin IVA desglosado
        iva      = 0.0
        iva_pct  = 0.0
    else:
        subtotal = _limpiar_num_bonzini(extraer_campo(texto, PATRONES_BONZINI['subtotal']))
        iva      = _limpiar_num_bonzini(extraer_campo(texto, PATRONES_BONZINI['iva']))
        total    = _limpiar_num_bonzini(extraer_campo(texto, PATRONES_BONZINI['total_factura']))
        m_iva_pct = re.search(PATRONES_BONZINI['iva_pct'], texto)
        iva_pct  = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else 21.0
        if not total and subtotal:
            total = round((subtotal or 0) + (iva or 0), 2)

    factura = {
        'proveedor':      'bonzini',
        'es_presupuesto': es_presup,
        'numero':         extraer_campo(texto, PATRONES_BONZINI['numero']),
        'fecha':          normalizar_fecha(fecha_raw),
        'tipo_factura':   tipo_doc,
        'cae':            None if es_presup else extraer_campo(texto, PATRONES_BONZINI['cae']),
        'cae_vto':        None if es_presup else normalizar_fecha(extraer_campo(texto, PATRONES_BONZINI['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_BONZINI['condicion_pago']) or '').strip(),
        'cliente_nombre': (cliente or '').strip(),
        'cliente_cuit':   extraer_campo(texto, PATRONES_BONZINI['cliente_cuit']),
        'subtotal':       subtotal or None,
        'iva_pct':        iva_pct,
        'iva':            iva,
        'pers_IIBB':      None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    items = extraer_items_bonzini(texto, es_presup)
    return factura, items

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

    proveedor = detectar_proveedor(texto)
    app.logger.warning("[OCR] proveedor detectado: %s", proveedor)

    if proveedor == 'placasur':
        factura, items = parsear_placasur(texto)
    elif proveedor == 'bonzini':
        factura, items = parsear_bonzini(texto)
    else:
        m_tipo = re.search(PATRONES['tipo_factura'], texto)
        tipo_factura = m_tipo.group(1).upper() if m_tipo else None
        m_iva_pct = re.search(PATRONES['iva_pct'], texto)
        iva_pct = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else None
        subtotal_val, iva_val, pers_val, total_leido, total = extraer_totales(texto)
        app.logger.warning(
            "[OCR] total_leido=%s sub=%s iva=%s pers=%s total_final=%s",
            total_leido, subtotal_val, iva_val, pers_val, total)
        factura = {
            'proveedor':      proveedor,
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

    proveedor = detectar_proveedor(texto)
    app.logger.warning("[OCR-PDF] proveedor detectado: %s", proveedor)

    if proveedor == 'placasur':
        factura, items = parsear_placasur(texto)
    elif proveedor == 'bonzini':
        factura, items = parsear_bonzini(texto)
    else:
        m_tipo   = re.search(PATRONES['tipo_factura'], texto)
        tipo_factura = m_tipo.group(1).upper() if m_tipo else None
        m_iva_pct = re.search(PATRONES['iva_pct'], texto)
        iva_pct   = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else None
        subtotal_val, iva_val, pers_val, total_leido, total = extraer_totales(texto)
        app.logger.warning(
            "[OCR-PDF] total_leido=%s sub=%s iva=%s pers=%s total_final=%s",
            total_leido, subtotal_val, iva_val, pers_val, total)
        factura = {
            'proveedor':      proveedor,
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
