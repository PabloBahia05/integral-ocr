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
    "bonzini":   re.compile(r'(?i)bonzini|HERRAJES BONZINI|\b9997-\d{8}\b'),
    "inomax":    re.compile(r'(?is)inomax|herrajesinomax|STAAL\s+S\.A\.S|(?=.*Subtotal Neto:)(?=.*CAE\s*Nro:)'),
    "intervidrio": re.compile(r'(?i)intervidrio|LA CASA DE LOS CRISTALES'),
    "metalurgica_gg": re.compile(r'(?i)metal[uú]rgica\s*g\.?\s*g\.?|30-71051717-3'),
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
    # tipo_factura: en facturas viejas aparece "Factura A"; en el layout nuevo pdfplumber
    # aplana las dos columnas del encabezado y la letra queda ANTES, sola en su línea
    # ("A\nFactura\n01\n..."). Se intentan ambos patrones.
    "tipo_factura_nuevo": r"(?m)^([ABCME])\s*\n(?=Factura)",
    "tipo_factura":   r"Factura\s+([A-Z])\b",
    "cae":            r"CAE:\s*(\d+)",
    "cae_vto":        r"VTO:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "condicion_pago": r"Condicion de Venta:\s*(.+?)(?:\n|Cnel|$)",
    # cliente_nombre: cortar antes de "Vencimiento:" que en el layout nuevo queda
    # pegado en la misma línea por el aplanado de columnas de pdfplumber.
    "cliente_nombre": r"Señor\(es\):\s*(.+?)(?:\s+Vencimiento:|\n|$)",
    # cliente_cuit: hay dos "CUIT:" en el documento (emisor y cliente); se toma el que
    # sigue a "Cliente Nro:" para no quedarse con el CUIT de PlacaSur. Si el layout no
    # trae "Cliente Nro:" pegado, cae al patrón genérico (comportamiento anterior).
    "cliente_cuit_cerca_cliente": r"Cliente Nro:[^\n]*\n\s*CUIT:\s*([\d-]+)",
    "cliente_cuit":   r"CUIT:\s*([\d-]+)",
    # Totales: "Gravado ARS 1.561.659,29" (formato AR) o "Gravado ARS 656,236.60" (formato US,
    # visto en facturas PlacaSur más nuevas). Se captura el número completo sin fijar qué
    # símbolo es el decimal; limpiar_numero() decide el formato al parsear.
    "gravado":        r"(?i)Gravado\s+ARS\s+([\d.,]+)",
    "subtotal":       r"(?i)Subtotal\s+ARS\s+([\d.,]+)",
    "iva":            r"(?i)IVA:\s+ARS\s+([\d.,]+)",
    "iva_pct":        r"(?i)IVA\s+(21|10[,.]?5|27)\s*%",
    "pers_IIBB":      r"(?i)Total Percep\.\s*:\s*ARS\s+([\d.,]+)",
    "total":          r"(?i)Total:\s*ARS\s+([\d.,]+)",
}

# Items PlacaSur: CODIGO  DESCRIPCION  CANTIDAD  UN  (UNxART)  PRECIO_ARS  PRECIO_DESC_ARS  DESC%  TOTAL_ARS
# Ej. formato AR: "EHB35C9 BISAGRAS EUROHARD 35 C 9 250,00 UN (1) 607,09 394,76 34,98% 98.690,82"
# Ej. formato US (facturas más nuevas): "EHB35C0 ... 250.00 UN (1) 607.09 394.76 34,98% 98,690.82"
# Acepta tanto "." como "," como separador, en cualquier posición (miles o decimal);
# limpiar_numero() resuelve el formato real al convertir cada valor capturado.
_NUM_ARG_PS = r'[\d]{1,3}(?:[.,][\d]{3})*[.,][\d]{2}'
_CANT_PS    = r'[\d]+[.,][\d]{2}'
_DESC_PS    = r'[\d]+[.,][\d]{2}%'

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
        'tipo_factura':   (extraer_campo(texto, PATRONES_PLACASUR['tipo_factura_nuevo'])
                            or extraer_campo(texto, PATRONES_PLACASUR['tipo_factura'])),
        'cae':            extraer_campo(texto, PATRONES_PLACASUR['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_PLACASUR['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_PLACASUR['condicion_pago']) or '').strip(),
        'cliente_nombre': extraer_campo(texto, PATRONES_PLACASUR['cliente_nombre']),
        'cliente_cuit':   (extraer_campo(texto, PATRONES_PLACASUR['cliente_cuit_cerca_cliente'])
                            or extraer_campo(texto, PATRONES_PLACASUR['cliente_cuit'])),
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

# ── Patrones específicos Bonzini ──────────────────────────────────────────────
PATRONES_BONZINI = {
    "numero":         r"(\d{4}-\d{8})",
    "fecha_factura":  r"Fecha de Emision[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
    "fecha_presup":   r"FECHA[:\s—\-]+(\d{1,2}/\d{1,2}/\d{4})",
    "tipo_factura":   r"FACTURA\s*\n[^A-Z\n]*([A-Z])\b",
    "cae":            r"N[°º]?\s*de\s*CAE[:\s]*([\d]+)",
    "cae_vto":        r"Fecha de Vto de CAE[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
    # "Condicion de Venta: A 30 DIAS" — evitar capturar "Condicion frente al IVA"
    "condicion_pago": r"Condici[oó]n de Venta[:\s]*(.+?)(?:\n|$)",
    "cliente_factura":r"Apellido Nombre/Raz[oó]n Social[:\s]*(.+?)(?:\n|$)",
    "cliente_presup": r"SE[ÑN]ORES[:\s]*(.+?)(?:\n|$)",
    "cliente_cuit":   r"CUIT[:\s]*([\d]+-[\d]+-[\d])",
    # Totales Bonzini usan punto decimal (371812.73), no formato AR
    "subtotal":       r"Neto Gravado[:\$\s]+([\d]+[.,][\d]{2})",
    "iva":            r"Tot[a!]+[l!]\s+Iva[:\s\$]+([\d]+[.,][\d]{2})",
    "total_factura":  r"Importe Total[:\s\$]*\s*([\d]+[.,][\d]{2})",
    "total_presup":   r"TOTAL[:\s]+([\d]+[.,][\d]{2})",
    "iva_pct":        r"IVA\s+(21|10[,.]?5|27)[,.]?0*\s*%",
}

def _es_presupuesto_bonzini(texto):
    return bool(re.search(r'(?i)presupuesto', texto)) or bool(re.search(r'\b9997-\d{8}\b', texto))

def _limpiar_num_bonzini(s, es_presup=False):
    """
    Bonzini usa punto como decimal en AMBOS documentos (371812.73, 78080.67).
    Presupuesto: igual.
    """
    if not s:
        return None
    s = s.strip().replace(' ', '')
    # Si tiene coma como decimal (formato AR: 371.812,73) → convertir
    if re.search(r'\d[.]\d{3}', s) or (s.count(',') == 1 and '.' not in s):
        s = s.replace('.', '').replace(',', '.')
    else:
        # Punto como decimal (371812.73) — eliminar comas si las hay
        s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None

def extraer_items_bonzini(texto, es_presup=False):
    items = []
    if es_presup:
        NUM_P = r'-?[\d]+[.,][\d]{2}'
        item_re = re.compile(
            r'^(\d+(?:[,.]\d+)?)\s+(.+?)\s+(' + NUM_P + r')\s+[\S]*\s*(' + NUM_P + r')\s*$'
        )
        item_re2 = re.compile(
            r'^(\d+(?:[,.]\d+)?)\s+(.+?)\s+(' + NUM_P + r')\s+(' + NUM_P + r')\s*$'
        )
        desc_re = re.compile(
            r'^(?:\d+\s+)?(Descuento\s+General).*?(-[\d]+[.,][\d]{2})\s+(-[\d]+[.,][\d]{2})\s*$',
            re.IGNORECASE
        )
        desc_re2 = desc_re
        IGNORAR_PRESUP = re.compile(
            r'(?i)^(se[ñn]ores|domicilio|iva:|condicion|fecha|presupuesto|'
            r'n[°º]|ped|observ|total\b|remito|bahia|cuit|cantidad|descripci)'
        )
        for linea_raw in texto.split('\n'):
            linea = re.sub(r'^[\s|]+', '', linea_raw).strip()
            if not linea or len(linea) < 5:
                continue
            if IGNORAR_PRESUP.match(linea):
                continue
            # Descuento general (puede tener ruido entre precio e importe)
            m_d = desc_re.match(linea) or desc_re2.match(linea)
            if m_d:
                desc_limpia = re.sub(r'[\s.\-=%\d]+$', '', m_d.group(1)).strip()
                items.append({
                    "codigo": None, "descripcion": desc_limpia,
                    "cantidad": 1,
                    "precio_unit":  _limpiar_num_bonzini(m_d.group(2)),
                    "subtotalprod": _limpiar_num_bonzini(m_d.group(3)),
                })
                continue
            # Ítem normal
            m = item_re.match(linea) or item_re2.match(linea)
            if m:
                desc = m.group(2).strip()
                if len(desc) < 3:
                    continue
                cant  = _limpiar_num_bonzini(m.group(1))
                precio = _limpiar_num_bonzini(m.group(3))
                subtotal = _limpiar_num_bonzini(m.group(4))
                # Si el importe está corrupto, calcularlo
                if subtotal is None and cant and precio:
                    subtotal = round(cant * precio, 2)
                items.append({
                    "codigo": None, "descripcion": desc,
                    "cantidad":     cant,
                    "precio_unit":  precio,
                    "subtotalprod": subtotal,
                })
                continue
            # Fallback: línea con cantidad, descripción y precio pero importe corrupto
            # Formato: "2 DESCRIPCION 65279.98 : texto_corrupto"
            NUM_P2 = r'[\d]+[.,][\d]{2}'
            m_fb = re.match(
                r'^(\d+)\s+(.{5,}?)\s+(' + NUM_P2 + r')\s*[:\|]',
                linea
            )
            if m_fb:
                desc = m_fb.group(2).strip()
                cant = _limpiar_num_bonzini(m_fb.group(1))
                precio = _limpiar_num_bonzini(m_fb.group(3))
                if cant and precio and len(desc) >= 3:
                    items.append({
                        "codigo": None, "descripcion": desc,
                        "cantidad":     cant,
                        "precio_unit":  precio,
                        "subtotalprod": round(cant * precio, 2),
                    })
    else:
        # Factura Bonzini: DESCRIPCION  CANTIDAD  PRECIO_UNIT  FINAL_C_IVA
        # Números con punto decimal: 6079.04  14711.28
        # El OCR mezcla líneas — capturar directamente por patrón sin esperar header
        NUM_F = r'-?[\d]+(?:[.][\d]+)?'
        item_re = re.compile(
            r'^(.+?)\s+(\d+)\s+(' + NUM_F + r')\s+(' + NUM_F + r')\s*$',
            re.IGNORECASE
        )
        desc_neg_re = re.compile(
            r'^(Desc\s+[\d.]+%[^\n]*?)\s+(\d+)\s+(' + NUM_F + r')\s+(' + NUM_F + r')\s*$',
            re.IGNORECASE
        )
        desc_gen_re = re.compile(
            r'^(Descuento\s+General[^\n]*?)\s+(\d+)\s+(' + NUM_F + r')\s+(' + NUM_F + r')\s*$',
            re.IGNORECASE
        )
        # Líneas a ignorar aunque matcheen el patrón
        IGNORAR_RE = re.compile(
            r'(?i)^(cuit|condici|domicilio|fecha|ingresos|responsable|'
            r'para[ná]|villa|cod\.|ped|herrajes|observ|otros|percep|'
            r'detalle|importe|neto|sub:|iva|los\s+cambios|coronel|'
            r'\d{3}\s+\d{3})',
        )
        for linea_raw in texto.split('\n'):
            linea = re.sub(r'^[|H\s]+', '', linea_raw).strip()
            if not linea or len(linea) < 5:
                continue
            if IGNORAR_RE.match(linea):
                continue
            # Descuento general
            m_gen = desc_gen_re.match(linea)
            if m_gen:
                items.append({
                    "codigo": None, "descripcion": m_gen.group(1).strip(),
                    "cantidad":     _limpiar_num_bonzini(m_gen.group(2)),
                    "precio_unit":  _limpiar_num_bonzini(m_gen.group(3)),
                    "subtotalprod": _limpiar_num_bonzini(m_gen.group(4)),
                })
                continue
            # Descuento por ítem
            m_neg = desc_neg_re.match(linea)
            if m_neg:
                items.append({
                    "codigo": None, "descripcion": m_neg.group(1).strip(),
                    "cantidad":     _limpiar_num_bonzini(m_neg.group(2)),
                    "precio_unit":  _limpiar_num_bonzini(m_neg.group(3)),
                    "subtotalprod": _limpiar_num_bonzini(m_neg.group(4)),
                })
                continue
            # Ítem normal
            m = item_re.match(linea)
            if m:
                desc = m.group(1).strip()
                if len(desc) < 5 or re.match(r'^[\d\s.]+$', desc):
                    continue
                items.append({
                    "codigo": None, "descripcion": desc,
                    "cantidad":     _limpiar_num_bonzini(m.group(2)),
                    "precio_unit":  _limpiar_num_bonzini(m.group(3)),
                    "subtotalprod": _limpiar_num_bonzini(m.group(4)),
                })
    return items

def parsear_bonzini(texto):
    es_presup = _es_presupuesto_bonzini(texto)
    fecha_raw = extraer_campo(texto, PATRONES_BONZINI['fecha_factura']) \
                or extraer_campo(texto, PATRONES_BONZINI['fecha_presup'])
    cliente = extraer_campo(texto, PATRONES_BONZINI['cliente_factura']) \
              or extraer_campo(texto, PATRONES_BONZINI['cliente_presup'])
    if es_presup:
        tipo_doc = "Presupuesto"
        total_raw = extraer_campo(texto, PATRONES_BONZINI['total_presup'])
        total    = _limpiar_num_bonzini(total_raw, True)
        subtotal = total
        iva      = 0.0
        iva_pct  = 0.0
    else:
        tipo_doc = extraer_campo(texto, PATRONES_BONZINI['tipo_factura'])
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

# ── Patrones específicos INOMAX ───────────────────────────────────────────────
# INOMAX usa PUNTO como separador decimal y SIN separador de miles (9933.00,
# 72204.36), a diferencia del resto de los proveedores que usan formato AR.
# El "Factura A/B/C" y el logo están renderizados como imagen en el PDF, por lo
# que pdfplumber no los extrae como texto — tipo_factura queda en None.
PATRONES_INOMAX = {
    "numero":         r"(\d{5}-\d{8})",
    "fecha":          r"(?i)(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})",
    "condicion_pago": r"(?i)Condici[oó]n de Venta:\s*(.+?)(?:\n|$)",
    "cliente_nombre": r"(?i)Raz[oó]n Social:\s*(.+?)(?:\n|$)",
    "cliente_cuit":   r"(?i)\bCUIT:\s*([\d-]+)",
    "pedido":         r"(?i)Pedido:\s*([\d]+)",
    "subtotal":       r"(?i)Subtotal Neto:\s*\$?\s*([\d.]+)",
    "iva_pct":        r"(?i)IVA\s+(21|10[,.]?5|27)\s*%",
    "iva":            r"(?i)IVA\s+(?:21|10[,.]?5|27)\s*%:\s*\$?\s*([\d.]+)",
    "pers_IIBB":      r"(?i)Percepci[oó]n IIBB:\s*\$?\s*([\d.]+)",
    "total":          r"(?i)Importe Total:\s*\$?\s*([\d.]+)",
    "cae":            r"(?i)CAE\s*Nro:\s*([\d]+)",
    "cae_vto":        r"(?i)Fecha Vto\.\s*CAE:\s*(\d{1,2}/\d{1,2}/\d{4})",
}

_NUM_INOMAX  = r'[\d]+\.\d{2}'
_CANT_INOMAX = r'\d+(?:[.,]\d+)?'

# Ítem INOMAX: CODIGO(pegado a veces con la descripción) DESCRIPCION CANTIDADUNI P.Unit %Bon Importe
# Ej: "IM4008 Bisagra Pared ViCromado - Con Bisel 4UNI 9933.00 20.00 39732.00"
#     "LNR03-1Mampara Frontal Alum Negro 1UNI 72204.36 20.00 72204.36"   (código y desc. sin espacio)
#     "IM6006-Corrediza FrontaNEGRO 1UNI 54424.41 20.00 54424.41"       (código y desc. sin espacio)
ITEM_RE_INOMAX = re.compile(
    r'^([A-Z0-9]+(?:-\d+)?)'          # codigo
    r'(.*?)\s*'                       # descripcion (puede venir pegada al código)
    r'(' + _CANT_INOMAX + r')\s*UNI\s+'  # cantidad + UNI (a veces pegado)
    r'(' + _NUM_INOMAX + r')\s+'      # precio unitario (valorlista/precio_unit)
    r'(' + _NUM_INOMAX + r')\s+'      # % bonificación
    r'(' + _NUM_INOMAX + r')'         # importe
    r'\s*$',
    re.IGNORECASE
)

STOP_RE_INOMAX = re.compile(
    r'(?i)^(codigo\s+descripci|subtotal|bonificaci[oó]n|iva\s|percepci[oó]n|'
    r'importe\s+total|son\s+pesos|tipo\s+de\s+cambio|cae\s+nro)'
)


def _limpiar_num_inomax(s):
    """INOMAX usa punto decimal sin separador de miles (ej: 166360.77)."""
    if not s:
        return None
    s = s.strip().replace(' ', '').replace('$', '')
    if ',' in s and '.' in s:
        # formato con miles tipo 1,234.56
        s = s.replace(',', '')
    elif ',' in s and '.' not in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None


def extraer_items_inomax(texto):
    items = []
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea or STOP_RE_INOMAX.match(linea):
            continue
        m = ITEM_RE_INOMAX.match(linea)
        if m:
            desc = m.group(2).strip().lstrip('-').strip()
            items.append({
                "codigo":       m.group(1),
                "descripcion":  desc,
                "cantidad":     _limpiar_num_inomax(m.group(3)),
                "precio_unit":  _limpiar_num_inomax(m.group(4)),
                "bonif_pct":    _limpiar_num_inomax(m.group(5)),
                "subtotalprod": _limpiar_num_inomax(m.group(6)),
            })
    return items


def parsear_inomax(texto):
    subtotal = _limpiar_num_inomax(extraer_campo(texto, PATRONES_INOMAX['subtotal']))
    iva      = _limpiar_num_inomax(extraer_campo(texto, PATRONES_INOMAX['iva']))
    pers     = _limpiar_num_inomax(extraer_campo(texto, PATRONES_INOMAX['pers_IIBB']))
    total    = _limpiar_num_inomax(extraer_campo(texto, PATRONES_INOMAX['total']))
    if not total and subtotal:
        total = round((subtotal or 0) + (iva or 0) + (pers or 0), 2)
    m_iva_pct = re.search(PATRONES_INOMAX['iva_pct'], texto)
    iva_pct = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else 21.0
    factura = {
        'proveedor':      'inomax',
        'numero':         extraer_campo(texto, PATRONES_INOMAX['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_INOMAX['fecha'])),
        'tipo_factura':   None,  # se renderiza como imagen/logo, no está en el texto extraído
        'cae':            extraer_campo(texto, PATRONES_INOMAX['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_INOMAX['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_INOMAX['condicion_pago']) or '').strip(),
        'cliente_nombre': (extraer_campo(texto, PATRONES_INOMAX['cliente_nombre']) or '').strip(),
        'cliente_cuit':   extraer_campo(texto, PATRONES_INOMAX['cliente_cuit']),
        'pedido':         extraer_campo(texto, PATRONES_INOMAX['pedido']),
        'subtotal':       subtotal or None,
        'iva_pct':        iva_pct,
        'iva':            iva or None,
        'pers_IIBB':      pers or None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    items = extraer_items_inomax(texto)
    return factura, items


# ── Patrones específicos Intervidrio (La Casa de los Cristales) ──────────────
# Formato de ítem, todo en una sola línea impresa:
# "2,00HO Templado Float Incoloro 8 mm 0,393 x 1,900 1,49 109.696,13 163.447,23"
# donde: cantidad_hojas | descripcion | ancho x largo | cantidad_m2 | precio_unit | importe
# La descripcion puede continuar en líneas siguientes sin datos numéricos
# (ej. "segun dibujo", "Paño fijo").
NUM_ARG_INTERVIDRIO = r'\d{1,3}(?:[.]\d{3})*[,]\d{2}'
# Tolera el error típico de Tesseract "37.091 ,86" (espacio antes de la coma)
NUM_ARG_INTERVIDRIO_OCR = r'\d{1,3}(?:[.]\d{3})*\s?,\s?\d{2}'
CANT_PAT_INTERVIDRIO = r'\d+[,.]\d{1,3}'

PATRONES_INTERVIDRIO = {
    "numero":         r"(?i)factura\s+[a-z]\s+(\d{4,5}\s*[-–]\s*\d{5,8})",
    "fecha":          r"(?i)fecha:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "vencimiento":    r"(?i)fecha\s+de\s+vencimiento:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "tipo_factura":   r"(?i)factura\s+([A-Z])\b",
    "cae":            r"(?i)C\.?A\.?E\.?\s*N[°º]?\s*(\d+)",
    "condicion_pago": r"(?i)condici[oóé]n de venta:\s*(.+?)(?:\n|$)",
    "cliente_nombre": None,  # aparece antes de "Dirección:", se toma con extraer_cliente_intervidrio
    "cliente_cuit":   r"(?i)C\.U\.I\.T\.:\s*([\d-]+)",
    "iva_pct":        r'(\d{1,2}[,.]\d{2})\s*%',
}

ITEM_RE_INTERVIDRIO = re.compile(
    r'^(\d+[,.]\d{2})\s*HO\s+'                                    # cantidad de hojas/bultos, ej "2,00HO"
    r'(.+?)\s+'                                                   # descripcion
    r'(\d+[,.]\d{3})\s*[.,]?\s*[xX]\s*(\d+[,.]\d{3})\s+'          # ancho x largo
    r'(' + CANT_PAT_INTERVIDRIO + r')\s+'                          # cantidad m2
    r'(' + NUM_ARG_INTERVIDRIO + r')\s+'                           # precio unit
    r'(' + NUM_ARG_INTERVIDRIO + r')\s*$',                         # importe
    re.IGNORECASE
)

# Líneas de cabecera/pie que jamás son continuación de descripción de un ítem
STOP_RE_INTERVIDRIO = re.compile(
    r'(?i)subtotal|conformidad|percep|impuesto|r[eé]gimen|c\.a\.e|'
    r'condiciones\s+generales|vidrio\s+seguro|alcance:|entrega\s+de|'
    r'redeterminaci|direcci[oó]n\s+comercial|observaciones|cuenta\s+n|'
    r'----pagina----|descripci[oó]n.*ancho|climanet|quanex|\bvasa\b|'
    r'condici[oó]n\s+de\s+(?:venta|pago)|remito'
)

# Fila de totales, ej:
# "1.159.120,71 0,00 1.159.120,71 243.415,35 37.091,86 1.439.627,92"
# columnas: Subtotal | Impuesto | Subtotal | IVA | Regimenes Esp.(percepciones) | Total
TOTALS_ROW_RE_INTERVIDRIO = re.compile(
    r'(' + NUM_ARG_INTERVIDRIO_OCR + r')\s+(' + NUM_ARG_INTERVIDRIO_OCR + r')\s+(' + NUM_ARG_INTERVIDRIO_OCR + r')\s+'
    r'(' + NUM_ARG_INTERVIDRIO_OCR + r')\s+(' + NUM_ARG_INTERVIDRIO_OCR + r')\s+(' + NUM_ARG_INTERVIDRIO_OCR + r')'
)


def extraer_items_intervidrio(texto):
    items = []
    current = None
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea:
            continue
        m = ITEM_RE_INTERVIDRIO.match(linea)
        if m:
            current = {
                "codigo":       None,
                "descripcion":  m.group(2).strip(),
                "ancho":        limpiar_numero(m.group(3)),
                "largo":        limpiar_numero(m.group(4)),
                "cantidad":     limpiar_numero(m.group(5)),
                "precio_unit":  limpiar_numero(m.group(6)),
                "subtotalprod": limpiar_numero(m.group(7)),
            }
            items.append(current)
            continue
        if STOP_RE_INTERVIDRIO.search(linea):
            current = None  # a partir de acá no hay más continuaciones de descripcion
            continue
        # linea de continuacion de descripcion (ej "segun dibujo", "Paño fijo")
        if current and re.match(r'^[A-Za-zÁÉÍÓÚÑáéíóúñ.\s]{2,40}$', linea) and len(linea.split()) <= 5:
            current["descripcion"] = (current["descripcion"] + " " + linea).strip()
    return items


def extraer_totales_intervidrio(texto):
    m = TOTALS_ROW_RE_INTERVIDRIO.search(texto)
    if not m:
        return None, None, None, None
    subtotal = limpiar_numero(m.group(1))
    iva      = limpiar_numero(m.group(4))
    pers     = limpiar_numero(m.group(5))
    total    = limpiar_numero(m.group(6))
    return subtotal, iva, pers, total


def extraer_cliente_intervidrio(texto):
    # El cliente aparece en su propia línea justo antes de "Dirección: ... BAHIA BLANCA"
    # (puede traer el número de comprobante pegado al final, ej "Daniel Roque S.R.L. 000186").
    m = re.search(
        r'(?i)\n([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\.\s]+S\.?R\.?L\.?)\s*[\d\s]*\n\s*Direcci[oóé]n:',
        texto
    )
    return m.group(1).strip() if m else None


def parsear_intervidrio(texto):
    subtotal_val, iva_val, pers_val, total = extraer_totales_intervidrio(texto)
    m_iva_pct = re.search(PATRONES_INTERVIDRIO['iva_pct'], texto)
    iva_pct = float(m_iva_pct.group(1).replace(',', '.')) if m_iva_pct else 21.0
    if not total and subtotal_val:
        total = round((subtotal_val or 0) + (iva_val or 0) + (pers_val or 0), 2)
    numero = extraer_campo(texto, PATRONES_INTERVIDRIO['numero'])
    if numero:
        numero = re.sub(r'\s+', '', numero)
    factura = {
        'proveedor':      'intervidrio',
        'numero':         numero,
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_INTERVIDRIO['fecha'])),
        'vencimiento':    normalizar_fecha(extraer_campo(texto, PATRONES_INTERVIDRIO['vencimiento'])),
        'tipo_factura':   extraer_campo(texto, PATRONES_INTERVIDRIO['tipo_factura']),
        'cae':            extraer_campo(texto, PATRONES_INTERVIDRIO['cae']),
        'condicion_pago': (extraer_campo(texto, PATRONES_INTERVIDRIO['condicion_pago']) or '').strip(),
        'cliente_nombre': extraer_cliente_intervidrio(texto),
        'cliente_cuit':   extraer_campo(texto, PATRONES_INTERVIDRIO['cliente_cuit']),
        'subtotal':       subtotal_val or None,
        'iva_pct':        iva_pct,
        'iva':            iva_val or None,
        'pers_IIBB':      pers_val or None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    items = extraer_items_intervidrio(texto)
    return factura, items


# ── Patrones específicos Metalúrgica G.G. ────────────────────────────────────
# OJO: esta factura usa formato numérico "US" (coma = miles, punto = decimales),
# al revés que el resto de los proveedores (que usan formato argentino).
# Ej: "138,705.60"  "277,411.20"
NUM_US_METALURGICA = r'\d{1,3}(?:,\d{3})*\.\d{2}'

PATRONES_METALURGICA = {
    "numero":         r"(?i)([A-Z]\d{5}\s*-\s*\d{8})",
    "fecha":          r"(?i)fecha:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "cae":            r"(?i)C\.?A\.?E\.?\s*:?\s*(\d{10,})",
    "cae_vto":        r"(?i)VTO\.?\s*C\.?A\.?E\.?\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
    "condicion_pago": r"(?i)Cond\.?\s*de\s*Venta:\s*(.+?)(?:\n|$)",
    "cliente_nombre": r"(?i)Cliente:\s*\d+\s*-\s*(.+?)(?:\n|$)",
    "pedido":         r"(?i)Pedido:\s*(\S+)",
}

CUIT_RE_METALURGICA = re.compile(r'(\d{2}-\d{8}-\d{1})')

# Item: CODIGO  CANTIDAD(us)  DESCRIPCION  P.UNITARIO(us)  [BONIF%]  IMPORTE(us)
# Ej: "IST-OCULTO-1  2.00 SISTEMA OCULTO 150 mts C.C. 80  138,705.60  277,411.20"
ITEM_RE_METALURGICA = re.compile(
    r'^(\S+)\s+'                              # codigo
    r'(\d+\.\d{2})\s+'                        # cantidad (formato us: 2.00)
    r'(.+?)\s+'                                # descripcion
    r'(' + NUM_US_METALURGICA + r')\s+'        # precio unitario
    r'(?:(\d+(?:\.\d+)?)\s*%\s+)?'             # bonif % (opcional)
    r'(' + NUM_US_METALURGICA + r')\s*$',      # importe
    re.IGNORECASE
)

STOP_RE_METALURGICA = re.compile(
    r'(?i)subtotal|bonificaci[oó]n|^iva|total\s+pesos|c\.a\.e|son:|pesos|'
    r'c[oó]digo\s+cantidad|responsable\s+inscripto|condici[oó]n\s+de\s+venta'
)


def limpiar_numero_us(s):
    """Limpia números en formato US (coma=miles, punto=decimal), usado sólo por Metalúrgica GG."""
    if not s:
        return None
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extraer_items_metalurgica(texto):
    items = []
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea:
            continue
        if STOP_RE_METALURGICA.search(linea):
            continue
        m = ITEM_RE_METALURGICA.match(linea)
        if not m:
            continue
        items.append({
            "codigo":       m.group(1).strip(),
            "descripcion":  m.group(3).strip(),
            "cantidad":     limpiar_numero_us(m.group(2)),
            "precio_unit":  limpiar_numero_us(m.group(4)),
            "bonif_pct":    float(m.group(5)) if m.group(5) else 0.0,
            "subtotalprod": limpiar_numero_us(m.group(6)),
        })
    return items


def extraer_totales_metalurgica(texto):
    # Puede haber dos líneas "Subtotal" (antes y después de Bonificación);
    # nos quedamos con la última, que es el subtotal neto de bonificación.
    subtotales = re.findall(r'(?i)subtotal\s*:?\s*(' + NUM_US_METALURGICA + r')', texto)
    subtotal_val = limpiar_numero_us(subtotales[-1]) if subtotales else None

    m_iva = re.search(r'(?i)IVA\s+(\d{1,2}(?:[.,]\d+)?)\s*%\s*:?\s*(' + NUM_US_METALURGICA + r')', texto)
    iva_pct = float(m_iva.group(1).replace(',', '.')) if m_iva else 21.0
    iva_val = limpiar_numero_us(m_iva.group(2)) if m_iva else None

    m_total = re.search(r'(?i)Total\s+Pesos\s*:?\s*(' + NUM_US_METALURGICA + r')', texto)
    total = limpiar_numero_us(m_total.group(1)) if m_total else None

    return subtotal_val, iva_pct, iva_val, total


def extraer_cuits_metalurgica(texto):
    # La primera coincidencia suele ser el CUIT propio de Metalúrgica GG,
    # la segunda (o única, si sólo hay una) es el CUIT del cliente.
    cuits = CUIT_RE_METALURGICA.findall(texto)
    if not cuits:
        return None
    return cuits[-1]


def parsear_metalurgica_gg(texto):
    subtotal_val, iva_pct, iva_val, total = extraer_totales_metalurgica(texto)
    numero = extraer_campo(texto, PATRONES_METALURGICA['numero'])
    if numero:
        numero = re.sub(r'\s+', '', numero)
    factura = {
        'proveedor':      'metalurgica_gg',
        'numero':         numero,
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_METALURGICA['fecha'])),
        'tipo_factura':   numero[0].upper() if numero else None,
        'cae':            extraer_campo(texto, PATRONES_METALURGICA['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_METALURGICA['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_METALURGICA['condicion_pago']) or '').strip(),
        'cliente_nombre': (extraer_campo(texto, PATRONES_METALURGICA['cliente_nombre']) or '').strip() or None,
        'cliente_cuit':   extraer_cuits_metalurgica(texto),
        'pedido':         extraer_campo(texto, PATRONES_METALURGICA['pedido']),
        'subtotal':       subtotal_val,
        'iva_pct':        iva_pct,
        'iva':            iva_val,
        'pers_IIBB':      None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    items = extraer_items_metalurgica(texto)
    return factura, items


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

def ocr_texto_multipagina(archivos):
    """
    Recibe una lista de FileStorage (una por hoja/página de la misma factura),
    hace OCR a cada una en orden y devuelve el texto combinado con un
    separador de página, para que el detector de proveedor y los parsers
    de items/totales operen sobre el documento completo.
    """
    partes = []
    for archivo in archivos:
        img = preparar_imagen(archivo)
        texto_pagina = pytesseract.image_to_string(img, config='--psm 6 -l spa+eng')
        partes.append(texto_pagina)
    return '\n\n----PAGINA----\n\n'.join(partes)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post('/ocr')
@app.post('/ocr-preview')
def ocr():
    # Soporta 1 o varias imágenes bajo la misma clave 'imagen' (una factura
    # que viene en varias hojas: el frontend debe hacer
    # formData.append('imagen', archivo) por cada hoja, en orden).
    archivos = request.files.getlist('imagen')
    if not archivos:
        return jsonify({'error': 'No se recibió imagen'}), 400

    texto = ocr_texto_multipagina(archivos)
    app.logger.warning("OCR TEXTO (%d hoja/s):\n%s", len(archivos), texto)

    proveedor = detectar_proveedor(texto)
    app.logger.warning("[OCR] proveedor detectado: %s", proveedor)

    if proveedor == 'placasur':
        factura, items = parsear_placasur(texto)
    elif proveedor == 'bonzini':
        factura, items = parsear_bonzini(texto)
    elif proveedor == 'inomax':
        factura, items = parsear_inomax(texto)
    elif proveedor == 'intervidrio':
        factura, items = parsear_intervidrio(texto)
    elif proveedor == 'metalurgica_gg':
        factura, items = parsear_metalurgica_gg(texto)
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
    elif proveedor == 'inomax':
        factura, items = parsear_inomax(texto)
    elif proveedor == 'intervidrio':
        factura, items = parsear_intervidrio(texto)
    elif proveedor == 'metalurgica_gg':
        factura, items = parsear_metalurgica_gg(texto)
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
