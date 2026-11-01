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
    "burgert":   re.compile(r'(?i)burgert|pinturasburgert|30-50203626-9'),
    "ruedamas":   re.compile(r'(?i)ruedamas|rueda\s*mas|30-?62446907-?7'),
    "woodtools":  re.compile(r'(?i)wood\s*tools|33-?68381019-?9'),
    "ciarrocchi": re.compile(r'(?i)ciarrocchi|33-?68808215-?9'),
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
    # Cuando la descripción de un ítem no entra en una línea, PlacaSur la
    # completa en el renglón siguiente, suelto, justo antes de "Lote Nro Lote:".
    # Ej.: "EH35SCP- PLACA DE MONT.EH BISAGRA CLIP 100.00 UN (1) 191.39 124.45 34,98% 12,445.08"
    #      "ALT.2MM"
    #      "Lote Nro Lote: ..."
    # Esta bandera solo queda "abierta" entre el match del ítem y la línea de
    # Lote (o el próximo ítem), así no se pega texto de otras secciones del
    # documento (totales, leyendas, etc.) a la descripción del último ítem.
    esperando_continuacion = False
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea:
            continue
        if LOTE_RE_PLACASUR.match(linea):
            esperando_continuacion = False
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
            esperando_continuacion = True
        elif esperando_continuacion and items:
            items[-1]["descripcion"] = (items[-1]["descripcion"] + ' ' + linea).strip()
            esperando_continuacion = False
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
    "fecha_presup":   r"FECHA[^:\n\d]{0,25}[:\s—\-]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
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
                or extraer_campo(texto, PATRONES_BONZINI['fecha_presup']) \
                or extraer_campo(texto, PATRONES['fecha'])
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


# ── Patrones específicos BURGERT S.A. (BSA - pinturas) ───────────────────────
# OJO: igual que Metalúrgica G.G., esta factura usa formato numérico "US"
# (coma = miles, punto = decimales). El precio unitario a veces viene sin
# separador de miles ("497192.00") y el importe con él ("497,192.00"); el
# patrón admite ambos casos.
NUM_US_BURGERT = r'\d+(?:,\d{3})*\.\d{2}'

PATRONES_BURGERT = {
    "numero":         r"(?i)N[°ºo]?\s*[:.]?\s*(A?\d{4,5}\s*-\s*\d{5,8})",
    "fecha":          r"(?i)FECHA\s+(\d{1,2}/\d{1,2}/\d{4})",
    "tipo_factura":   r'(?i)FACTURA\s+"?([ABCME])"?\s+ELECTR[OÓ]NICA',
    "cae":            r"(?i)C\.?A\.?E\.?\s*N[°º]?\s*:?\s*(\d{10,})",
    "cae_vto":        r"(?i)vto\.?\s*C\.?A\.?E\.?\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
    "condicion_pago": r"(?i)Cond\.?\s*Venta:\s*(.+?)(?:\n|REMITO|$)",
    "remito":         r"(?i)REMITO:\s*(\S+)",
    "cliente_cuit":   r"(?i)Responsable inscripto\s*\n?\s*([\d-]{11,13})",
}

# Item: CANTIDAD  CODIGO  DESCRIPCION  PRECIO(us)  IMPORTE(us)
# Ej: "1.00 CPU5552KL LACA BLANCA SM (50g) 5552 20L 497192.00 497,192.00"
ITEM_RE_BURGERT = re.compile(
    r'^(\d+\.\d{2})\s+'                     # cantidad
    r'(\S+)\s+'                             # codigo
    r'(.+?)\s+'                             # descripcion
    r'(' + NUM_US_BURGERT + r')\s+'         # precio unitario
    r'(' + NUM_US_BURGERT + r')\s*$',       # importe
    re.IGNORECASE
)

STOP_RE_BURGERT = re.compile(
    r'(?i)^(cantidad\s+codigo|subtotal|descuento|i\.?v\.?a\.?|perc\.?\s*i{1,2}b{1,2}|'
    r'total\b|c\.a\.e|remito|cond\.?\s*venta|responsable\s+inscripto)'
)


def extraer_items_burgert(texto):
    items = []
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea or STOP_RE_BURGERT.match(linea):
            continue
        m = ITEM_RE_BURGERT.match(linea)
        if not m:
            continue
        items.append({
            "codigo":       m.group(2).strip(),
            "descripcion":  m.group(3).strip(),
            "cantidad":     limpiar_numero_us(m.group(1)),
            "precio_unit":  limpiar_numero_us(m.group(4)),
            "subtotalprod": limpiar_numero_us(m.group(5)),
        })
    return items


def extraer_totales_burgert(texto):
    # Hay dos líneas "Subtotal" (bruto y neto de descuento); nos quedamos
    # con la última, que es el subtotal neto que se usa para el IVA.
    subtotales = re.findall(r'(?i)subtotal\s*:?\s*(' + NUM_US_BURGERT + r')', texto)
    subtotal_val = limpiar_numero_us(subtotales[-1]) if subtotales else None

    descuento_val = limpiar_numero_us(
        extraer_campo(texto, r"(?i)Descuento\s*:?\s*(" + NUM_US_BURGERT + r")"))
    iva_val = limpiar_numero_us(
        extraer_campo(texto, r"(?i)I\.?V\.?A\.?\s*:?\s*(" + NUM_US_BURGERT + r")"))
    pers_val = limpiar_numero_us(
        extraer_campo(texto, r"(?i)Perc\.?\s*I{1,2}B{1,2}\s*:?\s*(" + NUM_US_BURGERT + r")"))
    total = limpiar_numero_us(
        extraer_campo(texto, r"(?i)\bTOTAL\s+(" + NUM_US_BURGERT + r")"))

    iva_pct = round(iva_val / subtotal_val * 100, 2) if (iva_val and subtotal_val) else 21.0

    return subtotal_val, descuento_val, iva_pct, iva_val, pers_val, total


def parsear_burgert(texto):
    subtotal_val, descuento_val, iva_pct, iva_val, pers_val, total = extraer_totales_burgert(texto)
    numero = extraer_campo(texto, PATRONES_BURGERT['numero'])
    if numero:
        numero = re.sub(r'\s+', '', numero)
    factura = {
        'proveedor':      'burgert',
        'numero':         numero,
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_BURGERT['fecha'])),
        'tipo_factura':   extraer_campo(texto, PATRONES_BURGERT['tipo_factura']),
        'cae':            extraer_campo(texto, PATRONES_BURGERT['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_BURGERT['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_BURGERT['condicion_pago']) or '').strip(),
        'cliente_cuit':   extraer_campo(texto, PATRONES_BURGERT['cliente_cuit']),
        'remito':         extraer_campo(texto, PATRONES_BURGERT['remito']),
        'subtotal':       subtotal_val,
        'descuento':      descuento_val,
        'iva_pct':        iva_pct,
        'iva':            iva_val,
        'pers_IIBB':      pers_val,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    items = extraer_items_burgert(texto)
    return factura, items


# ── Helpers compartidos por los parsers nuevos (Ruedamas / WoodTools / Ciarrocchi) ─
# Prefijos de CUIT válidos (personas jurídicas y físicas). Se restringe el prefijo
# para no confundir un código de cliente de 11 dígitos (ej. "00000001292") con un CUIT.
CUIT_RE_NUEVOS = re.compile(r'\b((?:20|23|24|27|30|33|34)-?\d{8}-?\d)\b')

# Código de comprobante AFIP -> letra de factura
TIPO_POR_COD_AFIP = {'01': 'A', '06': 'B', '11': 'C', '19': 'E', '51': 'M'}


def formatear_cuit(s):
    """Normaliza a NN-NNNNNNNN-N (algunos proveedores lo imprimen sin guiones)."""
    d = re.sub(r'\D', '', s or '')
    return f"{d[:2]}-{d[2:10]}-{d[10]}" if len(d) == 11 else (s or None)


def extraer_cuit_cliente(texto, cuit_emisor):
    """
    Las tres facturas traen dos CUIT (emisor y cliente) y el orden cambia según el
    layout. Se descarta el del emisor (conocido) y se devuelve el primero que quede.
    """
    propio = re.sub(r'\D', '', cuit_emisor)
    for m in CUIT_RE_NUEVOS.finditer(texto):
        if re.sub(r'\D', '', m.group(1)) != propio:
            return formatear_cuit(m.group(1))
    return None


def iva_pct_estandar(iva, base, default=21.0):
    """Deduce la alícuota desde iva/base y la ajusta a 10.5 / 21 / 27 si está cerca."""
    if not iva or not base:
        return default
    pct = iva / base * 100
    for std in (10.5, 21.0, 27.0):
        if abs(pct - std) <= 0.2:
            return std
    return round(pct, 2)


def _validar_totales(nombre, items, subtotal, iva, total):
    """Sólo loguea: ayuda a detectar lecturas erróneas sin cambiar lo que se devuelve."""
    if items and subtotal:
        suma = round(sum((it.get('subtotalprod') or 0) for it in items), 2)
        if abs(suma - subtotal) > 0.05:
            app.logger.warning("[%s] suma de ítems %.2f != subtotal %.2f", nombre, suma, subtotal)
    if subtotal and total is not None:
        esperado = round(subtotal + (iva or 0), 2)
        if abs(esperado - total) > 0.05:
            app.logger.warning("[%s] subtotal+iva %.2f != total %.2f", nombre, esperado, total)


# ── Patrones específicos Metalúrgica Ruedamas (ERP clouderp) ──────────────────
# Formato numérico "US". En los ítems los importes vienen SIN separador de miles
# (18059.62, 152146.02) pero en los totales CON coma de miles ($442,438.63), por
# eso el mismo patrón admite ambos casos. El CUIT del emisor y del cliente vienen
# sin guiones.
NUM_US_RUEDAMAS = r'\d+(?:,\d{3})*\.\d{2}'

PATRONES_RUEDAMAS = {
    "numero":         r"(?i)Factura\s+Venta\s*-\s*(\d{4,5}-\d{8})",
    "fecha":          r"(?i)\bFecha:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "tipo_factura":   r"(?m)^([ABCME])\s+Fecha:",
    "cae":            r"(?i)C\.?A\.?E\.?\s*(\d{10,})",
    "cae_vto":        r"(?i)Venc\.?\s*C\.?A\.?E\.?\s*(\d{1,2}/\d{1,2}/\d{4})",
    "cliente_nombre": r"(?im)Cliente:\s*(.+?)\s*\(\d+\)\s*$",
    "remito":         r"(?i)Remito:\s*(\S+)",
}

# Ítem: CODIGO DESCRIPCION [DESPACHO] CANT PRECIO IVA% BONIF% SUBTOTAL
# Ej: "G38A KIT MR PLUS 2.0 (RUL. BLINDADOS + FRENOS) 13 11703.54 21 0% 152146.02"
ITEM_RE_RUEDAMAS = re.compile(
    r'^(\S+)\s+'                                  # codigo
    r'(.+?)\s+'                                   # descripcion
    r'(\d+(?:\.\d+)?)\s+'                         # cantidad
    r'(' + NUM_US_RUEDAMAS + r')\s+'              # precio unitario
    r'(\d{1,2}(?:\.\d+)?)\s+'                     # % IVA
    r'(\d+(?:\.\d+)?)%\s+'                        # % bonificacion
    r'(' + NUM_US_RUEDAMAS + r')\s*$',            # subtotal del ítem
    re.IGNORECASE
)


def extraer_items_ruedamas(texto):
    items = []
    for linea_raw in texto.split('\n'):
        m = ITEM_RE_RUEDAMAS.match(linea_raw.strip())
        if not m:
            continue
        items.append({
            "codigo":       m.group(1).strip(),
            "descripcion":  m.group(2).strip(),
            "cantidad":     limpiar_numero_us(m.group(3)),
            "precio_unit":  limpiar_numero_us(m.group(4)),
            "bonif_pct":    float(m.group(6)),
            "subtotalprod": limpiar_numero_us(m.group(7)),
        })
    return items


def extraer_totales_ruedamas(texto):
    def _num(patron):
        m = re.search(patron, texto)
        return limpiar_numero_us(m.group(1)) if m else None

    bruto     = _num(r'(?im)^\s*Subtotal\s+\$\s*(' + NUM_US_RUEDAMAS + r')')
    descuento = _num(r'(?im)^\s*Descuento\s+Global\s+\$\s*(' + NUM_US_RUEDAMAS + r')')
    neto      = _num(r'(?im)^\s*Subtotal\s+Neto\s+\$\s*(' + NUM_US_RUEDAMAS + r')')
    iva       = _num(r'(?im)^\s*IVA\s+\$\s*(' + NUM_US_RUEDAMAS + r')')
    total     = _num(r'(?im)^\s*TOTAL\s+\$\s*(' + NUM_US_RUEDAMAS + r')')
    # Percepciones: una línea por jurisdicción (CABA, ARBA, ...)
    percs = re.findall(r'(?im)^\s*Percepci[oó]n\s+IIBB\s+\w+\s+\$\s*(' + NUM_US_RUEDAMAS + r')', texto)
    pers = round(sum(limpiar_numero_us(p) or 0 for p in percs), 2)

    # El IVA se calcula sobre el Subtotal NETO (ya sin el descuento global)
    if neto is None and bruto is not None:
        neto = round(bruto - (descuento or 0), 2)
    return neto, descuento, iva, pers, total


def parsear_ruedamas(texto):
    subtotal, descuento, iva, pers, total = extraer_totales_ruedamas(texto)
    if total is None and subtotal:
        total = round(subtotal + (iva or 0) + pers, 2)
    items = extraer_items_ruedamas(texto)
    _validar_totales('ruedamas', items, (subtotal + (descuento or 0)) if subtotal else None, None, None)
    factura = {
        'proveedor':      'ruedamas',
        'numero':         extraer_campo(texto, PATRONES_RUEDAMAS['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_RUEDAMAS['fecha'])),
        'tipo_factura':   extraer_campo(texto, PATRONES_RUEDAMAS['tipo_factura']),
        'cae':            extraer_campo(texto, PATRONES_RUEDAMAS['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_RUEDAMAS['cae_vto'])),
        'condicion_pago': '',   # la factura no trae condición de venta (sólo un comentario libre)
        'cliente_nombre': (extraer_campo(texto, PATRONES_RUEDAMAS['cliente_nombre']) or '').strip() or None,
        'cliente_cuit':   extraer_cuit_cliente(texto, '30-62446907-7'),
        'remito':         extraer_campo(texto, PATRONES_RUEDAMAS['remito']),
        'subtotal':       subtotal,
        'descuento':      descuento,
        'iva_pct':        iva_pct_estandar(iva, subtotal),
        'iva':            iva,
        'pers_IIBB':      pers or None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    # Con descuento global el IVA va sobre el neto: subtotal + iva + perc == total
    _validar_totales('ruedamas', None, subtotal, (iva or 0) + pers, total)
    return factura, items


# ── Patrones específicos Ciarrocchi Hnos S.A. (PDF de Odoo) ───────────────────
# Formato argentino (punto de miles, coma decimal). El código del artículo va
# entre corchetes al principio de la descripción y la descripción se parte en
# dos líneas: la segunda trae el resto del texto + "Unidades" (la unidad).
NUM_AR_CIARROCCHI = r'\d{1,3}(?:\.\d{3})*,\d{2}'

PATRONES_CIARROCCHI = {
    "numero":         r"(?i)N[uú]mero:\s*(\d{4,5}-\d{8})",
    "fecha":          r"(?m)\bFecha:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "vencimiento":    r"(?i)Fecha de vencimiento:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "tipo_factura":   r"(?m)^([ABCME])\s+FACTURA\b",
    "cae":            r"(?i)\bCAE:\s*(\d{10,})",
    "cae_vto":        r"(?i)Fecha de vencimiento CAE:\s*(\d{1,2}/\d{1,2}/\d{4})",
    "cliente_nombre": r"(?i)Cliente:\s*(.+?)(?:\s+Fecha de vencimiento:|\n|$)",
}

# Ej: "[HBIS70011HAF] Bisagra Cierre Suave Hafele clip codo 0 40,00 3.442,50 IVA 21% $ 137.700,00"
ITEM_RE_CIARROCCHI = re.compile(
    r'^\[([^\]]+)\]\s+'                             # [codigo]
    r'(.+?)\s+'                                     # descripcion (1ra línea)
    r'(' + NUM_AR_CIARROCCHI + r')\s+'              # cantidad
    r'(' + NUM_AR_CIARROCCHI + r')\s+'              # precio unitario
    r'IVA\s+\d{1,2}(?:,\d+)?\s*%\s+'                # % IVA
    r'\$\s*(' + NUM_AR_CIARROCCHI + r')\s*$',       # importe
    re.IGNORECASE
)

STOP_RE_CIARROCCHI = re.compile(r'(?i)^(subtotal|iva\s|total\b|cae\b|fecha\s+de\s+vencimiento|p[aá]gina)')


def extraer_items_ciarrocchi(texto):
    items = []
    current = None
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea:
            continue
        m = ITEM_RE_CIARROCCHI.match(linea)
        if m:
            current = {
                "codigo":       m.group(1).strip(),
                "descripcion":  m.group(2).strip(),
                "cantidad":     limpiar_numero(m.group(3)),
                "precio_unit":  limpiar_numero(m.group(4)),   # puede ser 0,00: se conserva el ítem
                "subtotalprod": limpiar_numero(m.group(5)),
            }
            items.append(current)
            continue
        if STOP_RE_CIARROCCHI.search(linea):
            current = None
            continue
        if current:
            # 2da línea de la descripción: "base 3D Unidades" -> "base 3D"
            resto = re.sub(r'(?i)\s*\bUnidades?\s*$', '', linea).strip()
            if resto:
                current["descripcion"] = (current["descripcion"] + ' ' + resto).strip()
    return items


def extraer_totales_ciarrocchi(texto):
    m_sub = re.search(r'(?im)^Subtotal\s+\$\s*(' + NUM_AR_CIARROCCHI + r')\s*$', texto)
    m_tot = re.search(r'(?im)^Total\s+\$\s*(' + NUM_AR_CIARROCCHI + r')\s*$', texto)
    # Las líneas de ítem también contienen "IVA 21% $ ..." pero nunca al inicio de línea.
    # Puede haber una línea por alícuota (IVA 21%, IVA 10,5%): se suman.
    ivas = re.findall(r'(?im)^IVA\s+(\d{1,2}(?:,\d+)?)\s*%\s+\$\s*(' + NUM_AR_CIARROCCHI + r')\s*$', texto)
    subtotal = limpiar_numero(m_sub.group(1)) if m_sub else None
    total = limpiar_numero(m_tot.group(1)) if m_tot else None
    iva = round(sum(limpiar_numero(i[1]) or 0 for i in ivas), 2) if ivas else None
    if len(ivas) == 1:
        iva_pct = float(ivas[0][0].replace(',', '.'))
    else:
        iva_pct = iva_pct_estandar(iva, subtotal)
    return subtotal, iva_pct, iva, total


def parsear_ciarrocchi(texto):
    subtotal, iva_pct, iva, total = extraer_totales_ciarrocchi(texto)
    if total is None and subtotal:
        total = round(subtotal + (iva or 0), 2)
    items = extraer_items_ciarrocchi(texto)
    _validar_totales('ciarrocchi', items, subtotal, iva, total)
    factura = {
        'proveedor':      'ciarrocchi',
        'numero':         extraer_campo(texto, PATRONES_CIARROCCHI['numero']),
        'fecha':          normalizar_fecha(extraer_campo(texto, PATRONES_CIARROCCHI['fecha'])),
        'vencimiento':    normalizar_fecha(extraer_campo(texto, PATRONES_CIARROCCHI['vencimiento'])),
        'tipo_factura':   extraer_campo(texto, PATRONES_CIARROCCHI['tipo_factura']),
        'cae':            extraer_campo(texto, PATRONES_CIARROCCHI['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_CIARROCCHI['cae_vto'])),
        'condicion_pago': '',   # Odoo no la imprime en este layout
        'cliente_nombre': (extraer_campo(texto, PATRONES_CIARROCCHI['cliente_nombre']) or '').strip() or None,
        'cliente_cuit':   extraer_cuit_cliente(texto, '33-68808215-9'),
        'subtotal':       subtotal,
        'iva_pct':        iva_pct,
        'iva':            iva,
        'pers_IIBB':      None,
        'total':          total,
        'moneda':         'ARS',
        'texto_raw':      texto,
    }
    return factura, items


# ── Patrones específicos WoodTools S.R.L. (foto escaneada → OCR) ──────────────
# Es la única de las tres que llega por Tesseract, así que el parser es tolerante:
#  * La columna "Total" del ítem sale CORTADA en la foto -> no se lee; el importe
#    del ítem se calcula como cantidad × precio con bonificación.
#  * Las columnas "Precio", "Bonif." y "Prec c/bon." se leen y se cruzan entre sí
#    (precio × (1 - bonif/100) == prec c/bon): esa relación permite ubicar las
#    columnas aunque el OCR pierda comas, agregue espacios o meta "|" y "]".
#  * Los códigos salen con O/0, S/5 confundidos ("MCDO8" por "MCD08").
#  * Totales: se lee el subtotal; IVA y total se buscan en el texto cerca del valor
#    esperado y, si el OCR no los leyó, se calculan.
PATRONES_WOODTOOLS = {
    "numero":         r"(?i)\bN\s*[°º*o]?\s*:\s*(\d{4,5}\s*[-–]\s*\d{8})",
    "numero_generico": r"\b(\d{4,5}\s*[-–]\s*\d{8})\b",
    "fecha":          r"(?i)\bFecha\s+(\d{1,2}/\d{1,2}/\d{4})",
    "cod_afip":       r"(?i)C[óo]d\.?\s*n\S{0,2}\s*(\d{2})\b",
    "cae":            r"(?i)C\.?A\.?E\.?\s*:?\s*(\d{10,})",
    "cae_vto":        r"(?i)\bVto\.?\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
    "condicion_pago": r"(?i)Condiciones\s+de\s+Venta:\s*(.+?)(?:\s+Remito|\n|$)",
    "cliente_nombre": r"(?i)Se[ñn]or\s*/\s*es\s+(.+?)\s+C[óo]digo\s+de\s+Cliente",
    "subtotal":       r"(?i)SUBTOTAL\s+\$?\s*(\d{1,3}(?:\.\d{3})+\s?[,.]?\s?\d{2})\b",
    "iva_pct":        r"(10[,.]50|21[,.]00|27[,.]00)\s*%",
}

# Ej OCR: "S MCDO8 MECHA CIEGA DERECHA 0001 Unidad 2.00 24088,50 10,00 21679,65 43.21"
ITEM_RE_WOODTOOLS = re.compile(
    r'^(?:\S{1,2}\s+)?'                                   # basura OCR al margen ("S ")
    r'([A-Z]{2,4}[0-9OSIl]{2,6})\s+'                      # código
    r'(.+?)\s+'                                           # descripción (1ra línea)
    r'[0-9O]{3,4}\s+(?:Unidad(?:es)?|Un\.?)(?=\s|$)\s*'   # u. de medida ("0001 Unidad")
    r'(.*)$'                                              # cant, precio, bonif, prec c/bon, total cortado
)

STOP_RE_WOODTOOLS = re.compile(r'(?i)subtotal|impuestos|percep|transporte|observaciones|total\b|c\.a\.e|concepto\s+en')


def corregir_codigo_woodtools(codigo):
    """Los códigos son 3 letras + dígitos; el OCR confunde O/0, S/5, I/1 en la parte numérica."""
    m = re.match(r'^([A-Z]{3})([0-9OSIl]+)$', codigo)
    if not m:
        return codigo
    return m.group(1) + m.group(2).translate(str.maketrans('OSIl', '0511'))


def _num_woodtools(tok):
    """Convierte un token numérico del OCR. Las columnas de ítem no llevan separador de miles."""
    if re.fullmatch(r'\d{5,}', tok):            # coma perdida: "3099690" -> 30996.90
        tok = tok[:-2] + '.' + tok[-2:]
    tok = tok.replace(',', '.')
    partes = tok.split('.')
    if len(partes) > 2:
        tok = ''.join(partes[:-1]) + '.' + partes[-1]
    try:
        return float(tok)
    except ValueError:
        return None


def _valores_cola_woodtools(cola):
    """Lista de floats (o None si el token no es numérico) de la cola de un ítem."""
    t = re.sub(r'[|\[\]]', ' ', cola)
    t = re.sub(r'(?<=\d)\s+(?=[.,]\s?\d)', '', t)    # "24088 .50"  -> "24088.50"
    t = re.sub(r'(?<=\d[.,])\s+(?=\d)', '', t)       # "53176, 50"  -> "53176,50"
    return [_num_woodtools(tok) if re.fullmatch(r'\d[\d.,]*', tok) else None for tok in t.split()]


def _interpretar_cola_woodtools(cola):
    """
    Devuelve (cantidad, precio_lista, bonif_pct, precio_con_bonif) o None.
    Busca tres valores consecutivos con precio × (1 - bonif/100) == prec c/bon;
    el valor anterior (si es numérico) es la cantidad.
    """
    vals = _valores_cola_woodtools(cola)
    for i in range(len(vals) - 2):
        p, b, pb = vals[i:i + 3]
        if None in (p, b, pb) or p <= 0 or not (0 <= b <= 100):
            continue
        if abs(p * (1 - b / 100) - pb) <= 0.02:
            cant = vals[i - 1] if i >= 1 else None   # None si el OCR no la leyó ("OD" en vez de "1,00")
            return cant, p, b, pb
    # Plan B: posicional, recalculando el precio con bonificación
    nums = [v for v in vals if v is not None]
    if len(nums) >= 4 and 0 <= nums[2] <= 100:
        cant, p, b = nums[0], nums[1], nums[2]
        return cant, p, b, round(p * (1 - b / 100), 2)
    return None


def extraer_items_woodtools(texto):
    items = []
    current = None
    for linea_raw in texto.split('\n'):
        linea = linea_raw.strip()
        if not linea:
            continue
        m = ITEM_RE_WOODTOOLS.match(linea)
        if m:
            datos = _interpretar_cola_woodtools(m.group(3))
            if not datos:
                app.logger.warning("[woodtools] línea de ítem sin interpretar: %r", linea)
                current = None
                continue
            cant, precio, bonif, precio_bon = datos
            current = {
                "codigo":       corregir_codigo_woodtools(m.group(1)),
                "descripcion":  m.group(2).strip(),
                "cantidad":     cant,
                "precio_unit":  precio_bon,   # precio ya bonificado (igual que PlacaSur)
                "bonif_pct":    bonif,
                "subtotalprod": round(cant * precio_bon, 2) if cant is not None else None,
            }
            items.append(current)
            continue
        if STOP_RE_WOODTOOLS.search(linea):
            current = None
            continue
        # 2da línea de la descripción ("Y=8mm L=57.5 CABO 10x20")
        if current and re.search(r'(?i)\bCABO\b|\b[YL]\s*=', linea):
            extra = re.sub(r'^[^A-Za-z0-9]+', '', linea)       # saca basura inicial del OCR
            extra = re.sub(r'^[a-z]\s+(?=[A-Z]\s*=)', '', extra)  # "a Y=15mm" -> "Y=15mm"
            current["descripcion"] = (current["descripcion"] + ' ' + extra).strip()
            current = None
    return items


def _importe_ar_tolerante(s):
    """'559.798 56' / '559.798,56' / '559.798.56' -> 559798.56"""
    m = re.fullmatch(r'(\d{1,3}(?:\.\d{3})+)[,.]?(\d{2})', re.sub(r'\s+', '', s))
    return float(m.group(1).replace('.', '') + '.' + m.group(2)) if m else None


def _buscar_importe_cercano(texto, objetivo, tol):
    """Busca en el texto un importe con formato AR a ±tol del valor esperado."""
    for m in re.finditer(r'\d{1,3}(?:\.\d{3})+\s?[,.]\s?\d{2}\b', texto):
        v = _importe_ar_tolerante(m.group(0))
        if v is not None and abs(v - objetivo) <= tol:
            return v
    return None


def _completar_cantidad_faltante(items, subtotal):
    """Si exactamente un ítem quedó sin cantidad, se despeja desde el subtotal de la factura."""
    faltan = [it for it in items if it['cantidad'] is None]
    if len(faltan) != 1 or not subtotal:
        return
    it = faltan[0]
    suma = sum(round(o['cantidad'] * o['precio_unit'], 2) for o in items if o['cantidad'] is not None)
    cant = (subtotal - suma) / it['precio_unit'] if it['precio_unit'] else None
    if cant and cant >= 1 and abs(cant - round(cant)) < 0.01:
        it['cantidad'] = float(round(cant))
        it['subtotalprod'] = round(it['cantidad'] * it['precio_unit'], 2)
        app.logger.warning("[woodtools] cantidad de %s no leída; despejada del subtotal: %s",
                           it['codigo'], it['cantidad'])
    else:
        app.logger.warning("[woodtools] cantidad de %s no leída y no se pudo despejar", it['codigo'])


def _vencimiento_woodtools(texto, fecha_iso):
    """
    La foto suele salir cortada a la derecha ("Vencimiento 21/08/2"). Si falta parte del
    año se completa con el de la fecha de emisión (o el siguiente si quedaría anterior).
    """
    m = re.search(r'(?i)Vencimiento\s+(\d{1,2})/(\d{1,2})/(\d{0,4})', texto)
    if not m:
        return None
    d, mes, anio = m.groups()
    if len(anio) == 4:
        return normalizar_fecha(f"{d}/{mes}/{anio}")
    if not fecha_iso or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', fecha_iso):
        return None
    a_emision = int(fecha_iso[:4])
    for cand in (a_emision, a_emision + 1):
        if str(cand).startswith(anio):
            iso = f"{cand:04d}-{int(mes):02d}-{int(d):02d}"
            if iso >= fecha_iso:
                return iso
    return None


def parsear_woodtools(texto):
    items = extraer_items_woodtools(texto)

    m_sub = re.search(PATRONES_WOODTOOLS['subtotal'], texto)
    subtotal = _importe_ar_tolerante(m_sub.group(1)) if m_sub else None
    _completar_cantidad_faltante(items, subtotal)
    if not subtotal and items and all(it['subtotalprod'] is not None for it in items):
        subtotal = round(sum(it['subtotalprod'] for it in items), 2)

    m_pct = re.search(PATRONES_WOODTOOLS['iva_pct'], texto)
    iva_pct = float(m_pct.group(1).replace(',', '.')) if m_pct else 21.0

    iva = total = None
    if subtotal:
        esperado_iva = round(subtotal * iva_pct / 100, 2)
        iva = _buscar_importe_cercano(texto, esperado_iva, 1.0)
        if iva is None:
            iva = esperado_iva   # el OCR no leyó la celda de IVA
            app.logger.warning("[woodtools] IVA no leído; calculado %.2f", iva)
        # El total impreso sale cortado en la foto ("$677.356,2"): se busca completo o se calcula.
        total = _buscar_importe_cercano(texto, round(subtotal + iva, 2), 0.5)
        if total is None:
            total = round(subtotal + iva, 2)
    _validar_totales('woodtools', items, subtotal, iva, total)

    numero = (extraer_campo(texto, PATRONES_WOODTOOLS['numero'])
              or extraer_campo(texto, PATRONES_WOODTOOLS['numero_generico']))
    if numero:
        numero = re.sub(r'\s+', '', numero).replace('–', '-')
    fecha = normalizar_fecha(extraer_campo(texto, PATRONES_WOODTOOLS['fecha']))
    cod = extraer_campo(texto, PATRONES_WOODTOOLS['cod_afip'])
    factura = {
        'proveedor':      'woodtools',
        'numero':         numero,
        'fecha':          fecha,
        'vencimiento':    _vencimiento_woodtools(texto, fecha),
        'tipo_factura':   TIPO_POR_COD_AFIP.get(cod) if cod else None,
        'cae':            extraer_campo(texto, PATRONES_WOODTOOLS['cae']),
        'cae_vto':        normalizar_fecha(extraer_campo(texto, PATRONES_WOODTOOLS['cae_vto'])),
        'condicion_pago': (extraer_campo(texto, PATRONES_WOODTOOLS['condicion_pago']) or '').strip(),
        'cliente_nombre': (extraer_campo(texto, PATRONES_WOODTOOLS['cliente_nombre']) or '').strip() or None,
        'cliente_cuit':   extraer_cuit_cliente(texto, '33-68381019-9'),
        'subtotal':       subtotal,
        'iva_pct':        iva_pct,
        'iva':            iva,
        'pers_IIBB':      None,
        'total':          total,
        'moneda':         'ARS',   # la leyenda "1USD = 1515 PESOS" de observaciones no es la moneda del comprobante
        'texto_raw':      texto,
    }
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
    elif proveedor == 'burgert':
        factura, items = parsear_burgert(texto)
    elif proveedor == 'ruedamas':
        factura, items = parsear_ruedamas(texto)
    elif proveedor == 'woodtools':
        factura, items = parsear_woodtools(texto)
    elif proveedor == 'ciarrocchi':
        factura, items = parsear_ciarrocchi(texto)
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
    elif proveedor == 'burgert':
        factura, items = parsear_burgert(texto)
    elif proveedor == 'ruedamas':
        factura, items = parsear_ruedamas(texto)
    elif proveedor == 'woodtools':
        factura, items = parsear_woodtools(texto)
    elif proveedor == 'ciarrocchi':
        factura, items = parsear_ciarrocchi(texto)
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
