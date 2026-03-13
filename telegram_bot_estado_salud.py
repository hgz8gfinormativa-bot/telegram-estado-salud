import json
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Any

import fitz  # PyMuPDF
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5.4')
MAX_TEXT_CHARS = int(os.getenv('MAX_TEXT_CHARS', '120000'))
AUTHORIZED_CHAT_IDS = {
    int(x.strip()) for x in os.getenv('AUTHORIZED_CHAT_IDS', '').split(',') if x.strip()
}
INSTITUTION_NAME = os.getenv('INSTITUTION_NAME', 'Unidad Médica')
INCLUDE_JSON_FILE = os.getenv('INCLUDE_JSON_FILE', 'true').lower() == 'true'

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError('Falta la variable de entorno TELEGRAM_BOT_TOKEN')
if not OPENAI_API_KEY:
    raise RuntimeError('Falta la variable de entorno OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Eres un asistente de apoyo documental clínico-administrativo. Tu función es elaborar exclusivamente un BORRADOR de estado de salud, sustentado solo en las notas médicas aportadas por la persona usuaria.

Criterios obligatorios:
1. No inventes datos ni completes vacíos con suposiciones.
2. Si un dato no aparece de forma clara, escribe 'No se documenta'.
3. No emitas diagnósticos nuevos, no modifiques tratamiento y no hagas recomendaciones clínicas nuevas.
4. No afirmes gravedad, estabilidad o pronóstico si ello no se sustenta en las notas.
5. Cuando existan varias notas, prioriza la más reciente para el estado actual, pero integra antecedentes relevantes documentados.
6. Si detectas contradicciones, ambigüedad, texto insuficiente o mala calidad documental, consígnalo en observaciones.
7. Usa redacción formal, prudente y objetiva.
8. El texto final debe dejar claro que se trata de un borrador generado a partir de notas aportadas y sujeto a validación médica.
9. Evita datos personales innecesarios; si aparecen, solo transcribe los estrictamente documentados en el expediente textual.
10. No cites normas jurídicas ni hagas análisis legal, salvo que el usuario lo pida de manera expresa en otro flujo.

Debes responder exclusivamente en JSON válido con la estructura exacta indicada.
""".strip()

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tipo_documento": {"type": "string"},
        "institucion": {"type": "string"},
        "identificacion": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nombre": {"type": "string"},
                "nss": {"type": "string"},
                "edad": {"type": "string"},
                "sexo": {"type": "string"}
            },
            "required": ["nombre", "nss", "edad", "sexo"]
        },
        "fecha_referencia": {"type": "string"},
        "fuentes": {"type": "array", "items": {"type": "string"}},
        "resumen_clinico": {"type": "string"},
        "diagnosticos_documentados": {"type": "array", "items": {"type": "string"}},
        "estado_actual": {"type": "string"},
        "tratamiento_actual_documentado": {"type": "array", "items": {"type": "string"}},
        "pronostico_documentado": {"type": "string"},
        "observaciones": {"type": "array", "items": {"type": "string"}},
        "texto_final": {"type": "string"}
    },
    "required": [
        "tipo_documento",
        "institucion",
        "identificacion",
        "fecha_referencia",
        "fuentes",
        "resumen_clinico",
        "diagnosticos_documentados",
        "estado_actual",
        "tratamiento_actual_documentado",
        "pronostico_documentado",
        "observaciones",
        "texto_final"
    ]
}


def is_authorized(chat_id: int) -> bool:
    if not AUTHORIZED_CHAT_IDS:
        return True
    return chat_id in AUTHORIZED_CHAT_IDS



def sanitize_text(text: str) -> str:
    text = text.replace('\x00', ' ')
    text = re.sub(r'\u200b|\ufeff', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()



def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    try:
        pages = []
        for idx, page in enumerate(doc, start=1):
            page_text = page.get_text('text') or ''
            pages.append(f'\n--- PÁGINA {idx} ---\n{page_text}')
        return sanitize_text('\n'.join(pages))
    finally:
        doc.close()



def build_user_prompt(document_text: str) -> str:
    return f"""
Genera un borrador de estado de salud con base en el siguiente contenido textual extraído de PDF de notas médicas.

Institución de referencia: {INSTITUTION_NAME}
Fecha de elaboración automática: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contenido del PDF:
{document_text[:MAX_TEXT_CHARS]}
""".strip()



def generate_estado_salud(document_text: str) -> dict[str, Any]:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": build_user_prompt(document_text)}]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "estado_salud_schema",
                "schema": JSON_SCHEMA,
                "strict": True
            }
        }
    )
    return json.loads(response.output_text)



def markdown_escape(text: str) -> str:
    replacements = {
        '_': '\\_',
        '*': '\\*',
        '[': '\\[',
        ']': '\\]',
        '(': '\\(',
        ')': '\\)',
        '~': '\\~',
        '`': '\\`',
        '>': '\\>',
        '#': '\\#',
        '+': '\\+',
        '-': '\\-',
        '=': '\\=',
        '|': '\\|',
        '{': '\\{',
        '}': '\\}',
        '.': '\\.',
        '!': '\\!'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text



def render_message(data: dict[str, Any]) -> str:
    ident = data.get('identificacion', {}) or {}
    dx = data.get('diagnosticos_documentados', []) or []
    tx = data.get('tratamiento_actual_documentado', []) or []
    obs = data.get('observaciones', []) or []

    def bullet(items: list[str]) -> str:
        if not items:
            return 'No se documenta'
        return '\n'.join(f'• {markdown_escape(str(x))}' for x in items)

    lines = [
        '🏥 *Estado de salud \(borrador para validación médica\)*',
        f"*Institución:* {markdown_escape(str(data.get('institucion', INSTITUTION_NAME)))}",
        f"*Nombre:* {markdown_escape(str(ident.get('nombre', 'No se documenta')))}",
        f"*NSS:* {markdown_escape(str(ident.get('nss', 'No se documenta')))}",
        f"*Edad:* {markdown_escape(str(ident.get('edad', 'No se documenta')))}",
        f"*Sexo:* {markdown_escape(str(ident.get('sexo', 'No se documenta')))}",
        f"*Fecha de referencia:* {markdown_escape(str(data.get('fecha_referencia', 'No se documenta')))}",
        '',
        '*Resumen clínico:*',
        markdown_escape(str(data.get('resumen_clinico', 'No se documenta'))),
        '',
        '*Diagnósticos documentados:*',
        bullet(dx),
        '',
        '*Estado actual:*',
        markdown_escape(str(data.get('estado_actual', 'No se documenta'))),
        '',
        '*Tratamiento actual documentado:*',
        bullet(tx),
        '',
        '*Pronóstico documentado:*',
        markdown_escape(str(data.get('pronostico_documentado', 'No se documenta'))),
        '',
        '*Observaciones:*',
        bullet(obs),
        '',
        '*Texto final sugerido:*',
        markdown_escape(str(data.get('texto_final', 'No se documenta')))
    ]
    return '\n'.join(lines)



def chunk_text(text: str, chunk_size: int = 3900) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    if not is_authorized(update.effective_chat.id):
        await update.message.reply_text('Chat no autorizado para uso de este bot.')
        return

    await update.message.reply_text(
        'Envíame un PDF con notas médicas y generaré un borrador de estado de salud basado únicamente en el contenido del documento.'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    if not is_authorized(update.effective_chat.id):
        await update.message.reply_text('Chat no autorizado para uso de este bot.')
        return

    help_text = (
        'Comandos disponibles:\n'
        '/start \- Iniciar\n'
        '/help \- Ayuda\n\n'
        'Uso:\n'
        '1\. Envía un archivo PDF con notas médicas\.\n'
        '2\. El bot extrae el texto\.\n'
        '3\. El bot devuelve un borrador de estado de salud sujeto a validación médica\.\n\n'
        'Nota: si el PDF es un escaneo sin texto, deberás integrar OCR posteriormente\.'
    )
    await update.message.reply_text(help_text, parse_mode='MarkdownV2')


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return

    if not is_authorized(update.effective_chat.id):
        await update.message.reply_text('Chat no autorizado para uso de este bot.')
        return

    document = update.message.document
    if not document:
        return

    filename = document.file_name or 'archivo'
    mime = document.mime_type or ''

    if mime != 'application/pdf' and not filename.lower().endswith('.pdf'):
        await update.message.reply_text('Por favor envía un archivo PDF.')
        return

    status_msg = await update.message.reply_text('Procesando PDF...')

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        tg_file = await document.get_file()
        pdf_bytes = await tg_file.download_as_bytearray()
        extracted_text = extract_text_from_pdf_bytes(bytes(pdf_bytes))

        if not extracted_text.strip():
            await status_msg.edit_text(
                'No fue posible extraer texto del PDF. Probablemente es un documento escaneado sin OCR.'
            )
            return

        data = generate_estado_salud(extracted_text)
        rendered = render_message(data)
        parts = chunk_text(rendered)

        await status_msg.edit_text(parts[0], parse_mode='MarkdownV2')
        for extra in parts[1:]:
            await update.message.reply_text(extra, parse_mode='MarkdownV2')

        if INCLUDE_JSON_FILE:
            with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as tmp:
                json.dump(data, tmp, ensure_ascii=False, indent=2)
                temp_json_path = tmp.name

            with open(temp_json_path, 'rb') as fh:
                await update.message.reply_document(
                    document=fh,
                    filename='estado_salud.json',
                    caption='Salida estructurada del borrador de estado de salud'
                )

    except Exception as exc:
        logger.exception('Error procesando PDF')
        await status_msg.edit_text(f'Ocurrió un error al procesar el PDF: {exc}')



def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
