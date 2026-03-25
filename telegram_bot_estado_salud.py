import html
import json
import logging
import os
import re
import tempfile
import asyncio
from datetime import datetime
from typing import Any, Dict, List

import fitz
from openai import OpenAI
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from docx import Document
from docx.shared import Pt

# =========================
# CONFIGURACIÓN
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

INSTITUTION_NAME = "HGZ/MF 8 DR. GILBERTO FLORES IZQUIERDO"

MAX_CHUNK_CHARS = 12000
MAX_PROMPT_CHARS = 50000
TELEGRAM_MAX_MESSAGE = 4000
MAX_FILE_SIZE_MB = 20

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# =========================
# UTILIDADES
# =========================

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def split_text(text: str) -> List[str]:
    return [text[i:i+MAX_CHUNK_CHARS] for i in range(0, len(text), MAX_CHUNK_CHARS)]

def chat_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt[:MAX_PROMPT_CHARS]},
        ],
    )
    return json.loads(response.choices[0].message.content)

def build_pdf(text: str, path: str):
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    elements = []
    for line in text.split("\n"):
        elements.append(Paragraph(html.escape(line), style))
        elements.append(Spacer(1, 6))

    doc.build(elements)

def build_docx(text: str, path: str):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(path)

def split_message(msg: str):
    return [msg[i:i+TELEGRAM_MAX_MESSAGE] for i in range(0, len(msg), TELEGRAM_MAX_MESSAGE)]

# =========================
# PROCESAMIENTO PESADO
# =========================

SYSTEM_PROMPT = """
Genera un estado de salud en formato JSON.

No inventes datos.
"""

def process_pdf_and_generate(pdf_bytes: bytes) -> Dict[str, Any]:
    text = extract_text_from_pdf_bytes(pdf_bytes)

    if not text.strip():
        return {"_error": "No se pudo leer el PDF"}

    chunks = split_text(text)
    resumenes = []

    for chunk in chunks:
        data = chat_json(SYSTEM_PROMPT, chunk)
        resumenes.append(data)

    texto_final = "\n\n".join([str(x) for x in resumenes])

    return {
        "texto_final": texto_final,
        "tipo_documento": "estado_salud"
    }

# =========================
# ENVÍO
# =========================

async def send_outputs(update, text, base_name):

    for part in split_message(text):
        await update.message.reply_text(part)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        build_pdf(text, f.name)
        await update.message.reply_document(
            document=open(f.name, "rb"),
            filename=f"{base_name}_{timestamp}.pdf"
        )

    # DOCX
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        build_docx(text, f.name)
        await update.message.reply_document(
            document=open(f.name, "rb"),
            filename=f"{base_name}_{timestamp}.docx"
        )

# =========================
# TELEGRAM
# =========================

HELP = """
/estado - Estado de salud
/resumen - Resumen
/cronologia - Cronología
"""

async def start(update, context):
    context.user_data["mode"] = "estado"
    await update.message.reply_text(HELP)

async def set_estado(update, context):
    context.user_data["mode"] = "estado"
    await update.message.reply_text("Modo estado listo.")

async def handle_pdf(update, context):

    document = update.message.document

    if document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text("Archivo muy grande.")
        return

    await update.message.reply_text("Procesando...")

    tg_file = await document.get_file()
    pdf_bytes = await tg_file.download_as_bytearray()

    # 🔴 PROCESO EN SEGUNDO PLANO (CLAVE)
    result = await asyncio.to_thread(
        process_pdf_and_generate,
        bytes(pdf_bytes)
    )

    if "_error" in result:
        await update.message.reply_text(result["_error"])
        return

    await send_outputs(update, result["texto_final"], "estado_salud")

    await update.message.reply_text("✅ Listo.\n\n" + HELP)

# =========================
# MAIN
# =========================

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("estado", set_estado))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    app.run_polling()

if __name__ == "__main__":
    main()
