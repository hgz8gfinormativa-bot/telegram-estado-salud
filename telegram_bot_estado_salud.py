import json
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Any

import fitz
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
INSTITUTION_NAME = os.getenv("INSTITUTION_NAME", "Unidad Médica")
INCLUDE_JSON_FILE = os.getenv("INCLUDE_JSON_FILE", "true").lower() == "true"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN")

if not GROQ_API_KEY:
    raise RuntimeError("Falta GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """
Eres un asistente clínico-administrativo.

Debes generar un BORRADOR de estado de salud basado únicamente en el contenido
de notas médicas proporcionadas.

Reglas:

- No inventes información
- Si falta información escribe "No se documenta"
- No generes diagnósticos nuevos
- Usa redacción formal
- El resultado debe ser JSON válido

Estructura JSON obligatoria:

{
"tipo_documento":"",
"institucion":"",
"identificacion":{
"nombre":"",
"nss":"",
"edad":"",
"sexo":""
},
"fecha_referencia":"",
"fuentes":[],
"resumen_clinico":"",
"diagnosticos_documentados":[],
"estado_actual":"",
"tratamiento_actual_documentado":[],
"pronostico_documentado":"",
"observaciones":[],
"texto_final":""
}
"""


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()

    return "\n".join(pages)


def generate_estado_salud(text: str) -> dict:

    prompt = f"""
Notas médicas:

{text[:12000]}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content

    return json.loads(content)


def render_message(data: dict) -> str:

    ident = data.get("identificacion", {})

    return f"""
🏥 Estado de salud (borrador)

Institución: {data.get("institucion", INSTITUTION_NAME)}

Nombre: {ident.get("nombre","No se documenta")}
NSS: {ident.get("nss","No se documenta")}
Edad: {ident.get("edad","No se documenta")}
Sexo: {ident.get("sexo","No se documenta")}

Resumen clínico:
{data.get("resumen_clinico","No se documenta")}

Diagnósticos:
{data.get("diagnosticos_documentados","No se documenta")}

Estado actual:
{data.get("estado_actual","No se documenta")}

Tratamiento:
{data.get("tratamiento_actual_documentado","No se documenta")}

Pronóstico:
{data.get("pronostico_documentado","No se documenta")}

Observaciones:
{data.get("observaciones","No se documenta")}
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "Envíame un PDF con notas médicas para generar un estado de salud."
    )


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):

    document = update.message.document

    if not document:
        return

    await update.message.reply_text("Procesando PDF...")

    try:

        tg_file = await document.get_file()
        pdf_bytes = await tg_file.download_as_bytearray()

        text = extract_text_from_pdf_bytes(bytes(pdf_bytes))

        if not text.strip():
            await update.message.reply_text(
                "No se pudo extraer texto del PDF."
            )
            return

        data = generate_estado_salud(text)

        message = render_message(data)

        await update.message.reply_text(message)

        if INCLUDE_JSON_FILE:

            with tempfile.NamedTemporaryFile(
                "w",
                delete=False,
                suffix=".json",
                encoding="utf-8",
            ) as f:

                json.dump(data, f, ensure_ascii=False, indent=2)

                path = f.name

            with open(path, "rb") as f:

                await update.message.reply_document(
                    document=f,
                    filename="estado_salud.json",
                )

    except Exception as e:

        await update.message.reply_text(
            f"Ocurrió un error al procesar el PDF: {e}"
        )


def main():

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    app.run_polling()


if __name__ == "__main__":
    main()
