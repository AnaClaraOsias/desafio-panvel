
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agente import agente


# --- 1. CONFIGURAÇÃO ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Assistente Panvel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 2. MODELO DA REQUISIÇÃO ---

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(min_length=1)


# --- 3. FORMATAÇÃO DOS EVENTOS SSE ---

def sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )


# --- 4. ENDPOINT DE CHAT COM STREAMING ---

@app.post("/chat")
async def chat(request: ChatRequest):
    trace_id = str(uuid.uuid4())
    inicio = time.perf_counter()

    async def gerar():
        try:
            logger.info(
                "Início da requisição | trace_id=%s",
                trace_id,
            )

            yield sse("trace", {"trace_id": trace_id})

            async for evento in agente.astream_events(
                {"messages": request.messages},
                config={"recursion_limit": 8},
                version="v2",
            ):
                tipo = evento.get("event", "")
                nome = evento.get("name", "")
                dados = evento.get("data", {})

                if tipo == "on_tool_start":
                    argumentos = dados.get("input", {})

                    logger.info(
                        "Ferramenta iniciada | trace_id=%s | "
                        "ferramenta=%s | argumentos=%s",
                        trace_id,
                        nome,
                        argumentos,
                    )

                    yield sse(
                        "tool_start",
                        {
                            "trace_id": trace_id,
                            "name": nome,
                            "arguments": argumentos,
                        },
                    )

                elif tipo == "on_tool_end":
                    saida = dados.get("output", "")

                    if hasattr(saida, "content"):
                        saida = saida.content

                    if not isinstance(saida, str):
                        saida = json.dumps(
                            saida,
                            ensure_ascii=False,
                            default=str,
                        )

                    logger.info(
                        "Ferramenta finalizada | trace_id=%s | "
                        "ferramenta=%s",
                        trace_id,
                        nome,
                    )

                    yield sse(
                        "tool_end",
                        {
                            "trace_id": trace_id,
                            "name": nome,
                            "result": saida,
                        },
                    )

                elif tipo == "on_chat_model_stream":
                    chunk = dados.get("chunk")
                    conteudo = getattr(chunk, "content", "")

                    if isinstance(conteudo, str) and conteudo:
                        yield sse(
                            "token",
                            {
                                "trace_id": trace_id,
                                "text": conteudo,
                            },
                        )

                    elif isinstance(conteudo, list):
                        for bloco in conteudo:
                            if isinstance(bloco, dict):
                                texto = bloco.get("text", "")

                                if texto:
                                    yield sse(
                                        "token",
                                        {
                                            "trace_id": trace_id,
                                            "text": texto,
                                        },
                                    )
                elif tipo == "on_chat_model_end":
                                    saida_modelo = dados.get("output", {})
                                    uso_tokens = getattr(saida_modelo, "usage_metadata", None)
                                    
                                    if uso_tokens:
                                        logger.info(
                                            "Tokens consumidos | trace_id=%s | prompt=%s | gerados=%s | total=%s",
                                            trace_id,
                                            uso_tokens.get("input_tokens", 0),
                                            uso_tokens.get("output_tokens", 0),
                                            uso_tokens.get("total_tokens", 0),
                                        )
            duracao_ms = round(
                (time.perf_counter() - inicio) * 1000,
                2,
            )

            logger.info(
                "Requisição concluída | trace_id=%s | duracao_ms=%s",
                trace_id,
                duracao_ms,
            )

            yield sse(
                "done",
                {
                    "trace_id": trace_id,
                    "duration_ms": duracao_ms,
                },
            )

        except Exception:
            duracao_ms = round(
                (time.perf_counter() - inicio) * 1000,
                2,
            )

            logger.exception(
                "Erro na requisição | trace_id=%s | duracao_ms=%s",
                trace_id,
                duracao_ms,
            )

            yield sse(
                "error",
                {
                    "trace_id": trace_id,
                    "message": (
                        "Ocorreu um erro ao processar a solicitação. "
                        "Tente novamente."
                    ),
                },
            )

    return StreamingResponse(
        gerar(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --- 5. HEALTH CHECK ---

@app.get("/health")
def health():
    return {"status": "ok"}
