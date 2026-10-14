#!/usr/bin/env python3
"""
AGY Bridge — OpenAI-Compatible Multi-threaded API Server powered by Antigravity (AGY) CLI.
Configurado para VPS Linux (Ubuntu), porta 5000.
Modelos: Gemini 3.8 Flash High/Medium, Claude Sonnet 4.6, Claude Opus 4.6, etc.
"""

import sys
import os
import json
import time
import uuid
import shutil
import logging
import threading
import subprocess
import tempfile
import queue
import re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

LOG_DIR = os.path.expanduser("~/.hermes/logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "agy_bridge.log")

handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")]
if sys.stdout is not None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
    handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AGY-Bridge] %(message)s",
    handlers=handlers
)
logger = logging.getLogger("agy_bridge")

PORT = int(os.environ.get("AGY_BRIDGE_PORT", 5000))
HOST = os.environ.get("AGY_BRIDGE_HOST", "127.0.0.1")
AGY_TIMEOUT = int(os.environ.get("AGY_TIMEOUT", 600))
AGY_MAX_CONCURRENCY = int(os.environ.get("AGY_MAX_CONCURRENCY", 3))
AGY_CONCURRENCY_SEM = threading.Semaphore(AGY_MAX_CONCURRENCY)

def get_agy_bin():
    candidates = [
        shutil.which("agy"),
        os.path.expanduser("~/.local/bin/agy"),
        "/home/ubuntu/.local/bin/agy",
        "/usr/local/bin/agy",
        "/usr/bin/agy",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return "agy"

AGY_BIN = get_agy_bin()

MODEL_ALIASES = {
    "default": "gemini-3.8-flash-high",
    "gemini": "gemini-3.8-flash-high",
    "gemini-flash": "gemini-3.8-flash-high",
    "gemini-3.8": "gemini-3.8-flash-high",
    "gemini-3.8-flash": "gemini-3.8-flash-high",
    "gemini-3.8-flash-high": "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium": "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low": "gemini-3.8-flash-low",
    "gemini-3.8-pro": "gemini-3.1-pro-high",
    "gemini-3.7": "gemini-3.7-flash-high",
    "gemini-3.7-flash": "gemini-3.7-flash-high",
    "gemini-3.7-flash-high": "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium": "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low": "gemini-3.7-flash-low",
    "gemini-3.7-pro": "gemini-3.1-pro-high",
    "gemini-3.6": "gemini-3.6-flash-high",
    "gemini-3.6-flash": "gemini-3.6-flash-high",
    "gemini-3.6-flash-high": "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium": "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low": "gemini-3.6-flash-low",
    "gemini-pro": "gemini-3.1-pro-high",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "gemini-3.1-pro-high": "gemini-3.1-pro-high",
    "gemini-3.1-pro-low": "gemini-3.1-pro-low",
    # Claude 5.5 (agy models, 09/10/2026). Nomes antigos 4.6 viram aliases p/ não quebrar configs.
    "claude": "claude-sonnet-5-5-high",
    "claude-sonnet": "claude-sonnet-5-5-high",
    "claude-sonnet-5-5": "claude-sonnet-5-5-high",
    "claude-sonnet-5-5-low": "claude-sonnet-5-5-low",
    "claude-sonnet-5-5-medium": "claude-sonnet-5-5-medium",
    "claude-sonnet-5-5-high": "claude-sonnet-5-5-high",
    "claude-opus": "claude-opus-5-5-high",
    "claude-opus-5-5": "claude-opus-5-5-high",
    "claude-opus-5-5-low": "claude-opus-5-5-low",
    "claude-opus-5-5-medium": "claude-opus-5-5-medium",
    "claude-opus-5-5-high": "claude-opus-5-5-high",
    "claude-sonnet-4-6": "claude-sonnet-5-5-high",
    "claude-3.7-sonnet": "claude-sonnet-5-5-high",
    "claude-opus-4-6-thinking": "claude-opus-5-5-high",
    "gpt-oss-120b-medium": "gpt-oss-120b-medium",
    "nemotron-3.5-lightning-free": "gemini-3.8-flash-medium",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "gemini-3.8-flash-medium",
}

AVAILABLE_MODELS = [
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.8-pro",
    "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.7-pro",
    "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-low",
    "claude-sonnet-5-5-low",
    "claude-sonnet-5-5-medium",
    "claude-sonnet-5-5-high",
    "claude-opus-5-5-low",
    "claude-opus-5-5-medium",
    "claude-opus-5-5-high",
    "gpt-oss-120b-medium"
]


def resolve_model(model_name: str) -> str:
    if not model_name:
        return "gemini-3.8-flash-high"
    cleaned = model_name.strip().lower()
    if "/" in cleaned:
        cleaned = cleaned.split("/")[-1]
    return MODEL_ALIASES.get(cleaned, cleaned)


def format_tool_compact(tool: dict) -> str:
    fn = tool.get("function", {})
    name = fn.get("name", "")
    desc = fn.get("description", "").split("\n")[0].strip()
    params = fn.get("parameters", {}).get("properties", {})
    param_list = []
    for k, v in params.items():
        t = v.get("type", "any")
        req = " (required)" if k in fn.get("parameters", {}).get("required", []) else ""
        param_list.append(f"{k}: {t}{req}")
    return f"- {name}({', '.join(param_list)}): {desc}"


SYSTEM_ISOLATION_DIRECTIVE = (
    "[DIRETRIZ DE SISTEMA: Você é estritamente uma API de inferência de linguagem para o assistente Hermes. "
    "NÃO execute ferramentas locais do Antigravity, não procure arquivos no sistema e não execute comandos no terminal do AGY. "
    "Responda diretamente com o texto para o usuário ou com tool_calls no formato JSON solicitado para o Hermes executar. "
    "Ao chamar ferramentas, gere EXCLUSIVAMENTE o JSON estruturado contendo apenas os argumentos válidos da ferramenta (nunca adicione propriedades extras inventadas como 'subcontract').]\n\n"
    "[REGRA CRÍTICA DE FORMATAÇÃO JSON]: O campo 'arguments' de um tool_call DEVE ser um objeto JSON "
    "sintaticamente válido. Isso significa: (1) NUNCA insira quebras de linha reais dentro de uma string "
    "JSON - sempre escape como \\n; (2) se o valor de um argumento (ex: 'command') contiver um script "
    "Python multi-linha com aspas simples ou duplas internas, escape corretamente TODAS as aspas internas "
    "e todas as quebras de linha antes de colocar o script dentro da string JSON - NUNCA cole o script "
    "literalmente com quebras de linha reais; (3) prefira comandos de uma linha ou escreva o script em um "
    "arquivo temporario e depois execute-o, ao inves de embutir scripts multi-linha complexos direto no "
    "JSON; (4) nunca envolva o JSON de tool_calls em blocos markdown de codigo - emita o JSON puro. "
    "Se voce nao tiver certeza de que consegue escapar corretamente um script complexo, prefira dividir "
    "em multiplos comandos mais simples.]\n\n"
    "[DIRETRIZ DE EXECUÇÃO DE TAREFAS]: "
    "Quando o usuário solicitar uma tarefa (como revisar código, analisar projetos, criar arquivos ou rodar automações), "
    "NUNCA encerre seu turno apenas prometendo que 'vai iniciar' ou que 'vai delegar no futuro'. "
    "Execute as ações imediatamente através das ferramentas disponíveis ou entregue a revisão e o veredito completo diretamente agora mesmo.\n\n"
)


def get_adaptive_timeout(prompt: str, base_timeout: int = AGY_TIMEOUT) -> int:
    extra = (len(prompt) // 40000) * 30
    return max(base_timeout, min(900, base_timeout + extra))


def call_agy_cli_stream(prompt: str, model: str, timeout: int | None = None, _allow_transient_retry: bool = True):
    if timeout is None:
        timeout = get_adaptive_timeout(prompt)
    resolved_model = resolve_model(model)

    sem_wait_start = time.time()
    AGY_CONCURRENCY_SEM.acquire()
    sem_wait = time.time() - sem_wait_start
    if sem_wait > 1:
        logger.warning(f"AGY call queued {sem_wait:.1f}s waiting for a free slot (max concurrency={AGY_MAX_CONCURRENCY}).")
    released = False
    def _release_once():
        nonlocal released
        if not released:
            released = True
            AGY_CONCURRENCY_SEM.release()

    logger.info(f"Invoking AGY (stdin stream) with model '{resolved_model}' (prompt len: {len(prompt)} chars, timeout: {timeout}s)...")
    
    cmd = [
        AGY_BIN,
        "--input-format", "text",
        "--output-format", "stream-json",
        "--model", resolved_model,
        "--disable-slash-commands",
        "--dangerously-skip-permissions"
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=tempfile.gettempdir()
    )
    
    try:
        def write_stdin():
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception:
                pass

        writer_t = threading.Thread(target=write_stdin, daemon=True)
        writer_t.start()

        q = queue.Queue()
        def reader():
            try:
                for line in proc.stdout:
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(None)

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        full_response = ""
        last_usage = {}
        last_error = ""
        while True:
            try:
                line = q.get(timeout=timeout)
            except queue.Empty:
                proc.kill()
                _release_once()
                if full_response.strip():
                    logger.warning(
                        f"AGY call timed out (no output for {timeout}s) but {len(full_response)} chars "
                        f"were already accumulated via streaming — attempting partial recovery instead of discarding."
                    )
                    yield ("result", full_response, last_usage)
                    return
                logger.error(f"AGY call timed out (no output for {timeout}s)")
                yield ("error", "AGY execution timed out", {})
                return

            if line is None:
                break

            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                evt_name = event.get("event")
                if evt_name == "step_update":
                    update = event.get("step_update", {})
                    delta = update.get("text_delta", "")
                    if delta:
                        full_response += delta
                        yield ("delta", delta, None)
                elif evt_name == "result":
                    res = event.get("result", {})
                    full_response = res.get("response", "") or full_response
                    last_usage = res.get("usage", {})
                    last_error = res.get("error", "")
                    if not full_response and not last_error and res.get("status") == "ERROR":
                        last_error = "Error status returned by AGY"
                    if full_response:
                        yield ("result", full_response, last_usage)
            except Exception:
                pass

        if (last_error or not full_response) and resolved_model != "gemini-3.8-flash-medium":
            # Erros TRANSITÓRIOS de elegibilidade/verificação da conta Antigravity
            # ("Eligibility check failed", "Verification Required") não significam
            # que o modelo está indisponível — em 25/09 eles apareceram por ~30s e
            # o AGY voltou sozinho. Sem retry, o Hermes desistia do bridge e caía no
            # fallback pago do OpenRouter, que numa conta free tier responde HTTP 402
            # e DERRUBA o cron inteiro. Tentar de novo antes de degradar.
            transient_markers = ("eligibility check failed", "verification required",
                                 "temporarily", "try again", "rate limit")
            err_low = (last_error or "").lower()
            if _allow_transient_retry and any(m in err_low for m in transient_markers):
                for backoff in (3, 8):
                    logger.warning(
                        f"AGY erro transitório em '{resolved_model}': '{last_error[:120]}'. "
                        f"Retry do MESMO modelo em {backoff}s..."
                    )
                    time.sleep(backoff)
                    retry_text = ""
                    retry_usage = {}
                    # _allow_transient_retry=False evita recursão: a chamada de retry
                    # não pode disparar outra rodada de retries.
                    for ev in call_agy_cli_stream(prompt, resolved_model, timeout=timeout,
                                                  _allow_transient_retry=False):
                        if ev[0] == "result":
                            retry_text, retry_usage = ev[1], ev[2]
                        elif ev[0] == "delta":
                            continue
                    if retry_text.strip():
                        logger.info(f"AGY recuperado no retry com '{resolved_model}'.")
                        _release_once()
                        yield ("result", retry_text, retry_usage)
                        return

            logger.warning(f"AGY model '{resolved_model}' error: '{last_error}'. Fallback para gemini-3.8-flash-medium...")
            _release_once()
            yield from call_agy_cli_stream(prompt, "gemini-3.8-flash-medium", timeout=timeout)
            return
                
        if not full_response:
            yield ("result", full_response, last_usage)
    except Exception as e:
        proc.kill()
        logger.error(f"AGY execution error: {e}")
        yield ("error", str(e), {})
    finally:
        _release_once()


# ── Janela de contexto (11/10/2026) ─────────────────────────────────────────
# Antes: dialogue_messages[-12:] (contagem fixa, incluindo tool calls/resultados).
# Após ~6 ferramentas o pedido do usuário saía do prompt e o modelo passava a
# reler arquivos em loop e encerrar com respostas genéricas. Agora: orçamento por
# caracteres, mensagem atual do usuário e resumo de compactação sempre fixados,
# resultados antigos de ferramenta encurtados (head/tail) em vez de descartados.
AGY_HISTORY_BUDGET_CHARS = int(os.environ.get("AGY_HISTORY_BUDGET_CHARS", 120000))
_RECENT_TOOL_FULL = 6          # resultados de ferramenta mais recentes com limite maior
_RECENT_TOOL_CAP = 20000
_OLD_TOOL_CAP = 1500
_OLD_TOOLCALL_CAP = 2000
_OLD_ASSISTANT_CAP = 3000
_COMPACTION_MARKER = "[CONTEXT COMPACTION"


def _head_tail(text: str, cap: int) -> str:
    if not text or len(text) <= cap:
        return text
    head = int(cap * 0.7)
    tail = cap - head
    return (f"{text[:head]}\n[... {len(text) - cap} caracteres omitidos pela bridge ...]\n"
            f"{text[-tail:]}")


def _last_user_index(dialogue_messages: list) -> int:
    for i in range(len(dialogue_messages) - 1, -1, -1):
        role, content = dialogue_messages[i][0], dialogue_messages[i][1]
        if role == "user" and not str(content).lstrip().startswith(_COMPACTION_MARKER):
            return i
    return -1


def _select_dialogue_window(dialogue_messages: list, budget: int = None) -> list:
    """Retorna a lista (role, content, tool_calls, tool_call_id) a ser enviada,
    com marcadores de omissão no lugar dos trechos cortados."""
    budget = AGY_HISTORY_BUDGET_CHARS if budget is None else budget
    n = len(dialogue_messages)
    if n == 0:
        return dialogue_messages

    last_user = _last_user_index(dialogue_messages)
    pinned = set()
    if last_user >= 0:
        pinned.add(last_user)
    for i in range(n - 1, -1, -1):  # só o resumo de compactação mais recente
        r, c = dialogue_messages[i][0], dialogue_messages[i][1]
        if r == "user" and str(c).lstrip().startswith(_COMPACTION_MARKER):
            pinned.add(i)
            break

    # 1) encurta conteúdo conforme recência
    tool_rank = 0
    shaped = [None] * n
    for i in range(n - 1, -1, -1):
        role, content, tool_calls, tcid = dialogue_messages[i]
        content = content or ""
        in_current_turn = last_user >= 0 and i > last_user
        if role == "tool":
            tool_rank += 1
            cap = _RECENT_TOOL_CAP if tool_rank <= _RECENT_TOOL_FULL else _OLD_TOOL_CAP
            content = _head_tail(content, cap)
        elif role == "assistant" and not in_current_turn and i not in pinned:
            content = _head_tail(content, _OLD_ASSISTANT_CAP)
        if tool_calls and tool_rank > _RECENT_TOOL_FULL:
            tc_str = json.dumps(tool_calls, ensure_ascii=False)
            if len(tc_str) > _OLD_TOOLCALL_CAP:
                tool_calls = [{"_resumo_bridge": _head_tail(tc_str, _OLD_TOOLCALL_CAP)}]
        shaped[i] = (role, content, tool_calls, tcid)

    def _size(item):
        r, c, tc, _ = item
        return len(c or "") + (len(json.dumps(tc, ensure_ascii=False)) if tc else 0) + 32

    # 2) fixados entram sempre; o resto, do mais novo para o mais antigo, até o orçamento
    keep = set(pinned)
    used = sum(_size(shaped[i]) for i in pinned)
    for i in range(n - 1, -1, -1):
        if i in keep:
            continue
        s = _size(shaped[i])
        if used + s > budget:
            break  # mantém o histórico contíguo; fixados já estão dentro
        keep.add(i)
        used += s

    out, gap = [], 0
    for i in range(n):
        if i in keep:
            if gap:
                out.append(("system_note", f"[... {gap} mensagens anteriores omitidas pela janela de contexto da bridge ...]", None, None))
                gap = 0
            out.append(shaped[i])
        else:
            gap += 1
    if gap:
        out.append(("system_note", f"[... {gap} mensagens omitidas pela janela de contexto da bridge ...]", None, None))

    logger.info(f"[Context] janela: {len(keep)}/{n} msgs mantidas, ~{used} chars "
                f"(orçamento {budget}), pedido atual idx={last_user}")
    return out


def format_chat_prompt(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> str:
    prompt_parts = [SYSTEM_ISOLATION_DIRECTIVE.strip()]
    
    if tools:
        tool_descs = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "").split("\n")[0].strip()
            params = fn.get("parameters", {}).get("properties", {})
            param_list = []
            for k, v in params.items():
                t = v.get("type", "any")
                req = " (required)" if k in fn.get("parameters", {}).get("required", []) else ""
                param_list.append(f"{k}: {t}{req}")
            tool_descs.append(f"- {name}({', '.join(param_list)}): {desc}")
        if tool_descs:
            prompt_parts.append("AVAILABLE TOOLS:\n" + "\n".join(tool_descs) + '\nTo call tools, output JSON: {"tool_calls": [{"name": "tool_name", "arguments": {...}}]}')

    system_parts = []
    dialogue_messages = []

    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")
        
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
            
        if not content and not tool_calls:
            continue
        if str(content).strip() == "(empty)":
            continue
            
        if role in ("system", "developer"):
            system_parts.append(content)
        else:
            dialogue_messages.append((role, content, tool_calls, tool_call_id))

    if system_parts:
        prompt_parts.append("\n\n".join(system_parts))

    current_request = ""
    _lu = _last_user_index(dialogue_messages)
    if _lu >= 0:
        current_request = str(dialogue_messages[_lu][1]).strip()

    dialogue_messages = _select_dialogue_window(dialogue_messages)

    for role, content, tool_calls, tool_call_id in dialogue_messages:
        if role == "system_note":
            prompt_parts.append(content)
        elif role == "user":
            prompt_parts.append(f"Usuário:\n{content}")
        elif role == "assistant":
            if tool_calls:
                tc_str = json.dumps(tool_calls, ensure_ascii=False)
                if content:
                    prompt_parts.append(f"Assistente:\n{content}\nTool Calls: {tc_str}")
                else:
                    prompt_parts.append(f"Assistente (Tool Calls):\n{tc_str}")
            else:
                prompt_parts.append(f"Assistente:\n{content}")
        elif role == "tool":
            tid = f" (id: {tool_call_id})" if tool_call_id else ""
            prompt_parts.append(f"Resultado da Ferramenta{tid}:\n{content}")
        else:
            prompt_parts.append(f"{role.capitalize()}:\n{content}")

    if current_request:
        prompt_parts.append(
            "[PEDIDO ATUAL DO USUÁRIO — a sua resposta/ações devem atender exatamente a isto; "
            "não troque por relatório de status, auditoria genérica ou outro assunto. "
            "Se o pedido já foi concluído pelas ferramentas acima, relate o resultado real]:\n"
            + _head_tail(current_request, 1500)
        )
    prompt_parts.append(
        "[LEMBRETE FINAL: responda SOMENTE com texto para o usuário ou com o JSON de tool_calls "
        "descrito acima — nunca acione uma ferramenta nativa sua (run_command, write_to_file, etc.), "
        "só o Hermes executa ferramentas aqui.]\n\nAssistente:\n"
        if tools else "Assistente:\n"
    )
    return "\n\n".join(prompt_parts)


def normalize_tool_call(tc: dict) -> dict:
    call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
    call_type = tc.get("type", "function")
    
    if "function" in tc and isinstance(tc["function"], dict):
        fn_name = tc["function"].get("name", "")
        fn_args = tc["function"].get("arguments", "{}")
    else:
        fn_name = tc.get("name", "")
        fn_args = tc.get("arguments", "{}")

    if isinstance(fn_args, dict):
        fn_args = json.dumps(fn_args, ensure_ascii=False)
    elif not isinstance(fn_args, str):
        fn_args = json.dumps(fn_args or {}, ensure_ascii=False)

    return {
        "id": call_id,
        "type": call_type,
        "function": {
            "name": fn_name,
            "arguments": fn_args
        }
    }


def extract_balanced_json(text: str, start_idx: int) -> str | None:
    """A partir de um '{' em start_idx, retorna a substring JSON balanceada
    (respeitando strings/escapes), em vez de confiar em rfind('}') ou regex
    não-guloso — que quebram com arrays/objetos aninhados nos argumentos.

    CORREÇÃO (2026-10-08): a versão anterior só contava profundidade de
    '{'/'}' e ignorava '['/']' por completo. Isso faz um '}' sobrando (ex:
    o modelo gera '...}}}]}' em vez de '...}}]}') ser lido como "fechou o
    objeto externo" bem antes da hora, devolvendo um candidato truncado
    (sem o ']' do array) que nunca é um JSON válido. Agora rastreamos uma
    pilha com o TIPO de cada abertura e só aceitamos um fechamento se ele
    casar com o tipo esperado — um fechamento de tipo errado ou sem par
    sinaliza JSON genuinamente malformado (retorna None) em vez de mentir
    que fechou com sucesso."""
    stack: list[str] = []
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c in '{[':
            stack.append(c)
        elif c in '}]':
            if not stack:
                return None
            expected = '{' if c == '}' else '['
            if stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return text[start_idx:i + 1]
    return None


def extract_balanced_json_lenient(text: str, start_idx: int) -> str | None:
    """Reparo determinístico (sem round-trip ao modelo) para o erro mais comum
    visto em produção: exatamente UM fechamento sobrando/do tipo errado no
    meio do JSON (ex: '...}}}]}' em vez de '...}}]}'). Em vez de abortar como
    extract_balanced_json faz, descarta silenciosamente o fechamento que não
    casa com o topo da pilha e continua — se no final a pilha fechar
    corretamente, devolve o texto reparado (sem aquele caractere) para o
    chamador tentar json.loads nele. Isso resolve o caso real em
    microssegundos em vez dos ~40s por tentativa que o Self-Repair via LLM
    custava (e que, nesse caso específico, nem sequer corrigia o problema
    porque o prompt de reparo presumia erro de escaping, não de chave extra)."""
    stack: list[str] = []
    in_string = False
    escape = False
    out: list[str] = []
    for i in range(start_idx, len(text)):
        c = text[i]
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            out.append(c)
        elif c in '{[':
            stack.append(c)
            out.append(c)
        elif c in '}]':
            expected = '{' if c == '}' else '['
            if stack and stack[-1] == expected:
                stack.pop()
                out.append(c)
                if not stack:
                    return ''.join(out)
            # fechamento sem par / do tipo errado: descarta e segue
        else:
            out.append(c)
    return None


def try_decode_json(text: str) -> Any:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        t = text
        t = re.sub(r'("(?:\\.|[^"\\])*"|\btrue\b|\bfalse\b|\bnull\b|\d+|[\}\]])\s*([a-zA-Z_0-9]+)\s*:', r'\1, "\2":', t)
        t = re.sub(r'("(?:\\.|[^"\\])*"|\btrue\b|\bfalse\b|\bnull\b|\d+|[\}\]])\s*("(?:\\.|[^"\\])*")\s*:', r'\1, \2:', t)
        t = re.sub(r',\s*([\}\]])', r'\1', t)
        return json.loads(t)
    except Exception:
        pass

    return None


def parse_llm_response(raw_text: str) -> tuple[str, list | None]:
    cleaned = raw_text.strip()
    
    if "```json" in cleaned:
        try:
            block = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            parsed = try_decode_json(block)
            if isinstance(parsed, dict):
                if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                    norm_calls = [normalize_tool_call(tc) for tc in parsed["tool_calls"] if isinstance(tc, dict)]
                    if norm_calls:
                        return parsed.get("content", ""), norm_calls
                if "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
                    return "", [normalize_tool_call(parsed)]
                if "content" in parsed and len(parsed) == 1:
                    return parsed["content"], None
        except Exception:
            pass

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            parsed = try_decode_json(cleaned)
            if isinstance(parsed, dict):
                if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                    norm_calls = [normalize_tool_call(tc) for tc in parsed["tool_calls"] if isinstance(tc, dict)]
                    if norm_calls:
                        return parsed.get("content", ""), norm_calls
                if "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
                    return "", [normalize_tool_call(parsed)]
                if "content" in parsed and len(parsed) == 1:
                    return parsed["content"], None
        except Exception:
            pass

    match_start = cleaned.find('"tool_calls"')
    if match_start == -1:
        match_start = cleaned.find("'tool_calls'")
    if match_start != -1:
        brace_start = cleaned.rfind('{', 0, match_start)
        if brace_start != -1:
            candidate = extract_balanced_json(cleaned, brace_start)
            if candidate:
                parsed = try_decode_json(candidate)
                if isinstance(parsed, dict) and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                    norm_calls = [normalize_tool_call(tc) for tc in parsed["tool_calls"] if isinstance(tc, dict)]
                    if norm_calls:
                        content_before = cleaned[:brace_start].strip()
                        return content_before, norm_calls
            # Reparo determinístico (ver extract_balanced_json_lenient): cobre o
            # caso real de produção de um fechamento sobrando/errado no meio do
            # JSON, sem gastar um round-trip de ~40s pedindo pro modelo reemitir.
            lenient_candidate = extract_balanced_json_lenient(cleaned, brace_start)
            if lenient_candidate:
                parsed = try_decode_json(lenient_candidate)
                if isinstance(parsed, dict) and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                    norm_calls = [normalize_tool_call(tc) for tc in parsed["tool_calls"] if isinstance(tc, dict)]
                    if norm_calls:
                        content_before = cleaned[:brace_start].strip()
                        logger.info(
                            "Auto-Repair determinístico: JSON tinha um fechamento "
                            "sobrando/errado, corrigido sem chamar o modelo de novo."
                        )
                        return content_before, norm_calls

    if '"tool_calls"' in cleaned or "'tool_calls'" in cleaned:
        tc_start = cleaned.find('"tool_calls"')
        if tc_start == -1:
            tc_start = cleaned.find("'tool_calls'")
        brace_start = cleaned.rfind('{', 0, tc_start)
        brace_end = cleaned.rfind('}')
        if brace_start != -1 and brace_end > tc_start:
            candidate = cleaned[brace_start:brace_end + 1]
            parsed = try_decode_json(candidate)
            if isinstance(parsed, dict) and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                norm_calls = [normalize_tool_call(tc) for tc in parsed["tool_calls"] if isinstance(tc, dict)]
                if norm_calls:
                    content_before = cleaned[:brace_start].strip()
                    return content_before, norm_calls

    if '"tool_calls"' in cleaned or "'tool_calls'" in cleaned:
        logger.warning(
            f"PARSE_LEAK: texto contém 'tool_calls' mas nenhuma estratégia de parsing "
            f"conseguiu extrair um JSON válido. "
            f"Preview: {cleaned[:300]!r}"
        )

    return cleaned, None


HOLLOW_PROMISE_PATTERN = re.compile(
    r'(subagente.*(?:trabalhando|executando|iniciad|aplicando)|'
    r'vou delegar|vou iniciar a tarefa|aguard(?:e|ando) o subagente|'
    r'te aviso assim que concluir|deleguei para um subagente|'
    r'vou rodar.*e te aviso|vou executar.*e j[aá] volto)',
    re.IGNORECASE
)


def check_and_refine_hollow_promise(prompt: str, content: str, tool_calls: list | None, model: str) -> tuple[str, list | None]:
    if not tool_calls and content and HOLLOW_PROMISE_PATTERN.search(content):
        logger.warning("Supervisor: Promessa vazia detectada. Executando auto-refinamento...")
        refine_prompt = (
            prompt + f"\n\nAssistente:\n{content}\n\n"
            "Sistema (Supervisor):\n"
            "[AVISO: Você anunciou em texto que uma tarefa ou subagente está trabalhando, "
            "mas NENHUMA ferramenta foi acionada. Emita o JSON de tool_calls agora "
            "ou entregue o resultado completo imediatamente.]\n\n"
            "Assistente:\n"
        )
        refined_text = ""
        for evt_type, data, _ in call_agy_cli_stream(refine_prompt, model):
            if evt_type == "result":
                refined_text = data
        if refined_text:
            new_content, new_tool_calls = parse_llm_response(refined_text)
            logger.info("Supervisor: Resposta refinada.")
            return new_content, new_tool_calls
    return content, tool_calls


def check_and_repair_json_leak(prompt: str, content: str, tool_calls: list | None, model: str, max_attempts: int = 2) -> tuple[str, list | None]:
    """Se o texto contém 'tool_calls' mas o parsing falhou (JSON malformado),
    pede ao modelo para reemitir em JSON válido antes de deixar vazar texto cru."""
    attempt = 0
    current_content = content
    while (
        attempt < max_attempts
        and not tool_calls
        and current_content
        and ('"tool_calls"' in current_content or "'tool_calls'" in current_content)
    ):
        attempt += 1
        logger.warning(f"Self-Repair: Tentativa {attempt}/{max_attempts} de corrigir JSON malformado de tool_calls...")
        repair_prompt = (
            prompt + f"\n\nAssistente:\n{current_content}\n\n"
            "Sistema (Supervisor):\n"
            "[ERRO DE FORMATO: A resposta acima contém 'tool_calls' mas o JSON está malformado. "
            "NÃO presuma que o motivo é aspas ou quebra de linha mal escapadas — a causa mais comum "
            "observada é simplesmente CHAVES OU COLCHETES FALTANDO OU SOBRANDO (ex: emitir '}}}]}' "
            "quando o correto é '}}]}' — uma chave de fechamento a mais no meio da estrutura). "
            "Antes de reemitir: CONTE manualmente cada '{', '}', '[' e ']' que você abriu e fechou, "
            "na ordem, e confirme que casam 1 a 1. Reemita EXATAMENTE a mesma intenção de chamada de "
            "ferramenta, mas agora como um único objeto JSON válido em uma linha, com chaves/colchetes "
            "balanceados e todas as aspas internas e quebras de linha corretamente escapadas (\\\" e \\n). "
            "Não use blocos markdown. Se o comando original era muito complexo para montar com segurança, "
            "simplifique-o (ex: quebre em múltiplos comandos mais simples).]\n\n"
            "Assistente:\n"
        )
        repaired_text = ""
        for evt_type, data, _ in call_agy_cli_stream(repair_prompt, model):
            if evt_type == "result":
                repaired_text = data
        if not repaired_text:
            break
        new_content, new_tool_calls = parse_llm_response(repaired_text)
        if new_tool_calls:
            logger.info(f"Self-Repair: JSON corrigido com sucesso na tentativa {attempt}.")
            return new_content, new_tool_calls
        current_content = new_content

    if not tool_calls and current_content and ('"tool_calls"' in current_content or "'tool_calls'" in current_content):
        logger.error(
            f"Self-Repair: Falhou após {max_attempts} tentativas. Vazando como texto cru pro chat. "
            f"Preview: {current_content[:300]!r}"
        )
    return current_content, tool_calls


class AGYBridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

    def _send_json(self, status: int, data: Any):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            logger.debug(f"Failed to send JSON response: {e}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            models_data = [{"id": m, "object": "model", "created": int(time.time()), "owned_by": "antigravity"} for m in AVAILABLE_MODELS]
            self._send_json(200, {"object": "list", "data": models_data})
            return

        if path.startswith("/v1/models/") or path.startswith("/models/"):
            model_id = path.split("/models/")[1]
            self._send_json(200, {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "antigravity"})
            return
            
        if path in ("/health", "/v1/health", ""):
            self._send_json(200, {"status": "ok", "provider": "antigravity", "bridge": "AGY-Bridge 2.0 (Threaded)"})
            return
            
        self._send_json(404, {"error": f"Path '{path}' not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": f"Path '{path}' not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            req_data = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
            return

        model = req_data.get("model", "gemini-3.8-flash-high")
        messages = req_data.get("messages", [])
        tools = req_data.get("tools", [])
        is_stream = req_data.get("stream", False)

        prompt = format_chat_prompt(messages, tools)
        resolved = resolve_model(model)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_ts = int(time.time())

        if is_stream:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                stop_event = threading.Event()
                def keep_alive():
                    while not stop_event.wait(2.0):
                        try:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                        except Exception:
                            break

                ka_thread = threading.Thread(target=keep_alive, daemon=True)
                ka_thread.start()

                full_text = ""
                try:
                    for evt_type, data, usage in call_agy_cli_stream(prompt, model):
                        if evt_type == "delta":
                            full_text += data
                        elif evt_type == "result":
                            full_text = data or full_text
                        elif evt_type == "error":
                            full_text = f"Error: {data}"
                finally:
                    stop_event.set()
                    ka_thread.join(timeout=0.5)

                content, tool_calls = parse_llm_response(full_text)
                content, tool_calls = check_and_refine_hollow_promise(prompt, content, tool_calls, model)
                content, tool_calls = check_and_repair_json_leak(prompt, content, tool_calls, model)

                if tool_calls:
                    tc_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": resolved,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": "tool_calls"}]
                    }
                    self.wfile.write(f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                else:
                    text_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": resolved,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
                    }
                    self.wfile.write(f"data: {json.dumps(text_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            except Exception as e:
                logger.error(f"Stream error: {e}")
            return

        full_text = ""
        usage_info = {}
        for evt_type, data, usage in call_agy_cli_stream(prompt, model):
            if evt_type == "result":
                full_text = data
                usage_info = usage
            elif evt_type == "error":
                self._send_json(500, {"error": data})
                return

        content, tool_calls = parse_llm_response(full_text)
        content, tool_calls = check_and_refine_hollow_promise(prompt, content, tool_calls, model)
        content, tool_calls = check_and_repair_json_leak(prompt, content, tool_calls, model)
        finish_reason = "tool_calls" if tool_calls else "stop"

        response_payload = {
            "id": request_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": resolved,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content if not tool_calls or content else None,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": usage_info.get("input_tokens", 100),
                "completion_tokens": usage_info.get("output_tokens", 50),
                "total_tokens": usage_info.get("total_tokens", 150)
            }
        }

        self._send_json(200, response_payload)


class ThreadedBridgeServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    server_address = (HOST, PORT)
    httpd = ThreadedBridgeServer(server_address, AGYBridgeHandler)
    logger.info(f"🚀 AGY Bridge Threaded Server rodando em http://{HOST}:{PORT}/v1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    main()
