# Antigravity OpenAI Bridge (AGY Bridge)

Servidor proxy local e assíncrono compatível com a API da OpenAI (`/v1/chat/completions` e `/v1/models`), construído sobre o CLI do Google Antigravity (`agy`).

Permite utilizar modelos do Antigravity (Gemini 3.8 Flash, Gemini 3.1 Pro, Claude Sonnet/Opus via Antigravity) como um provedor customizado (`custom_providers`) dentro de agentes como Hermes Agent, OpenCode, Aider ou qualquer ferramenta compatível com OpenAI.

---

## Principais Funcionalidades

- **Compatibilidade OpenAI**: Endpoint padrão `/v1/chat/completions` com suporte a JSON, streaming e resolução de tool calls.
- **Parser Robusto de Tool Calls**: Reconhece formatos nativos do Hermes, chamadas multi-linha, blocos JSON embutidos e tolerância a falhas de formatação.
- **Sem Dependências Pesadas**: Utiliza a biblioteca padrão do Python (`http.server.ThreadingHTTPServer`), sem necessidade de frameworks pesados.
- **Controle de Concorrência e Timeout**: Fila com semáforo de concorrência (`AGY_MAX_CONCURRENCY`) e timeout configurável (`AGY_TIMEOUT`).
- **Aliases de Modelos**: Mapeia automaticamente nomes comuns (`gemini`, `gemini-flash`, `claude-sonnet`, `claude-opus`) para os nomes suportados pelo CLI `agy`.

---

## Arquitetura

```text
[ Cliente / Agente (ex: Hermes) ]
              │  (HTTP /v1/chat/completions)
              ▼
      [ agy_bridge.py ]
              │  (Subprocess CLI)
              ▼
       [ agy cli exec ]
              │
              ▼
      [ Antigravity API ]
```

---

## Pré-requisitos

- Python 3.10+
- CLI `agy` (Antigravity CLI) instalado e autenticado na máquina.

---

## Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/WellyngtonG/antigravity-openai-bridge.git
cd antigravity-openai-bridge
```

### 2. Execução Direta

```bash
python3 agy_bridge.py
```

Por padrão, o servidor escuta em `http://127.0.0.1:5000`.

### 3. Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `AGY_BRIDGE_HOST` | `127.0.0.1` | Endereço de escuta do servidor |
| `AGY_BRIDGE_PORT` | `5000` | Porta TCP do servidor |
| `AGY_TIMEOUT` | `600` | Tempo limite em segundos por inferência |
| `AGY_MAX_CONCURRENCY` | `3` | Número máximo de chamadas simultâneas |

---

## Executando como Serviço no Linux (systemd)

Um exemplo de arquivo de serviço está disponível em `agy-bridge.service.example`.

Para instalar como serviço de usuário:

```bash
mkdir -p ~/.config/systemd/user/
cp agy-bridge.service.example ~/.config/systemd/user/agy-bridge.service
systemctl --user daemon-reload
systemctl --user enable --now agy-bridge.service
```

Para verificar o status:

```bash
systemctl --user status agy-bridge.service
```

---

## Exemplo de Uso com cURL

```bash
curl -X POST http://127.0.0.1:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "Olá!"}]}'
```

---

## Licença

MIT License
