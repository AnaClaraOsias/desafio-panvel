Assistente Conversacional Panvel - IA Generativa
Este projeto implementa um assistente conversacional baseado em LLM para a rede de farmácias Panvel. A solução orquestra dados estruturados (cadastro de filiais em formato Parquet) e dados não estruturados (bulas farmacêuticas em PDF) através de um agente inteligente capaz de usar ferramentas (Tool Calling) e recuperar contexto (RAG).

Decisões Arquiteturais e Trade-offs (ADR)
Para garantir segurança clínica, rastreabilidade e eficiência, a arquitetura foi desenhada com base nas seguintes decisões críticas:

1. Segurança Clínica e Prevenção de Alucinações
A área da saúde exige tolerância zero a alucinações. O agente foi desenhado com um system_prompt rígido que atua como um "guardião".

Regra de Isolamento: O assistente recusa-se a cruzar informações de bulas diferentes se o utilizador perguntar sobre múltiplos princípios ativos em simultâneo.

Resposta Baseada em Evidência: Se a ferramenta de RAG não retornar contexto suficiente, o LLM está instruído a comunicar a limitação em vez de tentar adivinhar a dose ou os efeitos adversos.

2. Estratégia de RAG Híbrido (Metadados + Vetores)
Apenas a busca semântica (similaridade de vetores) não é suficiente para bulas, pois medicamentos diferentes podem ter descrições sintáticas semelhantes.

Solução: Implementou-se um filtro rígido (Hard Filter) por metadados na ingestão dos PDFs. A ferramenta consultar_bulas extrai o princípio ativo da pergunta e o tópico de interesse (ex: posologia, reações adversas). A busca vetorial no ChromaDB só ocorre nos chunks que deram match com esses metadados, eliminando o risco de retornar o efeito colateral do medicamento errado.

Embeddings Locais: A escolha do modelo all-MiniLM-L6-v2 da HuggingFace permite gerar embeddings localmente e sem custos, otimizando a latência e a privacidade dos dados.

3. Orquestração com Tool Calling e Streaming
A experiência do utilizador (UX) e a transparência foram priorizadas.

LangChain Agent: O agente decide de forma autónoma quando invocar buscar_filiais, detalhes_filial ou consultar_bulas.

FastAPI + SSE: O backend expõe um endpoint reativo (Server-Sent Events) que envia a resposta token a token, mas também transmite eventos de início e fim das ferramentas. Isto permite que o Frontend mostre ao utilizador exatamente quais as bases de dados que o assistente está a consultar em tempo real.

4. Observabilidade
O backend gera um trace_id único por requisição. No terminal, são registados:

O início e fim da invocação de cada ferramenta.

Os argumentos passados pela IA.

O tempo de execução (latência).

O consumo total de tokens (prompt e completion), fundamental para o cálculo de custos em produção.

Stack Tecnológica
Backend: Python 3.12, FastAPI (Servidor Assíncrono com SSE).

IA & Orquestração: LangChain, Groq API (LLM openai/gpt-oss-120b).

Vector Store & Embeddings: ChromaDB (Local), HuggingFace (all-MiniLM-L6-v2).

Manipulação de Dados: Pandas / PyArrow (Parquet), PyPDF (Documentos).

Frontend: Vanilla HTML/JS/CSS (sem frameworks externos para facilitar a execução local e demonstrar domínio do DOM e de Streams).

Como Executar o Projeto Localmente
Pré-requisitos
Certifique-se de ter o Python instalado e uma chave de API da Groq.