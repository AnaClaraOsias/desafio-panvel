
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from ferramentas import (
    buscar_filiais,
    detalhes_filial,
    consultar_bulas,
)

load_dotenv()


# 1. Inicializa o modelo
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# 2. Registra as ferramentas
lista_ferramentas = [
    buscar_filiais,
    detalhes_filial,
    consultar_bulas,
]


# 3. Cria o agente
agente = create_agent(
    model=llm,
    tools=lista_ferramentas,
    system_prompt=(
        "Você é o assistente virtual da Panvel. "
        "Responda em português brasileiro, de forma clara e direta. "

        "MEDICAMENTOS: "
        "Use consultar_bulas para responder perguntas sobre medicamentos. "
        "Envie à ferramenta a pergunta completa, preservando o princípio "
        "ativo e o assunto específico. "
        "Baseie as respostas exclusivamente nas informações sustentadas "
        "pelos trechos recuperados. "
        "Não invente efeitos adversos, doses, frequência, gravidade ou duração. "
        "Não misture informações de medicamentos diferentes. "
        "Uma combinação de princípios ativos não deve ser tratada como "
        "um medicamento isolado. "
        "Se a ferramenta informar que não há evidência suficiente, "
        "comunique essa limitação e não tente completar a resposta com "
        "conhecimento geral. "
        "Não apresente informações sobre superdosagem, interrupção do "
        "tratamento ou advertências como efeitos adversos, a menos que "
        "os trechos recuperados sustentem explicitamente essa interpretação. "
        "Informe o nome do arquivo e a página da bula quando disponíveis. "

        "FILIAIS: "
        "Use buscar_filiais para localizar filiais conforme os critérios "
        "informados pelo usuário. "
        "Use detalhes_filial quando o usuário perguntar sobre uma filial "
        "específica e houver um código disponível. "
        "Os resultados refletem somente a base de dados disponível. "
        "Não afirme que uma filial não existe fora dessa base. "
        "Se não houver resultados, informe isso claramente. "

        "USO DAS FERRAMENTAS: "
        "Não repita uma chamada que já retornou informação suficiente. "
        "Se uma consulta não encontrar evidências, informe a limitação "
        "em vez de repetir a mesma consulta ou inventar informações. "
        "Se a ferramenta não retornar dados suficientes, não afirme "
        "que a informação foi confirmada."
    ),
)


# 4. Teste manual: executado somente ao rodar este arquivo diretamente
if __name__ == "__main__":
    pergunta = (
        "Quais são os efeitos colaterais do paracetamol? "
        "Há alguma filial Panvel com atendimento 24 horas em Cascavel?"
    )

    print(f"Você: {pergunta}\n")

    resultado = agente.invoke(
        {
            "messages": [
                {"role": "user", "content": pergunta}
            ]
        },
        config={"recursion_limit": 8},
    )

    # Exibe chamadas de ferramentas e respectivos resultados
    for mensagem in resultado["messages"]:
        if mensagem.type == "human":
            continue

        if mensagem.type == "ai" and mensagem.tool_calls:
            for chamada in mensagem.tool_calls:
                print(f"\nFerramenta: {chamada['name']}")
                print(f"Argumentos: {chamada['args']}")

        elif mensagem.type == "tool":
            print(f"\nResultado de {mensagem.name}:")
            print(mensagem.content)

    # Exibe a resposta final
    resposta_final = resultado["messages"][-1].content
    print(f"\nAssistente Panvel:\n{resposta_final}")
