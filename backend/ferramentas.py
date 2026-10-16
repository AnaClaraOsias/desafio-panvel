
from pathlib import Path
import unicodedata

import pandas as pd
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --- 1. CONFIGURAÇÃO DOS CAMINHOS ---

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dados"


# --- 2. FERRAMENTAS DE FILIAIS ---

df_filiais = pd.read_parquet(DATA_DIR / "filiais.parquet")


@tool
def buscar_filiais(cidade: str = None, atendimento_24h: bool = None) -> str:
    """Busca filiais da Panvel por cidade e atendimento 24 horas."""
    resultados = df_filiais.copy()

    if cidade:
        resultados = resultados[
            resultados["localidade"].astype(str).str.strip().str.casefold()
            == cidade.strip().casefold()
        ]

    if atendimento_24h is not None:
        filtro_24h = "SIM" if atendimento_24h else "NÃO"
        resultados = resultados[
            resultados["atendimento_24_horas"]
            .astype(str).str.strip().str.upper()
            == filtro_24h
        ]

    if resultados.empty:
        return (
            "Nenhuma filial encontrada com esses critérios na base de dados "
            "disponível. Isso não confirma que não existam outras filiais."
        )

    resposta = [
        f"Encontradas {len(resultados)} filial(is) na base disponível:"
    ]

    for _, linha in resultados.iterrows():
        resposta.append(
            f"- Código: {linha['codigo_filial']} | "
            f"Cidade: {linha['localidade']} | "
            f"UF: {linha.get('uf', 'não informada')} | "
            f"24h: {linha['atendimento_24_horas']}"
        )

    return "\n".join(resposta)


@tool
def detalhes_filial(codigo_filial: str) -> str:
    """Retorna os dados disponíveis de uma filial pelo código."""
    codigo = str(codigo_filial).strip()

    resultados = df_filiais[
        df_filiais["codigo_filial"].astype(str).str.strip() == codigo
    ]

    if resultados.empty:
        return (
            f"Nenhuma filial com o código {codigo} foi encontrada "
            "na base de dados disponível."
        )

    linha = resultados.iloc[0]
    detalhes = [f"Detalhes da filial {codigo}:"]

    for coluna, valor in linha.items():
        nome_coluna = coluna.replace("_", " ")
        detalhes.append(f"- {nome_coluna}: {valor}")

    return "\n".join(detalhes)


# --- 3. CONFIGURAÇÃO DO RAG DE BULAS ---

print("Carregando a base vetorial de bulas...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db_bulas = Chroma(
    persist_directory=str(DATA_DIR / "chroma_db"),
    embedding_function=embeddings,
)


def normalizar(texto: str) -> str:
    """Converte para minúsculas e remove acentos."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )


ATIVOS_CONHECIDOS = [
    "pregabalina", "metoprolol", "ondansetrona", "paroxetina",
    "ibuprofeno", "celecoxibe", "bromazepam", "norfloxacino",
    "venlafaxina", "propafenona", "tramadol", "paracetamol",
    "ceftriaxona", "losartana", "memantina", "pantoprazol",
    "levocetirizina", "hidroxicloroquina", "metilfenidato",
    "flunitrazepam",
]


SECOES_POR_ASSUNTO = {
    "reacoes_adversas": [
        "reacao adversa",
        "reacoes adversas",
        "efeito colateral",
        "efeitos colaterais",
        "efeito indesejado",
        "efeitos indesejados",
    ],
    "contraindicacoes": [
        "contraindicacao",
        "contraindicacoes",
    ],
    "posologia": [
        "posologia",
        "dose",
        "dosagem",
        "como devo usar",
        "modo de uso",
    ],
    "advertencias": [
        "advertencia",
        "advertencias",
        "precaucao",
        "precaucoes",
    ],
}


@tool
def consultar_bulas(pergunta_medicamento: str) -> str:
    """
    Consulta as bulas indexadas, filtrando pelo princípio ativo
    e priorizando trechos classificados na seção solicitada.
    """
    consulta = pergunta_medicamento.strip()
    consulta_normalizada = normalizar(consulta)

    if not consulta:
        return "Informe o medicamento e a dúvida que deseja esclarecer."

    medicamentos = [
        ativo
        for ativo in ATIVOS_CONHECIDOS
        if normalizar(ativo) in consulta_normalizada
    ]

    if not medicamentos:
        return (
            "Não consegui identificar um princípio ativo conhecido na pergunta. "
            "Informe o nome do medicamento para consultar as bulas disponíveis."
        )

    # Esta versão não valida combinações de princípios ativos.
    if len(medicamentos) > 1:
        return (
            f"A pergunta menciona mais de um princípio ativo: "
            f"{', '.join(medicamentos)}. A consulta atual não valida "
            "combinações com segurança. Não vou misturar bulas diferentes. "
            "Faça uma pergunta sobre um único princípio ativo."
        )

    ativo = medicamentos[0]
    filtro = {"principios_ativos": {"$eq": ativo}}

    secao_desejada = None

    for secao, termos in SECOES_POR_ASSUNTO.items():
        if any(
            termo in consulta_normalizada
            for termo in termos
        ):
            secao_desejada = secao
            break

    resultados = db_bulas.similarity_search(
        consulta,
        k=20,
        filter=filtro,
    )

    if not resultados:
        return (
            f"Não encontrei trechos relevantes nas bulas indexadas para "
            f"o princípio ativo {ativo}. Não há evidência suficiente "
            "nos documentos disponíveis para responder com segurança."
        )

    # Não usar trechos de outra seção como se respondessem à pergunta.
    if secao_desejada:
        resultados_secao = [
            doc
            for doc in resultados
            if doc.metadata.get("secao") == secao_desejada
        ]

        if not resultados_secao:
            return (
                f"Encontrei trechos associados a {ativo}, mas nenhum foi "
                f"classificado como seção '{secao_desejada}'. Como a "
                "classificação automática das seções pode ser incompleta, "
                "não há evidência suficiente para responder com segurança."
            )

        resultados = resultados_secao

    # Limita a quantidade de trechos enviados ao agente.
    resultados = resultados[:5]
    partes = []

    for i, doc in enumerate(resultados, start=1):
        fonte = doc.metadata.get(
            "nome_arquivo",
            doc.metadata.get("source", "Fonte não identificada"),
        )
        pagina = doc.metadata.get("page", "não identificada")
        secao = doc.metadata.get("secao", "não_identificada")

        partes.append(
            f"[Trecho {i}]\n"
            f"Princípio ativo indexado: "
            f"{doc.metadata.get('principios_ativos', ativo)}\n"
            f"Fonte: {Path(str(fonte)).name}\n"
            f"Página (índice do PDF, começando em 0): {pagina}\n"
            f"Seção identificada: {secao}\n"
            f"Conteúdo: {doc.page_content.strip()}"
        )

    return "\n\n".join(partes)


# --- 4. TESTES MANUAIS ---

if __name__ == "__main__":
    print("\n--- Teste de busca de filiais ---")
    print(
        buscar_filiais.invoke({
            "cidade": "Cascavel",
            "atendimento_24h": True,
        })
    )

    print("\n--- Teste de detalhes de filial ---")
    print(
        detalhes_filial.invoke({
            "codigo_filial": "1863",
        })
    )

    print("\n--- Teste da ferramenta de bulas ---")
    print(
        consultar_bulas.invoke({
            "pergunta_medicamento":
                "Quais são as reações adversas da pregabalina?"
        })
    )
