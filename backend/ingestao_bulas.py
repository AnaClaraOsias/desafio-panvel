
import os
import re
import shutil
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


CAMINHO_BULAS = "../dados/corpus_bulas"
CAMINHO_DB = "../dados/chroma_db"


# Princípios ativos presentes no corpus atual.
# A lista é usada para interpretar os nomes dos arquivos,
# não para inferir informações médicas a partir do conteúdo.
PRINCIPIOS_ATIVOS = [
    "pregabalina",
    "metoprolol",
    "ondansetrona",
    "paroxetina",
    "ibuprofeno",
    "celecoxibe",
    "bromazepam",
    "norfloxacino",
    "venlafaxina",
    "propafenona",
    "tramadol",
    "paracetamol",
    "ceftriaxona",
    "losartana",
    "memantina",
    "pantoprazol",
    "levocetirizina",
    "hidroxicloroquina",
    "metilfenidato",
    "flunitrazepam",
]


def normalizar(texto):
    """Normaliza texto para comparar nomes sem diferenças de acentuação."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )
    return texto.lower()


def identificar_principios_ativos(caminho):
    """
    Identifica os princípios ativos encontrados no nome do arquivo.
    Retorna uma lista; não presume que o nome comercial seja
    um princípio ativo.
    """
    nome = normalizar(Path(caminho).stem)

    encontrados = [
        principio
        for principio in PRINCIPIOS_ATIVOS
        if re.search(
            rf"(?<![a-z]){re.escape(normalizar(principio))}(?![a-z])",
            nome,
        )
    ]

    return encontrados


def identificar_secao(texto):
    """
    Identifica uma seção quando há um título reconhecível
    no próprio chunk. Caso contrário, registra como não identificada.
    """
    texto = normalizar(texto)

    secoes = {
        "reacoes_adversas": [
            "reacoes adversas",
            "quais os males que este medicamento pode me causar",
        ],
        "contraindicacoes": [
            "contraindicacoes",
            "quando nao devo usar este medicamento",
        ],
        "posologia": [
            "posologia",
            "como devo usar este medicamento",
        ],
        "advertencias": [
            "advertencias e precaucoes",
            "o que devo saber antes de usar este medicamento",
        ],
        "superdosagem": [
            "superdose",
            "o que fazer se alguem usar uma quantidade maior",
        ],
        "indicacoes": [
            "indicacoes",
            "para que este medicamento e indicado",
        ],
    }

    for secao, titulos in secoes.items():
        if any(titulo in texto for titulo in titulos):
            return secao

    return "nao_identificada"


print("1. Carregando os PDFs das bulas...")
loader = PyPDFDirectoryLoader(CAMINHO_BULAS)
documentos = loader.load()

if not documentos:
    raise RuntimeError(
        f"Nenhum PDF foi carregado de {CAMINHO_BULAS}"
    )

print(f"Foram carregadas {len(documentos)} páginas.")


print("\n2. Adicionando metadados aos documentos...")

for doc in documentos:
    fonte = doc.metadata.get("source", "")
    nome_arquivo = os.path.basename(fonte)
    principios = identificar_principios_ativos(nome_arquivo)

    doc.metadata["nome_arquivo"] = nome_arquivo
    doc.metadata["principios_ativos"] = ",".join(principios)
    doc.metadata["secao"] = identificar_secao(doc.page_content)

    # Não presumimos o nome comercial a partir do arquivo.
    # Ele poderá ser adicionado futuramente com uma fonte confiável.
    doc.metadata["nome_comercial"] = ""


sem_principio = sorted({
    doc.metadata["nome_arquivo"]
    for doc in documentos
    if not doc.metadata["principios_ativos"]
})

if sem_principio:
    print(
        "\nATENÇÃO: não foi possível identificar automaticamente "
        "o princípio ativo destes arquivos:"
    )
    for nome in sem_principio:
        print(f"- {nome}")


print("\n3. Dividindo o texto em chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documentos)

print(f"Foram criados {len(chunks)} chunks.")


# Recalcula a seção para cada chunk.
for chunk in chunks:
    chunk.metadata["secao"] = identificar_secao(chunk.page_content)


print("\n4. Gerando embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


print("\n5. Criando a base vetorial...")

# Evita misturar os documentos antigos com a nova indexação.
# Confirme o caminho antes de executar: esta pasta será apagada.
if os.path.exists(CAMINHO_DB):
    shutil.rmtree(CAMINHO_DB)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CAMINHO_DB,
)

print("\nIndexação concluída com sucesso.")
print(f"PDFs: {len({d.metadata['nome_arquivo'] for d in documentos})}")
print(f"Páginas: {len(documentos)}")
print(f"Chunks: {len(chunks)}")
print(f"Base vetorial: {CAMINHO_DB}")
