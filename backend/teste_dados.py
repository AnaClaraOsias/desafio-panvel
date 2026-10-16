import pandas as pd

# Carregar o ficheiro que está na pasta de dados
caminho_arquivo = '../dados/filiais.parquet'
df_filiais = pd.read_parquet(caminho_arquivo)

print(f"Total de filiais carregadas: {len(df_filiais)}")
print("\nPrimeiras 3 filiais:")
print(df_filiais.head(3))