# U-Classifier
Este pacote é uma implementação em Python do U-Classifier para dados de alta dimensionalidade, conforme descrito no artigo de Ahmad & Pavlenko (2018).

## Instalação
Para configurar o ambiente e instalar o pacote para desenvolvimento, siga os passos abaixo. Recomenda-se o uso do `uv` para um gerenciamento de dependências rápido e fácil.

### Passo 1: Clone o repositório

Se você ainda não tem o projeto na sua máquina, comece por clonar o repositório:

```
git clone [https://github.com/Stephaniav1901/u_classifier_hd.git](https://github.com/Stephaniav1901/u_classifier_hd.git)
cd u_classifier_hd
```

### Passo 2: Instale o `uv`

Se você ainda não tem o `uv`, instale-o globalmente com pip:

```
pip install uv
```

### Passo 3: Crie e ative um ambiente virtual

Dentro da pasta do projeto, use o `uv` para criar um ambiente virtual isolado. Isso garante que as dependências do projeto não interfiram com outros projetos na sua máquina.

```
uv venv
```

Após a criação, ative o ambiente:

* No macOS/Linux:

```
source .venv/bin/activate
```

* No Windows (PowerShell):

```
.venv\Scripts\Activate.ps1
```

### Passo 4: Instale o pacote e suas dependências

Com o ambiente ativado, instale o pacote `uclassifier`. Este comando lerá o arquivo `pyproject.toml`, instalará todas as dependências listadas (como numpy, pandas, etc.) e o pacote.

```
uv pip install uclassifier
```

## Uso
O diretório examples/ contém notebooks que demonstram como usar o pacote.