U-Classifier
Este pacote é uma implementação em Python do U-Classifier para dados de alta dimensionalidade, conforme descrito no artigo de Ahmad & Pavlenko (2018).

Instalação
Para configurar o ambiente e instalar o pacote para desenvolvimento, siga os passos abaixo. Recomenda-se o uso de uv para um gerenciamento de dependências rápido e moderno.

Passo 1: Clone o repositório

Se você ainda não tem o projeto na sua máquina, comece por clonar o repositório:

git clone [https://github.com/Stephaniav1901/uclassifier-project.git](https://github.com/Stephaniav1901/uclassifier-project.git)
cd uclassifier-project

Passo 2: Instale o uv

Se você ainda não tem o uv, instale-o globalmente com pip:

pip install uv

Passo 3: Crie e ative um ambiente virtual

Dentro da pasta do projeto, use o uv para criar um ambiente virtual isolado. Isso garante que as dependências do projeto não interfiram com outros projetos na sua máquina.

uv venv

Após a criação, ative o ambiente:

No macOS/Linux:

source .venv/bin/activate

No Windows (PowerShell):

.venv\Scripts\Activate.ps1

Passo 4: Instale o pacote e suas dependências

Com o ambiente ativado, instale o pacote uclassifier em modo "editável". Este comando lerá o arquivo pyproject.toml, instalará todas as dependências listadas (como numpy, pandas, etc.) e o seu pacote.

uv pip install -e .

O modo editável (-e) é ideal para desenvolvimento, pois qualquer alteração que você fizer no código-fonte será refletida imediatamente ao executar os scripts, sem precisar reinstalar o pacote.

Uso
O diretório examples/ contém scripts que demonstram como usar o pacote. Para executar todas as simulações e a validação com dados reais:

python examples/run_all.py