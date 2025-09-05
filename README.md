# U-Classifier
Este pacote é uma implementação em Python do U-Classifier para dados de alta dimensionalidade, conforme descrito no artigo de Ahmad & Pavlenko (2018), para fins educativos.

## Instalação
Para configurar o ambiente e instalar o pacote para desenvolvimento, siga os passos abaixo. Recomenda-se o uso do `uv` para um gerenciamento de dependências rápido e fácil.

### Passo 1: Pré-requisitos

Se você ainda não tem o `uv`, instale-o com:

```
pip install uv
```

### Passo 2: Crie e ative um ambiente virtual (Recomendado)

Para evitar conflitos com outros pacotes Python e manter as dependências do projeto isoladas, é altamente recomendado criar um ambiente virtual antes de instalar o pacote.

### 2.1. Crie uma pasta para o seu projeto e navegue até ela:
```
mkdir meu-projeto-analise
cd meu-projeto-analise
```
### 2.2. Crie o ambiente virtual:
Este comando irá criar uma pasta `.venv` no seu diretório com uma instalação limpa do Python.
```
uv venv
```
### 2.3. Ative o ambiente:
Após a criação, ative o ambiente:

* No macOS/Linux:

```
source .venv/bin/activate
```

* No Windows (PowerShell):

```
.venv\Scripts\Activate.ps1
```
Quando o ambiente estiver ativo, você verá `(.venv)` no início do seu prompt de terminal.

### Passo 3: Instale o pacote e suas dependências

Com o ambiente ativado, instale o pacote `uclassifier`. Este comando lerá o arquivo `pyproject.toml`, instalará todas as dependências listadas (como numpy, pandas, etc.) e o pacote.

```
uv pip install git+https://github.com/Stephaniav1901/u_classifier_hd.git
```
Após a instalação, você já pode importar e usar o pacote nos seus scripts e notebooks.

## Uso
O diretório `examples/` contém notebooks que demonstram como usar o pacote para replicar as simulações do artigo e aplicá-lo a dados reais.

### Executando os Exemplos (Jupyter Notebooks)
Para executar os notebooks no diretório `examples/` (por exemplo, no VS Code ou Jupyter Lab) usando o ambiente virtual que você criou, é necessário instalar um pacote adicional que permite ao Jupyter encontrar o seu ambiente:
```
uv pip install ipykernel
```
Após instalar o `ipykernel`, abra o notebook. O seu editor (como o VS Code) deverá permitir que você selecione o interpretador Python do seu ambiente .venv como o "kernel" do notebook.