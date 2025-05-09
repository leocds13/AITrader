# **Status do Projeto - POC**

## **Passos Concluídos**
1. Criação da pasta `poc`.
2. Criação dos arquivos `requirements.txt`, `collect_data.py` e `backtest.py`.
3. Configuração da virtualenv com o Pipenv.
4. Conversão automática do `requirements.txt` para o `Pipfile`.

---

## **Passos Pendentes**

### **1. Instalar Dependências**
- Instalar as bibliotecas necessárias no ambiente virtual:
  - `ccxt`
  - `pandas`
  - `matplotlib`

### **2. Corrigir Erros no Código**
- **Arquivo `collect_data.py`**:
  - Declarar a variável `exchange` antes de usá-la.

### **3. Testar os Scripts**
- Executar os scripts `collect_data.py` e `backtest.py` após corrigir os erros e validar se estão funcionando corretamente.