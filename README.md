# Projeto Final SME5924 – Processos Dinâmicos em Redes Complexas

Este repositório contém o trabalho final desenvolvido para a disciplina SME5924 – Processos Dinâmicos em Redes Complexas, ministrada na Universidade de São Paulo (USP).

**Alunos:**
*   Hugo Gobato Souto
*   Tatiana Mein

**Professor:**
*   Francisco A. Rodrigues

---

## 📚 Conteúdo do Repositório

Este repositório está organizado para apresentar de forma clara e detalhada todo o desenvolvimento do nosso projeto, desde a fase exploratória e de compreensão até a apresentação dos resultados finais.

### `Final_Report.pdf`

Este documento é o nosso relatório final. Ele apresenta uma descrição detalhada do trabalho realizado, incluindo:

*   **Revisão de Literatura:** Uma análise minuciosa dos métodos de amostragem para a construção de redes ecológicas, com foco nas abordagens existentes e seus desafios.
*   **Metodologia:** Detalhes sobre a abordagem utilizada para recriar e aplicar o modelo bayesiano proposto por Young et al. (2021).
*   **Resultados e Análise:** Apresentação dos resultados obtidos a partir de nossos experimentos, seguidos de uma análise crítica e suas implicações.

Optamos por escrever o relatório em inglês para simular um formato de papercientífico, conforme solicitado pelo professor. Esta escolha visa também aprimorar nossas habilidades de escrita acadêmica na língua predominante para publicações científicas relevantes na área.

### 📁 `Notebooks_Work_In_Progress`

Esta pasta contém os Jupyter Notebooks e outros arquivos que documentam as etapas intermediárias do nosso desenvolvimento. O objetivo é demonstrar o processo iterativo e o esforço dedicado ao projeto, permitindo ao professor acompanhar a evolução do trabalho.

Os principais documentos e etapas aqui incluem:

1.  **Datasets Utilizados:**
    *   `M_PL_019_matrix.csv`: Conjunto de dados de redes de interação planta-polinizador utilizado para a reconstrução do estudo de Young et al. (2021).
    *   `eco-foodweb-baydry.txt` e `eco-foodweb-baywet.txt`: Dados de redes tróficas empíricas de pântanos de ciprestes no sul da Flórida, utilizados para a aplicação do método em um novo contexto.

2.  **Reconstrução do Método Bayesiano de Young et al. (2021) do Zero:**
    *   `Full_code_reconstruction_of_paper.ipynb`: Este notebook é central para a nossa compreensão do modelo. Nele, recriamos o método bayesiano proposto por Young et al. (2021) fórmula por fórmula, passo a passo. A motivação para essa reconstrução minuciosa é garantir um entendimento profundo e completo do modelo proposto. Ao reimplementar cada equação e cada etapa algorítmica, somos capazes de:
        *   **Validar a Compreensão:** Confirmar que interpretamos corretamente as bases teóricas e estatísticas do modelo.
        *   **Identificar Sensibilidades:** Compreender como cada componente do modelo (priors, likelihoods, etc.) influencia os resultados finais.
        *   **Facilitar Adaptações:** A capacidade de construir o modelo do zero nos permite, posteriormente, adaptá-lo e estendê-lo para diferentes tipos de redes ou problemas, como feito na aplicação a redes tróficas.
        *   **Depurar e Otimizar:** Uma reconstrução completa facilita a identificação de possíveis erros ou gargalos de desempenho caso houvesse a necessidade de otimização.

3.  **Reconstrução dos Resultados de Young et al. (2021) com Código dos Autores:**
    *   `M_PL_019.ipynb`, `M_PL_019_v2.ipynb`, e `M_PL_019_v3.ipynb`: Estes notebooks demonstram nossa tentativa de replicar os resultados do artigo original de Young et al. (2021) utilizando um dataset similar e o código original fornecido pelos autores: `plant_pol_inference.py`, `model.bin`, e `model.stan`. Este passo foi crucial para validar nosso ambiente de trabalho e nossa compreensão do uso da implementação original.

4.  **Aplicação do Método de Young et al. (2021) a Redes Tróficas:**
    *   `Foodweb_baydry.ipynb`, `Foodweb_baywet.ipynb`, `Foodweb_baydry_v2.ipynb`, `Foodweb_baywet_v2.ipynb`, `Foodweb_baydry_v3.ipynb`, e `Foodweb_baywet_v3.ipynb`: Nestes notebooks, aplicamos o método de Young et al. (2021), adaptado por nós em `plant_pol_inference_Hugo_Tati.py`, aos datasets de duas redes tróficas empíricas dos pântanos de ciprestes do sul da Flórida (`eco-foodweb-baydry.txt` e `eco-foodweb-baywet.txt`). Esta aplicação visa explorar a generalização do método para diferentes tipos de redes ecológicas.

### 📁 `Notebooks_Final`

Esta pasta contém a versão final e "limpa" dos Jupyter Notebooks, com comentários claros e análises concisas. Estes são os notebooks recomendados para a revisão final do professor, pois representam a culminação do nosso trabalho.

Os principais documentos aqui incluem:

1.  **Datasets Utilizados:**
    *   `M_PL_019_matrix.csv`
    *   `eco-foodweb-baydry.txt` e `eco-foodweb-baywet.txt`

2.  **Reconstrução dos Resultados de Young et al. (2021):**
    *   `M_PL_019_v3.ipynb`: Versão final e refinada da reconstrução dos resultados de Young et al. (2021) utilizando o código original dos autores (`plant_pol_inference.py`, `model.bin`, e `model.stan`).

3.  **Aplicação do Método a Redes Tróficas:**
    *   `Foodweb_baydry_v3.ipynb` e `Foodweb_baywet_v3.ipynb`: Versões finais e refinadas da aplicação do método de Young et al. (2021) aos datasets de redes tróficas, utilizando nossa adaptação do código (`plant_pol_inference_Hugo_Tati.py`).

---

## ⚠️ Observação Importante sobre Modelos Salvos

Os modelos gerados e salvos no formato `.pkl` não estão presentes neste repositório devido ao seu grande volume (aproximadamente 6 Gigabytes no total). No entanto, se o professor desejar acessá-los, teremos prazer em enviá-los via Google Drive. Alternativamente, os modelos podem ser gerados facilmente executando os respectivos Jupyter Notebooks, que incluem o código para treinar e salvar os modelos.

---

Esperamos que este repositório demonstre o esforço, a dedicação e o aprendizado contínuo ao longo do projeto. Agradecemos a oportunidade e o apoio durante o curso.
