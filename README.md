# projeto_SPECTRA-PD
Esse projeto visa fazer desenvolver uma interligência artificial capaz de fazer o diagnostico de parkinson utilizando informações genéticas e de imagens, além de avaliar a aplicação de quantização no modelo.


## SNPs

Os SNPs (Single Nucleotide Polymorphisms) são variações em uma única base do DNA entre indivíduos. São o tipo mais comum de varição genética humana.

Cada SNP recebe um identificador, exemplo: rs429358

Uma SNP pode ter dois alelos possíveis:

A/G
C/T

|Genótipo|Número de cópias do alelo alternativo|
|---|---|
|AA|0|
|AG|1|
|GG|2|

O arquivo gerado com o plink2, .raw, apresenta a seguinte estrutura:

|IID|rs356181|rs2736990|rs213202|rs2230288|rs35801418|
|---|---|---|---|---|---|
|Paciente 1|0|1|2|
|Paciente 2|1|0|1|
|Paciente 3|2|2|0|

Onde o cabeçalho das colunas indica os SNPs de interesse e ans linhas, o número de cópias do alelo alternativo.

Esses SNPs são importantes pois estão ligados ao risco aumentado de desenvolvimento de parkinson, ou podem influenciar na suscetiilidade, progressão e resposta a tratamentos.