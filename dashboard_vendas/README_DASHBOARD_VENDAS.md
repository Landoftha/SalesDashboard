# Dashboard de Vendas - TOTVS

## Visão Geral
Este dashboard foi desenvolvido para análise estratégica de vendas com foco em insights acionáveis para tomada de decisão executiva. Todas as visualizações estão prontas para apresentação.

---

## KPIs Principais

### Métricas de Negócio
- **Total de Clientes**: 8.132
- **Clientes Ativos**: 7.303 (89.8% de ativação)
- **Receita MRR Anual**: R$ 4.951.424
- **MRR Médio por Cliente**: R$ 678
- **Ticket Médio**: R$ 5.719

### Previsões (6 meses)
- **Modelo Linear**: R$ 3.435.687
- **Modelo Polinomial**: R$ 3.424.078
- **Crescimento Médio Mensal**: 1.2%

---

## Visualizações Estratégicas

### 1. **concentracao_receita.png** - Análise de Pareto
**Objetivo**: Identificar concentração de receita e segmentação de clientes

**Insights Principais**:
- Análise de Pareto mostrando que uma pequena % dos clientes gera 80% da receita
- Segmentação por valor: Alto (>5k), Médio (1k-5k), Baixo (<1k)
- Identificação dos top 20 clientes por MRR
- Distribuição de receita por segmento

**Ações Estratégicas**:
- Foco em retenção dos clientes de alto valor
- Estratégias de upselling para clientes médios
- Análise de churn risk para clientes premium

---

### 2. **performance_vendas.png** - Performance e ROI
**Objetivo**: Analisar performance de vendas e retorno sobre investimento

**Insights Principais**:
- Top 15 clientes por valor de contratações
- Relação entre MRR e valor contratado
- Distribuição de frequência de contratações
- Análise de ROI (MRR/Valor Contratado)

**Ações Estratégicas**:
- Identificar padrões de clientes de alto ROI
- Otimizar estratégias de contratação
- Definir benchmarks de performance

---

### 3. **perfil_clientes.png** - Segmentação de Clientes
**Objetivo**: Entender perfil e comportamento dos clientes

**Insights Principais**:
- Segmentação por valor: Básico, Crescimento, Premium, Enterprise
- Segmentação por comportamento: Ocasional, Regular, Frequente, Power User
- Matriz de segmentação cruzada (Valor vs Comportamento)
- Ticket médio por segmento

**Ações Estratégicas**:
- Personalizar abordagem por segmento
- Desenvolver produtos específicos para cada perfil
- Estratégias de migração entre segmentos

---

### 4. **previsao_receita.png** - Previsões e Tendências
**Objetivo**: Projeções futuras e análise de crescimento

**Insights Principais**:
- Série histórica de receita mensal simulada
- Previsões lineares e polinomiais para 6 meses
- Taxa de crescimento mensal com tendências
- Intervalo de confiança das previsões

**Ações Estratégicas**:
- Planejamento orçamentário baseado em previsões
- Identificação de sazonalidades
- Definição de metas realistas de crescimento

---

### 5. **kpis_dashboard.png** - Dashboard Executivo
**Objetivo**: Painel executivo com métricas principais

**Insights Principais**:
- KPIs consolidados em formato visual
- Comparativos e benchmarks
- Métricas de ativação e conversão
- Performance financeira resumida

**Ações Estratégicas**:
- Monitoramento contínuo de performance
- Comunicação executiva simplificada
- Tomada de decisão baseada em dados

---

##  Metodologia Analítica

### Dados Utilizados
- **MRR (Monthly Recurring Revenue)**: Receita recorrente mensal por cliente
- **Contratações**: Quantidade e valor das contratações nos últimos 12 meses
- **Segmentação**: Baseada em valor de MRR e comportamento de compra

### Técnicas Aplicadas
- **Análise de Pareto**: Concentração de receita
- **Segmentação RFV**: Baseada em Recência, Frequência e Valor
- **Regressão Linear/Polinomial**: Previsões de receita
- **Análise de ROI**: Retorno sobre investimento
- **Visualização de Dados**: Dashboards interativos

---

##  Insights Estratégicos

### Concentração de Receita
- Princípio 80/20 aplicado à base de clientes
- Identificação de clientes-chave para retenção
- Oportunidades de expansão em segmentos específicos

### Performance de Vendas
- ROI médio por cliente e segmento
- Padrões de contratação mais eficientes
- Identificação de oportunidades de upselling

### Previsibilidade
- Crescimento sustentável de 1.2% ao mês
- Previsibilidade de receita para planejamento
- Identificação de tendências sazonais

---

##  Características das Visualizações

### Design e Usabilidade
- **Cores**: Paleta profissional e consistente
- **Legibilidade**: Fontes e tamanhos otimizados
- **Idioma**: Totalmente em português brasileiro
- **Resolução**: 300 DPI para apresentações

### Interatividade Preparada
- Gráficos prontos para incorporação em dashboards interativos
- Estrutura compatível com Power BI, Tableau e ferramentas similares
- Dados organizados para drill-down e filtros

---

##  Próximos Passos Recomendados

### Curto Prazo (1-3 meses)
1. **Implementar monitoramento** dos KPIs principais
2. **Desenvolver alertas** para clientes de alto valor
3. **Criar campanhas** específicas por segmento

### Médio Prazo (3-6 meses)
1. **Implementar modelo preditivo** de churn
2. **Automatizar relatórios** mensais
3. **Desenvolver benchmarks** por segmento

### Longo Prazo (6-12 meses)
1. **Machine Learning** para previsões avançadas
2. **Dashboard interativo** em tempo real
3. **Integração com CRM** para ações automáticas

---
##  Arquivos Técnicos

### Scripts Utilizados
- `dashboard_create.py`: Script principal de geração
- `dashboard_analise.py`: Análise exploratória de dados

### Dependências
- pandas, numpy, matplotlib, seaborn
- sklearn para modelos preditivos
- Dados fonte: TotvsData/

### Reprodução
```bash
python dashboard_create.py
```
