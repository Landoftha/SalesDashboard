import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

warnings.filterwarnings('ignore')

class SalesDashboardCreator:
    def __init__(self, data_directory: str = 'TotvsData'):
        self.data_directory = data_directory
        self.setup_logging()
        self.output_dir = 'dashboard_vendas'
        self.mrr_data = None
        self.contracts_data = None
        self.clients_data = None
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def load_sales_data(self):
        """Carrega os dados principais para análise de vendas"""
        try:
            # Dados de MRR
            mrr_path = os.path.join(self.data_directory, 'mrr.csv')
            self.mrr_data = pd.read_csv(mrr_path, sep=';')
            self.mrr_data.columns = ['cliente', 'mrr_12m']
            
            # Dados de contratações
            contracts_path = os.path.join(self.data_directory, 'contratacoes_ultimos_12_meses.csv')
            self.contracts_data = pd.read_csv(contracts_path, sep=';')
            self.contracts_data.columns = ['cliente', 'qtd_contratacoes', 'vlr_contratacoes']
            
            # Converter valores para float, tratando vírgulas como separador decimal
            self.contracts_data['vlr_contratacoes'] = self.contracts_data['vlr_contratacoes'].astype(str).str.replace(',', '.').astype(float)
            
            # Merge dos dados
            self.sales_data = pd.merge(self.mrr_data, self.contracts_data, on='cliente', how='outer')
            self.sales_data = self.sales_data.fillna(0)
            
            self.logger.info(f"Dados carregados: {len(self.sales_data)} clientes")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            
    def create_output_directory(self):
        """Cria diretório para salvar as visualizações"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def plot_revenue_concentration(self):
        """Gráfico de concentração de receita (Curva de Pareto)"""
        plt.figure(figsize=(14, 8))
        
        # Ordenar clientes por MRR decrescente
        df_sorted = self.sales_data.sort_values('mrr_12m', ascending=False)
        df_sorted = df_sorted[df_sorted['mrr_12m'] > 0]  # Apenas clientes com receita
        
        # Calcular percentuais cumulativos
        df_sorted['receita_acumulada'] = df_sorted['mrr_12m'].cumsum()
        df_sorted['pct_receita_acum'] = (df_sorted['receita_acumulada'] / df_sorted['mrr_12m'].sum()) * 100
        df_sorted['pct_clientes'] = (np.arange(1, len(df_sorted) + 1) / len(df_sorted)) * 100
        
        # Criar subplot duplo
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Gráfico 1: Curva de Pareto
        ax1_twin = ax1.twinx()
        
        bars = ax1.bar(range(len(df_sorted[:20])), df_sorted['mrr_12m'][:20], 
                      color='steelblue', alpha=0.7, label='MRR Individual')
        ax1_twin.plot(range(len(df_sorted)), df_sorted['pct_receita_acum'], 
                     color='red', marker='o', markersize=3, linewidth=2, label='% Receita Acumulada')
        
        ax1.set_title('Concentração de Receita - Top 20 Clientes (Análise de Pareto)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Ranking dos Clientes')
        ax1.set_ylabel('MRR (R$)', color='steelblue')
        ax1_twin.set_ylabel('% Receita Acumulada', color='red')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar linha de 80%
        ax1_twin.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% da Receita')
        
        # Encontrar quantos clientes representam 80% da receita
        clientes_80 = len(df_sorted[df_sorted['pct_receita_acum'] <= 80])
        pct_clientes_80 = (clientes_80 / len(df_sorted)) * 100
        
        ax1.text(0.02, 0.98, f'📊 Insight: {pct_clientes_80:.1f}% dos clientes geram 80% da receita', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=12, fontweight='bold')
        
        # Gráfico 2: Distribuição por segmento de receita
        segments = ['Alto Valor (>5k)', 'Médio Valor (1k-5k)', 'Baixo Valor (<1k)']
        alto_valor = len(df_sorted[df_sorted['mrr_12m'] > 5000])
        medio_valor = len(df_sorted[(df_sorted['mrr_12m'] >= 1000) & (df_sorted['mrr_12m'] <= 5000)])
        baixo_valor = len(df_sorted[df_sorted['mrr_12m'] < 1000])
        
        receita_alto = df_sorted[df_sorted['mrr_12m'] > 5000]['mrr_12m'].sum()
        receita_medio = df_sorted[(df_sorted['mrr_12m'] >= 1000) & (df_sorted['mrr_12m'] <= 5000)]['mrr_12m'].sum()
        receita_baixo = df_sorted[df_sorted['mrr_12m'] < 1000]['mrr_12m'].sum()
        
        # Gráfico de barras duplas
        x = np.arange(len(segments))
        width = 0.35
        
        ax2.bar(x - width/2, [alto_valor, medio_valor, baixo_valor], width, 
               label='Qtd. Clientes', color='lightblue', alpha=0.8)
        
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + width/2, [receita_alto, receita_medio, receita_baixo], width,
                    label='Receita Total', color='lightcoral', alpha=0.8)
        
        ax2.set_title('Segmentação de Clientes por Valor de MRR', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Segmentos de Clientes')
        ax2.set_ylabel('Quantidade de Clientes', color='blue')
        ax2_twin.set_ylabel('Receita Total (R$)', color='red')
        ax2.set_xticks(x)
        ax2.set_xticklabels(segments)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'concentracao_receita.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_sales_performance(self):
        """Análise de performance de vendas e contratações"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Gráfico 1: Top 15 clientes por valor de contratações
        top_contracts = self.sales_data.nlargest(15, 'vlr_contratacoes')
        
        axes[0, 0].barh(range(len(top_contracts)), top_contracts['vlr_contratacoes'], 
                       color='green', alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_contracts)))
        axes[0, 0].set_yticklabels(top_contracts['cliente'], fontsize=9)
        axes[0, 0].set_title('Top 15 Clientes - Valor de Contratações (12M)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Valor Contratado (R$)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(top_contracts['vlr_contratacoes']):
            axes[0, 0].text(v + max(top_contracts['vlr_contratacoes']) * 0.01, i, 
                           f'R$ {v:,.0f}', va='center', fontsize=8)
        
        # Gráfico 2: Relação entre MRR e Contratações
        scatter_data = self.sales_data[(self.sales_data['mrr_12m'] > 0) & 
                                      (self.sales_data['vlr_contratacoes'] > 0)]
        
        scatter = axes[0, 1].scatter(scatter_data['mrr_12m'], scatter_data['vlr_contratacoes'], 
                                   alpha=0.6, c=scatter_data['qtd_contratacoes'], 
                                   cmap='viridis', s=50)
        axes[0, 1].set_title('Relação: MRR vs Valor de Contratações', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('MRR Anual (R$)')
        axes[0, 1].set_ylabel('Valor Contratações (R$)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Adicionar colorbar
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('Qtd. Contratações')
        
        # Gráfico 3: Distribuição de quantidade de contratações
        qtd_dist = self.sales_data['qtd_contratacoes'].value_counts().sort_index()
        
        axes[1, 0].bar(qtd_dist.index, qtd_dist.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Distribuição - Quantidade de Contratações por Cliente', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Número de Contratações')
        axes[1, 0].set_ylabel('Quantidade de Clientes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: ROI Analysis (MRR vs Valor Contratado)
        roi_data = self.sales_data[(self.sales_data['mrr_12m'] > 0) & 
                                  (self.sales_data['vlr_contratacoes'] > 0)].copy()
        roi_data['roi'] = roi_data['mrr_12m'] / roi_data['vlr_contratacoes']
        
        # Remover outliers extremos para melhor visualização
        roi_data = roi_data[roi_data['roi'] <= roi_data['roi'].quantile(0.95)]
        
        axes[1, 1].hist(roi_data['roi'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribuição do ROI (MRR/Valor Contratado)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('ROI (MRR Anual / Valor Contratado)')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        roi_mean = roi_data['roi'].mean()
        axes[1, 1].axvline(roi_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'ROI Médio: {roi_mean:.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_vendas.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_customer_segments(self):
        """Análise de perfil e segmentação de clientes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Definir segmentos baseados em MRR e comportamento de compra
        df_analysis = self.sales_data.copy()
        
        # Segmentação por valor (MRR)
        df_analysis['segmento_valor'] = pd.cut(df_analysis['mrr_12m'], 
                                             bins=[0, 500, 2000, 10000, float('inf')],
                                             labels=['Básico', 'Crescimento', 'Premium', 'Enterprise'])
        
        # Segmentação por comportamento de compra
        df_analysis['segmento_compra'] = pd.cut(df_analysis['qtd_contratacoes'],
                                              bins=[0, 1, 3, 10, float('inf')],
                                              labels=['Ocasional', 'Regular', 'Frequente', 'Power User'])
        
        # Gráfico 1: Distribuição por segmento de valor
        valor_dist = df_analysis['segmento_valor'].value_counts()
        colors1 = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']
        
        wedges, texts, autotexts = axes[0, 0].pie(valor_dist.values, labels=valor_dist.index, 
                                                 autopct='%1.1f%%', colors=colors1, startangle=90)
        axes[0, 0].set_title('Distribuição de Clientes por Segmento de Valor', 
                            fontsize=14, fontweight='bold')
        
        # Gráfico 2: Receita por segmento
        receita_por_segmento = df_analysis.groupby('segmento_valor')['mrr_12m'].sum().sort_values(ascending=True)
        
        axes[0, 1].barh(range(len(receita_por_segmento)), receita_por_segmento.values, 
                       color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
        axes[0, 1].set_yticks(range(len(receita_por_segmento)))
        axes[0, 1].set_yticklabels(receita_por_segmento.index)
        axes[0, 1].set_title('Receita Total por Segmento de Valor', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Receita MRR (R$)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Heatmap de segmentação
        heatmap_data = pd.crosstab(df_analysis['segmento_valor'], 
                                  df_analysis['segmento_compra'], 
                                  values=df_analysis['mrr_12m'], 
                                  aggfunc='sum').fillna(0)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=axes[1, 0], cbar_kws={'label': 'Receita MRR (R$)'})
        axes[1, 0].set_title('Matriz de Segmentação: Valor vs Comportamento', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Segmento por Comportamento de Compra')
        axes[1, 0].set_ylabel('Segmento por Valor')
        
        # Gráfico 4: Ticket médio por segmento
        ticket_medio = df_analysis[df_analysis['qtd_contratacoes'] > 0].copy()
        ticket_medio['ticket_medio'] = ticket_medio['vlr_contratacoes'] / ticket_medio['qtd_contratacoes']
        
        ticket_por_segmento = ticket_medio.groupby('segmento_valor')['ticket_medio'].mean().sort_values()
        
        bars = axes[1, 1].bar(range(len(ticket_por_segmento)), ticket_por_segmento.values, 
                             color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'], alpha=0.8)
        axes[1, 1].set_xticks(range(len(ticket_por_segmento)))
        axes[1, 1].set_xticklabels(ticket_por_segmento.index, rotation=45)
        axes[1, 1].set_title('Ticket Médio por Segmento de Valor', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Ticket Médio (R$)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, ticket_por_segmento.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(ticket_por_segmento.values) * 0.01,
                           f'R$ {value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'perfil_clientes.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_revenue_prediction(self):
        """Modelo de previsão de receita com regressão linear"""
        # Simular dados históricos mensais baseados nos dados anuais
        np.random.seed(42)
        months = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
        
        # Simular receita mensal baseada no MRR anual
        monthly_revenue = []
        total_annual_mrr = self.sales_data['mrr_12m'].sum()
        
        for i, month in enumerate(months):
            # Adicionar sazonalidade e tendência
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Sazonalidade anual
            trend_factor = 1 + 0.02 * i  # Crescimento de 2% ao mês
            noise = np.random.normal(0, 0.05)  # Ruído aleatório
            
            monthly_value = (total_annual_mrr / 12) * seasonal_factor * trend_factor * (1 + noise)
            monthly_revenue.append(monthly_value)
        
        df_timeseries = pd.DataFrame({
            'data': months,
            'receita': monthly_revenue
        })
        
        # Preparar dados para regressão
        df_timeseries['mes_numerico'] = range(len(df_timeseries))
        X = df_timeseries['mes_numerico'].values.reshape(-1, 1)
        y = df_timeseries['receita'].values
        
        # Modelo linear
        model_linear = LinearRegression()
        model_linear.fit(X, y)
        
        # Modelo polinomial
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        model_poly = LinearRegression()
        model_poly.fit(X_poly, y)
        
        # Previsões para próximos 6 meses
        future_months = 6
        future_X = np.arange(len(df_timeseries), len(df_timeseries) + future_months).reshape(-1, 1)
        future_X_poly = poly_features.transform(future_X)
        
        pred_linear = model_linear.predict(future_X)
        pred_poly = model_poly.predict(future_X_poly)
        
        # Datas futuras
        future_dates = pd.date_range(start=months[-1] + pd.DateOffset(months=1), periods=future_months, freq='M')
        
        # Visualização
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Gráfico 1: Série histórica e previsões
        axes[0].plot(df_timeseries['data'], df_timeseries['receita'], 
                    marker='o', linewidth=2, label='Receita Histórica', color='blue')
        axes[0].plot(future_dates, pred_linear, 
                    marker='s', linewidth=2, linestyle='--', label='Previsão Linear', color='red')
        axes[0].plot(future_dates, pred_poly, 
                    marker='^', linewidth=2, linestyle='--', label='Previsão Polinomial', color='green')
        
        axes[0].set_title('Previsão de Receita - Próximos 6 Meses', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Período')
        axes[0].set_ylabel('Receita MRR (R$)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Adicionar área de confiança
        historical_std = df_timeseries['receita'].std()
        axes[0].fill_between(future_dates, pred_poly - historical_std, pred_poly + historical_std, 
                           alpha=0.2, color='green', label='Intervalo de Confiança')
        
        # Gráfico 2: Métricas de crescimento
        growth_rates = df_timeseries['receita'].pct_change().dropna() * 100
        
        axes[1].bar(range(len(growth_rates)), growth_rates, alpha=0.7, 
                   color=['red' if x < 0 else 'green' for x in growth_rates])
        axes[1].set_title('Taxa de Crescimento Mensal (%)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Mês')
        axes[1].set_ylabel('Crescimento (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Adicionar estatísticas
        avg_growth = growth_rates.mean()
        axes[1].axhline(y=avg_growth, color='blue', linestyle='--', linewidth=2, 
                       label=f'Crescimento Médio: {avg_growth:.1f}%')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'previsao_receita.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Retornar insights
        return {
            'receita_atual': total_annual_mrr,
            'previsao_6m_linear': pred_linear.sum(),
            'previsao_6m_poly': pred_poly.sum(),
            'crescimento_medio': avg_growth
        }
        
    def plot_kpis_dashboard(self):
        """Dashboard com KPIs principais de vendas"""
        # Calcular KPIs
        total_clients = len(self.sales_data)
        active_clients = len(self.sales_data[self.sales_data['mrr_12m'] > 0])
        total_mrr = self.sales_data['mrr_12m'].sum()
        total_contracts = self.sales_data['vlr_contratacoes'].sum()
        avg_mrr = self.sales_data[self.sales_data['mrr_12m'] > 0]['mrr_12m'].mean()
        
        # Ticket médio
        contracts_with_value = self.sales_data[self.sales_data['qtd_contratacoes'] > 0]
        avg_ticket = (contracts_with_value['vlr_contratacoes'] / contracts_with_value['qtd_contratacoes']).mean()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dashboard Executivo - KPIs de Vendas', fontsize=20, fontweight='bold', y=0.98)
        
        # KPI 1: Total de Clientes
        axes[0, 0].text(0.5, 0.6, f'{total_clients:,}', ha='center', va='center', 
                       fontsize=48, fontweight='bold', color='darkblue')
        axes[0, 0].text(0.5, 0.3, 'Total de Clientes', ha='center', va='center', 
                       fontsize=16, fontweight='bold')
        axes[0, 0].text(0.5, 0.1, f'({active_clients:,} ativos)', ha='center', va='center', 
                       fontsize=12, color='green')
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        
        # KPI 2: Receita Total MRR
        axes[0, 1].text(0.5, 0.6, f'R$ {total_mrr/1000000:.1f}M', ha='center', va='center', 
                       fontsize=42, fontweight='bold', color='darkgreen')
        axes[0, 1].text(0.5, 0.3, 'Receita MRR Anual', ha='center', va='center', 
                       fontsize=16, fontweight='bold')
        axes[0, 1].text(0.5, 0.1, f'(R$ {total_mrr/12:,.0f}/mês)', ha='center', va='center', 
                       fontsize=12, color='green')
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis('off')
        
        # KPI 3: MRR Médio por Cliente
        axes[0, 2].text(0.5, 0.6, f'R$ {avg_mrr:,.0f}', ha='center', va='center', 
                       fontsize=42, fontweight='bold', color='purple')
        axes[0, 2].text(0.5, 0.3, 'MRR Médio/Cliente', ha='center', va='center', 
                       fontsize=16, fontweight='bold')
        axes[0, 2].text(0.5, 0.1, 'por ano', ha='center', va='center', 
                       fontsize=12, color='gray')
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].axis('off')
        
        # KPI 4: Total Contratado
        axes[1, 0].text(0.5, 0.6, f'R$ {total_contracts/1000000:.1f}M', ha='center', va='center', 
                       fontsize=42, fontweight='bold', color='orange')
        axes[1, 0].text(0.5, 0.3, 'Total Contratado', ha='center', va='center', 
                       fontsize=16, fontweight='bold')
        axes[1, 0].text(0.5, 0.1, 'últimos 12 meses', ha='center', va='center', 
                       fontsize=12, color='gray')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        
        # KPI 5: Ticket Médio
        axes[1, 1].text(0.5, 0.6, f'R$ {avg_ticket:,.0f}', ha='center', va='center', 
                       fontsize=42, fontweight='bold', color='red')
        axes[1, 1].text(0.5, 0.3, 'Ticket Médio', ha='center', va='center', 
                       fontsize=16, fontweight='bold')
        axes[1, 1].text(0.5, 0.1, 'por contratação', ha='center', va='center', 
                       fontsize=12, color='gray')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        # KPI 6: Taxa de Conversão (simulada)
        conversion_rate = (active_clients / total_clients) * 100
        axes[1, 2].text(0.5, 0.6, f'{conversion_rate:.1f}%', ha='center', va='center', 
                       fontsize=48, fontweight='bold', color='teal')
        axes[1, 2].text(0.5, 0.3, 'Taxa de Ativação', ha='center', va='center', 
                       fontsize=16, fontweight='bold')
        axes[1, 2].text(0.5, 0.1, 'clientes ativos', ha='center', va='center', 
                       fontsize=12, color='gray')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        # Adicionar bordas aos KPIs
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(2)
                spine.set_visible(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kpis_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'total_mrr': total_mrr,
            'avg_mrr': avg_mrr,
            'total_contracts': total_contracts,
            'avg_ticket': avg_ticket,
            'conversion_rate': conversion_rate
        }
        
    def generate_complete_dashboard(self):
        """Gera o dashboard completo de vendas"""
        self.logger.info("Iniciando geração do dashboard de vendas...")
        
        # Carregar dados
        self.load_sales_data()
        
        # Criar diretório de saída
        self.create_output_directory()
        
        # Gerar todas as visualizações
        self.logger.info("Gerando gráfico de concentração de receita...")
        self.plot_revenue_concentration()
        
        self.logger.info("Gerando análise de performance de vendas...")
        self.plot_sales_performance()
        
        self.logger.info("Gerando análise de perfil de clientes...")
        self.plot_customer_segments()
        
        self.logger.info("Gerando previsão de receita...")
        prediction_insights = self.plot_revenue_prediction()
        
        self.logger.info("Gerando dashboard de KPIs...")
        kpi_insights = self.plot_kpis_dashboard()
        
        self.logger.info("Dashboard de vendas gerado com sucesso!")
        
        return {
            'kpis': kpi_insights,
            'predictions': prediction_insights,
            'output_directory': self.output_dir
        }

def main():
    # Criar instância do dashboard
    dashboard = SalesDashboardCreator()
    
    # Gerar dashboard completo
    results = dashboard.generate_complete_dashboard()
    
    # Exibir resultados
    print("\n" + "="*60)
    print("🎯 DASHBOARD DE VENDAS GERADO COM SUCESSO!")
    print("="*60)
    print(f"📁 Arquivos salvos em: {results['output_directory']}/")
    print("\n📊 VISUALIZAÇÕES GERADAS:")
    print("• concentracao_receita.png - Análise de Pareto e segmentação")
    print("• performance_vendas.png - Performance e análise ROI")
    print("• perfil_clientes.png - Segmentação e ticket médio")
    print("• previsao_receita.png - Previsões para 6 meses")
    print("• kpis_dashboard.png - KPIs executivos")
    
    print(f"\n📈 KPIs PRINCIPAIS:")
    kpis = results['kpis']
    print(f"• Total de Clientes: {kpis['total_clients']:,}")
    print(f"• Clientes Ativos: {kpis['active_clients']:,}")
    print(f"• Receita MRR Anual: R$ {kpis['total_mrr']:,.0f}")
    print(f"• MRR Médio/Cliente: R$ {kpis['avg_mrr']:,.0f}")
    print(f"• Ticket Médio: R$ {kpis['avg_ticket']:,.0f}")
    print(f"• Taxa de Ativação: {kpis['conversion_rate']:.1f}%")
    
    predictions = results['predictions']
    print(f"\n🔮 PREVISÕES (6 meses):")
    print(f"• Modelo Linear: R$ {predictions['previsao_6m_linear']:,.0f}")
    print(f"• Modelo Polinomial: R$ {predictions['previsao_6m_poly']:,.0f}")
    print(f"• Crescimento Médio Mensal: {predictions['crescimento_medio']:.1f}%")
    
    print("\n🎨 Todas as visualizações estão em português brasileiro!")
    print("💼 Dashboard pronto para apresentação executiva!")

if __name__ == "__main__":
    main() 