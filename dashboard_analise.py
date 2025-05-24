import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional, List, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesDashboardAnalyzer:
    def __init__(self, data_directory: str = 'TotvsData'):
        self.data_directory = data_directory
        self.setup_logging()
        self.dataframes = {}
        self.original_dataframes = {}
        self.unified_dataset = None
        
        self.file_client_columns = {
            "clientes_desde.csv": "CLIENTE",
            "contratacoes_ultimos_12_meses.csv": "CD_CLIENTE",
            "dados_clientes.csv": "CD_CLIENTE",
            "historico.csv": "CD_CLI",
            "mrr.csv": "CLIENTE",
            "nps_relacional.csv": "metadata_codcliente",
            "nps_transacional_aquisicao.csv": "Cód. Cliente",
            "nps_transacional_implantacao.csv": "Cód. Cliente",
            "nps_transacional_onboarding.csv": "Cod Cliente",
            "nps_transacional_produto.csv": "Cód. T",
            "nps_transacional_suporte.csv": "cliente",
            "tickets.csv": "CODIGO_ORGANIZACAO"
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dashboard_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_directory(self) -> bool:
        if not os.path.exists(self.data_directory):
            self.logger.error(f"Directory not found: {self.data_directory}")
            return False
            
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        if not csv_files:
            self.logger.error(f"No CSV files found in {self.data_directory}")
            return False
            
        self.logger.info(f"Found {len(csv_files)} CSV files in {self.data_directory}")
        return True
        
    def read_csv_robust(self, filepath: str, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        encodings = ['utf-8-sig', 'utf-8', 'latin1', 'iso-8859-1']
        separators = [';', ',', '\t']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding=encoding, engine='python')
                    if len(df.columns) > 1 and len(df) > 0:
                        if sample_size and len(df) > sample_size:
                            df = df.sample(n=sample_size, random_state=42)
                        return df
                except Exception as e:
                    continue
                    
        self.logger.error(f"Failed to read {filepath} with any configuration")
        return None
        
    def analyze_data_quality(self, df: pd.DataFrame, name: str) -> Dict:
        total_rows = len(df)
        duplicates = df.duplicated().sum()
        missing_values = df.isnull().sum().sum()
        
        outliers = 0
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers += df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            
        quality_metrics = {
            'name': name,
            'total_rows': total_rows,
            'duplicates': duplicates,
            'missing_values': missing_values,
            'outliers': outliers,
            'duplicate_percentage': (duplicates / total_rows) * 100 if total_rows > 0 else 0,
            'missing_percentage': (missing_values / (total_rows * len(df.columns))) * 100 if total_rows > 0 else 0
        }
        
        return quality_metrics
        
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        for col in df_cleaned.select_dtypes(include=['float64']).columns:
            df_cleaned[col] = df_cleaned[col].astype('float32')
        for col in df_cleaned.select_dtypes(include=['int64']).columns:
            df_cleaned[col] = df_cleaned[col].astype('int32')
            
        df_cleaned = df_cleaned.drop_duplicates()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'UNKNOWN')
                    
        return df_cleaned
        
    def standardize_client_id(self, df: pd.DataFrame, original_column: str) -> pd.DataFrame:
        if original_column not in df.columns:
            self.logger.warning(f"Column '{original_column}' not found in dataframe")
            return df
            
        df_std = df.copy()
        df_std.rename(columns={original_column: "client_id"}, inplace=True)
        df_std["client_id"] = df_std["client_id"].astype(str).str.strip().str.upper()
        
        return df_std
        
    def load_all_datasets(self, sample_size: int = 1000) -> Dict[str, Dict]:
        analysis_results = {}
        
        if not self.validate_directory():
            return analysis_results
            
        for filename, client_column in self.file_client_columns.items():
            filepath = os.path.join(self.data_directory, filename)
            
            if not os.path.exists(filepath):
                self.logger.warning(f"File not found: {filename}")
                continue
                
            df = self.read_csv_robust(filepath, sample_size)
            if df is None:
                continue
                
            df.columns = df.columns.str.strip()
            
            df_original = df.copy()
            if client_column in df_original.columns:
                df_original.rename(columns={client_column: "client_id"}, inplace=True)
                df_original["client_id"] = df_original["client_id"].astype(str).str.strip().str.upper()
            
            self.original_dataframes[filename.replace('.csv', '')] = df_original
            
            quality_metrics = self.analyze_data_quality(df, filename)
            
            df_cleaned = self.clean_dataframe(df)
            df_standardized = self.standardize_client_id(df_cleaned, client_column)
            
            self.dataframes[filename.replace('.csv', '')] = df_standardized
            
            analysis_results[filename] = {
                'quality_metrics': quality_metrics,
                'shape': df_standardized.shape,
                'columns': df_standardized.columns.tolist()
            }
            
            self.logger.info(f"Processed {filename}: {len(df_standardized)} rows")
            
        return analysis_results
        
    def create_unified_dataset(self) -> Optional[pd.DataFrame]:
        if 'dados_clientes' not in self.dataframes:
            self.logger.error("Main dataset 'dados_clientes' not found")
            return None
            
        unified_df = self.dataframes['dados_clientes'].copy()
        
        for name, df in self.dataframes.items():
            if name == 'dados_clientes' or 'client_id' not in df.columns:
                continue
                
            try:
                unified_df = unified_df.merge(df, on='client_id', how='left', suffixes=('', f'_{name}'))
                self.logger.info(f"Merged with {name}: {unified_df.shape}")
            except Exception as e:
                self.logger.error(f"Failed to merge {name}: {e}")
                
        self.unified_dataset = unified_df
        return unified_df
        
    def generate_sales_insights(self) -> Dict:
        if self.unified_dataset is None:
            self.logger.error("Unified dataset not available")
            return {}
            
        insights = {}
        df = self.unified_dataset
        
        insights['total_clients'] = df['client_id'].nunique()
        insights['total_records'] = len(df)
        
        if 'mrr' in self.dataframes:
            mrr_df = self.dataframes['mrr']
            insights['revenue_metrics'] = {
                'total_mrr': mrr_df.select_dtypes(include=[np.number]).sum().sum(),
                'avg_mrr_per_client': mrr_df.select_dtypes(include=[np.number]).mean().mean(),
                'mrr_distribution': mrr_df.select_dtypes(include=[np.number]).describe().to_dict()
            }
            
        if 'tickets' in self.dataframes:
            tickets_df = self.dataframes['tickets']
            insights['support_metrics'] = {
                'total_tickets': len(tickets_df),
                'unique_clients_with_tickets': tickets_df['client_id'].nunique(),
                'avg_tickets_per_client': len(tickets_df) / tickets_df['client_id'].nunique() if tickets_df['client_id'].nunique() > 0 else 0
            }
            
        nps_files = [name for name in self.dataframes.keys() if 'nps' in name]
        if nps_files:
            insights['nps_metrics'] = {}
            for nps_file in nps_files:
                nps_df = self.dataframes[nps_file]
                numeric_cols = nps_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    insights['nps_metrics'][nps_file] = {
                        'responses': len(nps_df),
                        'avg_score': nps_df[numeric_cols].mean().mean() if len(numeric_cols) > 0 else 0
                    }
                    
        return insights
        
    def create_visualizations(self, output_dir: str = 'dashboard_plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.style.use('default')
        
        if self.unified_dataset is not None and not self.unified_dataset.empty:
            
            plt.figure(figsize=(12, 6))
            client_counts = self.unified_dataset.groupby('client_id').size()
            plt.hist(client_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribuição de Registros por Cliente')
            plt.xlabel('Número de Registros')
            plt.ylabel('Frequência')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'client_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        if 'mrr' in self.dataframes:
            mrr_df = self.dataframes['mrr']
            numeric_cols = mrr_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                plt.figure(figsize=(10, 6))
                for col in numeric_cols[:3]:  
                    plt.hist(mrr_df[col].dropna(), bins=30, alpha=0.5, label=col)
                plt.title('Distribuição de MRR')
                plt.xlabel('Valor MRR')
                plt.ylabel('Frequência')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'mrr_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        if 'tickets' in self.dataframes:
            tickets_df = self.dataframes['tickets']
            plt.figure(figsize=(10, 6))
            tickets_per_client = tickets_df['client_id'].value_counts()
            plt.hist(tickets_per_client, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title('Distribuição de Tickets de Suporte por Cliente')
            plt.xlabel('Número de Tickets')
            plt.ylabel('Número de Clientes')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'tickets_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        quality_data = []
        for name, df in self.dataframes.items():
            quality_metrics = self.analyze_data_quality(df, name)
            quality_data.append(quality_metrics)
            
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            
            short_names = []
            for name in quality_df['name']:
                if name.endswith('.csv'):
                    name = name[:-4]
                
                if 'nps_transacional' in name:
                    name = name.replace('nps_transacional_', 'NPS_')
                elif 'nps_relacional' in name:
                    name = 'NPS_rel'
                elif 'dados_clientes' in name:
                    name = 'clients'
                elif 'clientes_desde' in name:
                    name = 'clients_since'
                elif 'contratacoes_ultimos_12_meses' in name:
                    name = 'contracts_12m'
                elif len(name) > 12:
                    name = name[:12] + '...'
                    
                short_names.append(name)
            
            quality_df['short_name'] = short_names
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            axes[0, 0].bar(quality_df['short_name'], quality_df['total_rows'], color='steelblue')
            axes[0, 0].set_title('Total de Linhas por Dataset', fontsize=12, fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].bar(quality_df['short_name'], quality_df['duplicate_percentage'], color='orange')
            axes[0, 1].set_title('Percentual de Duplicatas por Dataset', fontsize=12, fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45, labelsize=9)
            axes[0, 1].set_ylabel('Percentual (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].bar(quality_df['short_name'], quality_df['missing_percentage'], color='red', alpha=0.7)
            axes[1, 0].set_title('Percentual de Valores Ausentes por Dataset', fontsize=12, fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
            axes[1, 0].set_ylabel('Percentual (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].bar(quality_df['short_name'], quality_df['outliers'], color='purple', alpha=0.7)
            axes[1, 1].set_title('Contagem de Outliers por Dataset', fontsize=12, fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
            axes[1, 1].set_ylabel('Contagem')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.savefig(os.path.join(output_dir, 'data_quality_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        self.logger.info(f"Visualizations saved to {output_dir}")
        
    def create_before_after_visualizations(self, output_dir: str = 'dashboard_plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.style.use('default')
        
        original_quality_data = []
        cleaned_quality_data = []
        
        for name in self.dataframes.keys():
            if name in self.original_dataframes:
                original_metrics = self.analyze_data_quality(self.original_dataframes[name], name)
                cleaned_metrics = self.analyze_data_quality(self.dataframes[name], name)
                
                original_quality_data.append(original_metrics)
                cleaned_quality_data.append(cleaned_metrics)
        
        if original_quality_data and cleaned_quality_data:
            original_df = pd.DataFrame(original_quality_data)
            cleaned_df = pd.DataFrame(cleaned_quality_data)
            
            short_names = []
            for name in original_df['name']:
                if name.endswith('.csv'):
                    name = name[:-4]
                
                if 'nps_transacional' in name:
                    name = name.replace('nps_transacional_', 'NPS_')
                elif 'nps_relacional' in name:
                    name = 'NPS_rel'
                elif 'dados_clientes' in name:
                    name = 'clients'
                elif 'clientes_desde' in name:
                    name = 'clients_since'
                elif 'contratacoes_ultimos_12_meses' in name:
                    name = 'contracts_12m'
                elif len(name) > 12:
                    name = name[:12] + '...'
                    
                short_names.append(name)
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            
            x_pos = np.arange(len(short_names))
            width = 0.35
            
            axes[0, 0].bar(x_pos - width/2, original_df['duplicate_percentage'], width, 
                          label='Antes da Limpeza', color='lightcoral', alpha=0.8)
            axes[0, 0].bar(x_pos + width/2, cleaned_df['duplicate_percentage'], width,
                          label='Após Limpeza', color='lightgreen', alpha=0.8)
            axes[0, 0].set_title('Comparação: Percentual de Duplicatas', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Percentual (%)')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(short_names, rotation=45, fontsize=9)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].bar(x_pos - width/2, original_df['missing_percentage'], width,
                          label='Antes da Limpeza', color='orange', alpha=0.8)
            axes[0, 1].bar(x_pos + width/2, cleaned_df['missing_percentage'], width,
                          label='Após Limpeza', color='skyblue', alpha=0.8)
            axes[0, 1].set_title('Comparação: Percentual de Valores Ausentes', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Percentual (%)')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(short_names, rotation=45, fontsize=9)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].bar(x_pos - width/2, original_df['outliers'], width,
                          label='Antes da Limpeza', color='purple', alpha=0.8)
            axes[1, 0].bar(x_pos + width/2, cleaned_df['outliers'], width,
                          label='Após Limpeza', color='gold', alpha=0.8)
            axes[1, 0].set_title('Comparação: Contagem de Outliers', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Contagem')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(short_names, rotation=45, fontsize=9)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            improvement_duplicates = original_df['duplicate_percentage'] - cleaned_df['duplicate_percentage']
            improvement_missing = original_df['missing_percentage'] - cleaned_df['missing_percentage']
            
            axes[1, 1].bar(short_names, improvement_duplicates, alpha=0.7, color='green', label='Redução Duplicatas')
            axes[1, 1].bar(short_names, improvement_missing, alpha=0.7, color='blue', label='Redução Valores Ausentes', bottom=improvement_duplicates)
            axes[1, 1].set_title('Melhoria na Qualidade dos Dados', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Redução (%)')
            axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.savefig(os.path.join(output_dir, 'before_after_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            total_original_duplicates = original_df['duplicates'].sum()
            total_cleaned_duplicates = cleaned_df['duplicates'].sum()
            total_original_missing = original_df['missing_values'].sum()
            total_cleaned_missing = cleaned_df['missing_values'].sum()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            categories = ['Duplicatas', 'Valores Ausentes']
            original_totals = [total_original_duplicates, total_original_missing]
            cleaned_totals = [total_cleaned_duplicates, total_cleaned_missing]
            
            x_pos = np.arange(len(categories))
            width = 0.35
            
            axes[0].bar(x_pos - width/2, original_totals, width, label='Antes da Limpeza', color='red', alpha=0.7)
            axes[0].bar(x_pos + width/2, cleaned_totals, width, label='Após Limpeza', color='green', alpha=0.7)
            axes[0].set_title('Resumo Geral: Antes vs Depois da Limpeza', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Contagem Total')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(categories)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            improvement_data = [original_totals[i] - cleaned_totals[i] for i in range(len(categories))]
            colors = ['lightcoral', 'lightblue']
            
            axes[1].bar(categories, improvement_data, color=colors, alpha=0.8)
            axes[1].set_title('Quantidade de Problemas Resolvidos', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Redução Absoluta')
            axes[1].grid(True, alpha=0.3)
            
            for i, v in enumerate(improvement_data):
                axes[1].text(i, v + max(improvement_data) * 0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'data_cleaning_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        self.logger.info(f"Before/After comparison visualizations saved to {output_dir}")
        
    def export_results(self, output_file: str = 'sales_analysis_results.csv'):
        if self.unified_dataset is not None:
            self.unified_dataset.to_csv(output_file, index=False)
            self.logger.info(f"Unified dataset exported to {output_file}")
            
            summary_file = output_file.replace('.csv', '_summary.txt')
            insights = self.generate_sales_insights()
            
            with open(summary_file, 'w') as f:
                f.write("SALES DASHBOARD ANALYSIS SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                
                for key, value in insights.items():
                    f.write(f"{key.upper()}:\n")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"  {value}\n")
                    f.write("\n")
                    
            self.logger.info(f"Analysis summary exported to {summary_file}")
            
    def run_complete_analysis(self, sample_size: int = 1000, create_plots: bool = True) -> Dict:
        self.logger.info("Starting complete sales dashboard analysis")
        
        analysis_results = self.load_all_datasets(sample_size)
        
        unified_dataset = self.create_unified_dataset()
        
        insights = self.generate_sales_insights()
        
        if create_plots:
            self.create_visualizations()
            
        self.create_before_after_visualizations()
        
        self.export_results()
        
        self.logger.info("Analysis completed successfully")
        
        return {
            'datasets_loaded': len(self.dataframes),
            'unified_dataset_shape': unified_dataset.shape if unified_dataset is not None else (0, 0),
            'insights': insights,
            'analysis_results': analysis_results
        }

def main():
    analyzer = SalesDashboardAnalyzer()
    
    results = analyzer.run_complete_analysis(
        sample_size=2000,  
        create_plots=True
    )
    
    print("\n" + "="*50)
    print("SALES DASHBOARD ANALYSIS COMPLETED")
    print("="*50)
    print(f"Datasets processed: {results['datasets_loaded']}")
    print(f"Unified dataset shape: {results['unified_dataset_shape']}")
    print(f"Total insights generated: {len(results['insights'])}")
    print("\nCheck the generated files:")
    print("- sales_analysis_results.csv (unified dataset)")
    print("- sales_analysis_results_summary.txt (insights summary)")
    print("- dashboard_plots/ (visualizations)")
    print("  • client_distribution.png (distribuição de clientes)")
    print("  • mrr_distribution.png (distribuição MRR)")
    print("  • tickets_distribution.png (distribuição tickets)")
    print("  • data_quality_overview.png (qualidade dos dados)")
    print("  • before_after_comparison.png (comparação antes/depois)")
    print("  • data_cleaning_summary.png (resumo da limpeza)")
    print("- dashboard_analysis.log (execution log)")
    print("\n🎯 Novos gráficos comparativos mostram o impacto da limpeza dos dados!")

if __name__ == "__main__":
    main() 