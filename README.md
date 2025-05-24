# Sales Dashboard Analysis

## Overview
This project provides comprehensive sales data analysis and dashboard generation capabilities for TOTVS sales data. It includes data cleaning, quality analysis, visualization generation, and insight extraction.

## Features

### Data Processing
- **Robust CSV Reading**: Handles multiple encodings (UTF-8, Latin1, ISO-8859-1) and separators
- **Data Quality Analysis**: Detects duplicates, missing values, and outliers using IQR method
- **Data Cleaning**: Optimizes data types, removes duplicates, and fills missing values
- **Client ID Standardization**: Unifies client identifiers across different datasets

### Sales Analytics
- **Revenue Metrics**: MRR analysis, distribution statistics, and per-client averages
- **Support Metrics**: Ticket analysis and client support patterns
- **NPS Analysis**: Customer satisfaction scores across different touchpoints
- **Customer Segmentation**: Analysis by client profiles and behavior patterns

### Visualizations
- Client distribution charts
- MRR distribution analysis
- Support ticket patterns
- Data quality overview dashboards

## File Structure

```
├── dashboard_analise.py          # Main analysis script
├── analise_exploratoria_V3.ipynb # Original Jupyter notebook
├── TotvsData/                    # Data directory
│   ├── dados_clientes.csv        # Main client data
│   ├── mrr.csv                   # Monthly recurring revenue
│   ├── tickets.csv               # Support tickets
│   ├── nps_*.csv                 # NPS surveys data
│   └── ...                       # Other CSV files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data is in TotvsData folder:**
   - Place all CSV files in the `TotvsData/` directory
   - The script automatically detects and processes available files

## Usage

### Basic Analysis
```bash
python dashboard_analise.py
```

### Advanced Usage
```python
from dashboard_analise import SalesDashboardAnalyzer

# Initialize analyzer
analyzer = SalesDashboardAnalyzer(data_directory='TotvsData')

# Run complete analysis with custom parameters
results = analyzer.run_complete_analysis(
    sample_size=5000,  # Increase sample size for larger datasets
    create_plots=True  # Generate visualizations
)

# Access specific insights
insights = analyzer.generate_sales_insights()
print(f"Total clients: {insights['total_clients']}")
print(f"Total MRR: {insights['revenue_metrics']['total_mrr']}")
```

## Output Files

After running the analysis, the following files are generated:

1. **sales_analysis_results.csv** - Unified dataset with all merged data
2. **sales_analysis_results_summary.txt** - Key insights and metrics summary
3. **dashboard_plots/** - Directory containing visualization PNG files
   - `client_distribution.png` - Client record distribution
   - `mrr_distribution.png` - Revenue distribution analysis
   - `tickets_distribution.png` - Support ticket patterns
   - `data_quality_overview.png` - Data quality metrics
4. **dashboard_analysis.log** - Execution log with processing details

## Key Improvements Over Original

### Object-Oriented Design
- Modular class-based architecture
- Reusable methods for different analysis types
- Better error handling and logging

### Enhanced Data Processing
- Automatic encoding detection
- Configurable sample sizes for large datasets
- Memory optimization through data type conversion

### Comprehensive Analytics
- Revenue analysis with statistical distributions
- Customer satisfaction metrics across touchpoints
- Support pattern analysis
- Data quality assessment

### Professional Output
- Structured logging with timestamps
- Export capabilities for further analysis
- High-quality visualizations with proper formatting

## Data Sources Supported

The script automatically processes these TOTVS data files:
- `dados_clientes.csv` - Main client information
- `clientes_desde.csv` - Client tenure data
- `contratacoes_ultimos_12_meses.csv` - Recent contracts
- `mrr.csv` - Monthly recurring revenue
- `historico.csv` - Historical data
- `tickets.csv` - Support tickets
- `nps_relacional.csv` - Relational NPS scores
- `nps_transacional_*.csv` - Transactional NPS by touchpoint

## Performance Considerations

- **Sample Size**: Default 2000 rows per file (configurable)
- **Memory Optimization**: Data type conversion for efficiency
- **Large Files**: Automatic sampling for files > 1000 rows
- **Processing Time**: ~10-15 seconds for standard dataset

## Troubleshooting

### Common Issues

1. **File Not Found Errors**: Ensure CSV files are in `TotvsData/` directory
2. **Encoding Issues**: Script automatically tries multiple encodings
3. **Memory Issues**: Reduce sample_size parameter for large datasets
4. **Missing Columns**: Check client ID column names in file mapping

### Logs
Check `dashboard_analysis.log` for detailed execution information and error messages.

## Next Steps

This analysis provides the foundation for:
- Interactive dashboard development (Power BI, Tableau)
- Advanced machine learning models
- Predictive analytics implementation
- Customer churn analysis
- Revenue forecasting

