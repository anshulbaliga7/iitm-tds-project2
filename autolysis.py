# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0.0",
#   "seaborn>=0.12.0",
#   "matplotlib>=3.7.0",
#   "openai>=1.0.0",
#   "scikit-learn>=1.3.0",
#   "requests>=2.31.0",
#   "statsmodels>=0.13.0"
# ]
# ///

import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import openai
import requests
import numpy as np

class DataAnalyzer:
    def __init__(self, csv_path):
        """
        Initialize the data analyzer with the given CSV file.
        
        Args:
            csv_path (str): Path to the input CSV file
        """
        # Validate input file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input file {csv_path} not found")
        
        # Create output directory based on CSV filename
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        self.output_dir = os.path.join(os.path.dirname(csv_path), csv_basename)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Try different encodings and handle empty files
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(csv_path, encoding=encoding)
                if self.df.empty:
                    raise ValueError("The CSV file is empty")
                if len(self.df.columns) < 2:
                    raise ValueError("The CSV file must have at least 2 columns")
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file is empty")
        else:
            raise UnicodeDecodeError(f"Unable to read file with any of the following encodings: {encodings}")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values globally
        self.df = self.df.replace(['nan', 'NaN', 'NULL', ''], pd.NA)
        
        # Set up OpenAI client
        openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = os.environ["AIPROXY_TOKEN"]
        #openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDI3NDNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.i6MpRliZ3nPhSAQ_bOkOW-isk4R9iXZY3cM-3AuFi3o"
        
        # Set up README path in the output directory
        self.readme_path = os.path.join(self.output_dir, 'README.md')
    
    def generate_data_summary(self):
        """
        Generate a comprehensive summary of the dataset.
        
        Returns:
            dict: A summary of dataset characteristics
        """
        try:
            # Basic dataset info with error handling
            summary = {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "columns": list(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "missing_values": self.df.isna().sum().to_dict(),  # isna() catches more types of missing values
                "duplicates": self.df.duplicated().sum()
            }
            
            # Numeric columns analysis with safeguards
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
            if len(numeric_cols) > 0:
                try:
                    desc = self.df[numeric_cols].describe()
                    # Handle potential inf values
                    desc = desc.replace([np.inf, -np.inf], np.nan)
                    summary["numeric_summary"] = desc.to_dict()
                except Exception as e:
                    summary["numeric_summary"] = f"Error analyzing numeric columns: {str(e)}"
            
            # Categorical columns analysis with size limits
            cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool', 'string']).columns
            if len(cat_cols) > 0:
                summary["categorical_summary"] = {}
                for col in cat_cols:
                    try:
                        # Limit analysis for columns with too many unique values
                        if self.df[col].nunique() > 1000:
                            summary["categorical_summary"][col] = {
                                "unique_values": self.df[col].nunique(),
                                "warning": "Too many unique values for detailed analysis"
                            }
                        else:
                            value_counts = self.df[col].value_counts()
                            # Handle potentially large string values
                            truncated_dict = {
                                str(k)[:100]: v for k, v in 
                                value_counts.head(5).to_dict().items()
                            }
                            summary["categorical_summary"][col] = {
                                "unique_values": self.df[col].nunique(),
                                "top_values": truncated_dict
                            }
                    except Exception as e:
                        summary["categorical_summary"][col] = f"Error analyzing column: {str(e)}"
            
            # Date columns detection and analysis with better error handling
            date_cols = []
            for col in self.df.columns:
                try:
                    # Check if column is already datetime
                    if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        date_cols.append(col)
                    else:
                        # Sample the column to test for datetime conversion
                        sample = self.df[col].dropna().head(1000)
                        pd.to_datetime(sample, errors='raise')
                        date_cols.append(col)
                except:
                    continue
            
            if date_cols:
                summary["date_columns"] = {}
                for col in date_cols:
                    try:
                        dates = pd.to_datetime(self.df[col], errors='coerce')
                        valid_dates = dates.dropna()
                        if len(valid_dates) > 0:
                            summary["date_columns"][col] = {
                                "min_date": str(valid_dates.min()),
                                "max_date": str(valid_dates.max()),
                                "valid_dates_count": len(valid_dates),
                                "invalid_dates_count": len(dates) - len(valid_dates)
                            }
                    except Exception as e:
                        summary["date_columns"][col] = f"Error analyzing dates: {str(e)}"
            
            return summary
        except Exception as e:
            print(f"Error generating data summary: {e}")
            return {"error": str(e)}
    
    def detect_correlations(self):
        """
        Detect and visualize correlations between numeric columns.
        
        Returns:
            dict: Correlation matrix and visualization path
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        
        if len(numeric_df.columns) < 2:
            return {"error": "Not enough numeric columns for correlation analysis"}
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create correlation heatmap with 512x512 size
        plt.figure(figsize=(6.4, 6.4))  # 6.4 inches = 512 pixels at 80 DPI
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        corr_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(corr_path, dpi=80, bbox_inches='tight')
        plt.close()
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "correlation_visualization": corr_path
        }

    def cluster_analysis(self, n_init=10):
        """
        Perform basic clustering analysis with preprocessing for missing values.
        
        Returns:
            dict: Clustering results and visualization
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        
        if len(numeric_df.columns) < 2:
            return {"error": "Not enough numeric columns for clustering"}
        
        # Handle missing values using mean imputation
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(numeric_df)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=n_init)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Visualize clusters using first two principal components
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        plt.figure(figsize=(6.4, 6.4))  # 6.4 inches = 512 pixels at 80 DPI
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
        plt.title('Cluster Analysis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        cluster_path = os.path.join(self.output_dir, 'cluster_analysis.png')
        plt.savefig(cluster_path, dpi=80, bbox_inches='tight')
        plt.close()
        
        return {
            "cluster_labels": clusters.tolist(),
            "cluster_visualization": cluster_path
        }
    
    def generate_story(self, data_summary, correlation_results, cluster_results):
        """
        Generate a narrative story combining technical and creative analyses.
        """
        try:
            # Prepare analysis insights
            insights = {
                "data_overview": {
                    "size": f"{data_summary['total_rows']} rows × {data_summary['total_columns']} columns",
                    "memory": data_summary['memory_usage'],
                    "duplicates": data_summary['duplicates']
                },
                "quantum_template": self._get_quantum_template(),
                "missing_data": {coll: count for coll, count in data_summary['missing_values'].items() if count > 0},
                "correlations": correlation_results.get('correlation_matrix', {}),
                "clusters": len(set(cluster_results.get('cluster_labels', []))) if 'cluster_labels' in cluster_results else 0
            }
            
            # Get both technical and creative analyses
            technical_analysis, creative_prompt = self._generate_dynamic_prompt(insights)
            
            # Get creative narrative
            creative_response = self._make_api_call(creative_prompt)
            
            if creative_response.status_code == 200:
                creative_story = creative_response.json()['choices'][0]['message']['content']
                
                # Create combined story
                combined_story = f"""# From Numbers to Narratives: Revealing Data Secrets 
## Anshul Ramdas Baliga, 22f3002743
## Executive Summary
This analysis presents a comprehensive examination of the dataset through two complementary lenses:
1. A creative quantum-temporal interpretation for innovative pattern discovery (My unique story-telling approach)
2. A technical statistical analysis for rigorous data insights 

## Quantum Temporal Analysis on the dataset  (My unique story-telling approach)
Note: The following section reframes our technical findings through a **quantum-temporal lens** to explore innovative patterns and relationships in the data. Hope you enjoy the story!\n
{creative_story}

## Technical Analysis
{technical_analysis}
---

---
## Visualizations
"""
                
                # Write to single README file
                with open(self.readme_path, 'w', encoding='utf-8') as f:
                    f.write(combined_story)
                    f.write('\n\n### Correlation Analysis\n![Correlation Heatmap](correlation_heatmap.png)\n')
                    f.write('\n\n### Cluster Analysis\n![Cluster Analysis](cluster_analysis.png)\n')
                    f.write('\n\n### Statistical Summary\n![Statistical Summary](statistical_summary.png)\n')
                    f.write('\n\n### Categorical Analysis\n![Categorical Analysis](categorical_analysis.png)\n')
                return combined_story
            else:
                raise Exception(f"API request failed with status code: {creative_response.status_code}")
        except Exception as e:
            print(f"Story generation error: {e}")
            return "# Data Analysis Story\n\nUnable to generate narrative due to an error."
    
    def analyze(self):
        """
        Perform comprehensive data analysis with advanced techniques.
        """
        try:
            # Suppress warnings
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # Create output directory
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Dynamic analysis based on data characteristics
            analysis_plan = self._determine_analysis_plan()
            
            # Core analyses
            data_summary = self.generate_data_summary()
            correlation_results = self.detect_correlations()
            cluster_results = self.cluster_analysis(n_init=10)
            
            # Advanced analyses based on data types
            if analysis_plan['time_series']:
                self._analyze_temporal_patterns()
            if analysis_plan['categorical']:
                self.analyze_categorical_patterns()
            if analysis_plan['numerical']:
                self._analyze_distributions()
                self.visualize_statistics()
                self._detect_anomalies()
            
            # Generate story
            story = self.generate_story(data_summary, correlation_results, cluster_results)
            
            print(f"Analysis complete. Check {self.readme_path} and generated images.")
            return True
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False
    
    def _determine_analysis_plan(self):
        """
        Dynamically determine appropriate analyses based on data characteristics.
        """
        plan = {
            'time_series': False,
            'categorical': False,
            'numerical': False,
            'advanced_stats': False
        }
        
        # Check for datetime columns
        date_pattern = r'\d{1,2}[-/]\w{3}[-/]\d{2,4}'
        for col in self.df.columns:
            if self.df[col].astype(str).str.match(date_pattern).any():
                plan['time_series'] = True
                break
        
        # Check for categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        plan['categorical'] = len(cat_cols) > 0
        
        # Check for numerical columns
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        plan['numerical'] = len(num_cols) > 0
        
        # Check if advanced statistical tests are appropriate
        if len(num_cols) >= 2:
            plan['advanced_stats'] = True
        
        return plan
    
    def _analyze_temporal_patterns(self):
        """
        Analyze temporal patterns in the data.
        """
        try:
            date_cols = []
            date_pattern = r'\d{1,2}[-/]\w{3}[-/]\d{2,4}'
            
            for col in self.df.columns:
                if self.df[col].astype(str).str.match(date_pattern).any():
                    date_cols.append(col)
            
            if not date_cols:
                return None
            
            # Convert to datetime and analyze trends
            for col in date_cols:
                dates = pd.to_datetime(self.df[col], format='%d-%b-%y', errors='coerce')
                if not dates.isna().all():
                    self._analyze_single_temporal(dates, col)
        except Exception as e:
            print(f"Temporal analysis error: {e}")
    
    def _analyze_single_temporal(self, dates, col_name):
        """
        Analyze a single temporal column.
        """
        # Basic temporal statistics
        stats = {
            'min_date': dates.min(),
            'max_date': dates.max(),
            'date_range': (dates.max() - dates.min()).days,
            'missing_dates': dates.isna().sum()
        }
        
        # Time series decomposition if enough data points
        if len(dates) > 50:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(dates.value_counts().sort_index(), period=12)
                
                plt.figure(figsize=(6.4, 6.4))
                plt.subplot(311)
                decomposition.trend.plot()
                plt.title('Trend')
                plt.subplot(312)
                decomposition.seasonal.plot()
                plt.title('Seasonal')
                plt.subplot(313)
                decomposition.resid.plot()
                plt.title('Residual')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'temporal_analysis_{col_name}.png'))
                plt.close()
            except Exception as e:
                print(f"Time series decomposition error: {e}")
    
    def _detect_anomalies(self):
        """
        Detect anomalies using Isolation Forest.
        """
        from sklearn.ensemble import IsolationForest
        
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return None
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(numeric_df)
        
        # Store anomaly information
        self.anomalies = {
            'total': sum(anomalies == -1),
            'percentage': sum(anomalies == -1) / len(anomalies) * 100,
            'columns': numeric_df.columns.tolist()
        }
    
    def _generate_dynamic_prompt(self, insights):
        """
        Generate structured prompts for technical and creative narratives.
        """
        # Extract key metrics
        total_rows = insights['data_overview']['size']
        missing_data = len(insights['missing_data'])
        num_clusters = insights['clusters']
        
        # Build technical analysis prompt with clear structure
        technical_prompt = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a data scientist creating a technical analysis report.
                    Follow this structure:
                    1. Data Overview: Describe the dataset scope and quality
                    2. Key Patterns: Highlight significant correlations and distributions
                    3. Cluster Analysis: Explain cluster characteristics and implications
                    4. Statistical Insights: Present key statistical findings
                    5. Recommendations: Suggest potential actions based on findings
                    
                    Use markdown formatting for clear organization. Create tables where appropriate.
                    Emphasize significant findings with bold text."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Please analyze this dataset with the following details:
                    
                    Dataset Characteristics:
                    - Size: {total_rows}
                    - Missing Values: {missing_data} columns affected
                    - Distinct Clusters: {num_clusters}
                    
                    Correlation Findings:                ```json
                    {json.dumps(insights['correlations'], indent=2)}                ```
                    
                    Data Quality Metrics:
                    - Duplicates: {insights['data_overview']['duplicates']}
                    - Memory Usage: {insights['data_overview']['memory']}
                    
                    Missing Data Patterns:                ```json
                    {json.dumps(insights['missing_data'], indent=2)}                ```
                    
                    Focus on actionable insights and statistical significance.
                    """
                }
            ]
        }
        
        # Build creative narrative prompt with quantum template
        creative_prompt = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a creative data storyteller using quantum mechanics metaphors.
                    Structure your narrative as follows:
                    1. Introduction: Set up the quantum analogy
                    2. Data Patterns as Quantum States
                    3. Correlations as Entanglement
                    4. Clusters as Energy Levels
                    5. Conclusion: Unifying Insights
                    
                    Use markdown for formatting. Make complex patterns relatable through metaphors."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Create a quantum-inspired narrative for this dataset:
                    
                    Key Elements:
                    - {total_rows}
                    - {num_clusters} distinct clusters identified
                    - Correlation patterns present
                    - {missing_data} areas of uncertainty (missing data)
                    
                    Quantum Template:
                    {insights['quantum_template']}
                    
                    Technical Foundation:                ```json
                    {json.dumps(insights['correlations'], indent=2)}                ```
                    
                    Emphasize the interconnections and emergent patterns in the data.
                    """
                }
            ]
        }
        
        return technical_prompt, creative_prompt

    
    def _make_api_call(self, prompt):
        """
        Make API call to OpenAI.
        
        Args:
            prompt (dict): The prompt payload
        
        Returns:
            requests.Response: API response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        
        response = requests.post(
            f"{openai.api_base}/chat/completions",
            headers=headers,
            json=prompt,
            timeout=30
        )
        
        return response
    
    def visualize_statistics(self):
        """
        Create statistical summary visualizations including box plots
        and violin plots for numeric columns, optimized for small images.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return None
        
        # Create figure with two rows: box plots and violin plots, optimized for 512x512 px
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, 
                                      figsize=(6.4, 6.4))  # 6.4 inches = 512 pixels at 80 DPI
        
        # Box plots
        sns.boxplot(data=numeric_df, ax=ax1)
        ax1.set_title('Box Plots of Numeric Variables')
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plots
        sns.violinplot(data=numeric_df, ax=ax2)
        ax2.set_title('Violin Plots of Numeric Variables')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        stats_path = os.path.join(self.output_dir, 'statistical_summary.png')
        plt.savefig(stats_path, dpi=80, bbox_inches='tight')  # Removed quality and format parameters
        plt.close()
        
        return stats_path
    
    def analyze_categorical_patterns(self):
        """
        Analyze and visualize patterns in categorical columns.
        
        Returns:
            str: Path to saved visualization
        """
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            return None
        
        # Select top 4 categorical columns (reduced from 6 to fit better in 512x512)
        plot_cols = cat_cols[:4]
        n_cols = len(plot_cols)
        
        if n_cols == 0:
            return None
        
        # Create 2x2 grid for better fit in 512x512
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 6.4))
        axes = axes.flatten()
        
        for idx, col in enumerate(plot_cols):
            try:
                # Get value counts and limit to top 5 (reduced from 10 for better readability)
                value_counts = self.df[col].value_counts().head(5)
                
                if len(value_counts) > 0:
                    sns.barplot(x=value_counts.values, 
                              y=[str(x)[:20] for x in value_counts.index],  # Truncate to 20 chars
                              ax=axes[idx])
                    
                    axes[idx].set_title(f'Top 5 in {col[:15]}...' if len(col) > 15 else f'Top 5 in {col}')
                    axes[idx].set_xlabel('Count')
            except Exception as e:
                print(f"Warning: Could not plot column {col}: {str(e)}")
                axes[idx].text(0.5, 0.5, f'Could not plot {col}',
                             ha='center', va='center')
        
        # Remove empty subplots if any
        for idx in range(len(plot_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        cat_path = os.path.join(self.output_dir, 'categorical_analysis.png')
        plt.savefig(cat_path, dpi=80, bbox_inches='tight')
        plt.close()
        
        return cat_path
    
    def _get_quantum_template(self):
        """
        Returns the quantum narrative template
        """
        return """
        Temporal Reconnaissance Mission:
        - Frame findings as quantum data journeys
        - Describe patterns as temporal convergences
        - Present insights as revelations across time
        - Maintain scientific accuracy in creative narrative
        """

    def _analyze_distributions(self):
        """
        Analyze and visualize distributions of numerical columns.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return None
        
        # Create distribution plots
        for col in numeric_df.columns:
            plt.figure(figsize=(6.4, 4.8))
            sns.histplot(numeric_df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            # Save plot
            dist_path = os.path.join(self.output_dir, f'distribution_{col}.png')
            plt.savefig(dist_path, dpi=80, bbox_inches='tight')
            plt.close()

def _analyze_visualizationsyyy(self):
    """
    Analyze generated visualizations using text-based pattern detection.
    Returns:
        dict: Visual insights for each plot type
    """
    visual_insights = {}
    
    # Analyze correlation heatmap
    corr_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
    if os.path.exists(corr_path):
        visual_insights['correlation'] = self._analyze_correlation_plot()
    
    # Analyze cluster plot
    cluster_path = os.path.join(self.output_dir, 'cluster_analysis.png')
    if os.path.exists(cluster_path):
        visual_insights['clusters'] = self._analyze_cluster_plot()
    
    # Analyze statistical summary
    stats_path = os.path.join(self.output_dir, 'statistical_summary.png')
    if os.path.exists(stats_path):
        visual_insights['statistics'] = self._analyze_statistical_plot()
    
    return visual_insights

def _analyze_correlation_plotyyyy(self):
    """Extract insights from correlation matrix."""
    insights = []
    correlations = self.df.corr()
    
    # Find strongest correlations
    strong_corrs = []
    for i in range(len(correlations.columns)):
        for j in range(i+1, len(correlations.columns)):
            corr = abs(correlations.iloc[i,j])
            if corr > 0.5:  # Strong correlation threshold
                strong_corrs.append((correlations.columns[i], 
                                   correlations.columns[j], 
                                   corr))
    
    if strong_corrs:
        insights.append("Strong correlations detected between:")
        for var1, var2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)[:3]:
            insights.append(f"- {var1} and {var2} (correlation: {corr:.2f})")
    else:
        insights.append("No strong correlations detected between variables")
    
    return "\n".join(insights)

def _analyze_cluster_plotyyy(self):
    """Extract insights from cluster analysis."""
    if not hasattr(self, 'cluster_labels'):
        return "Cluster analysis not available"
        
    n_clusters = len(set(self.cluster_labels))
    sizes = [sum(self.cluster_labels == i) for i in range(n_clusters)]
    
    insights = [f"Identified {n_clusters} distinct clusters:"]
    for i, size in enumerate(sizes):
        insights.append(f"- Cluster {i+1}: {size} points ({size/len(self.cluster_labels)*100:.1f}%)")
    
    return "\n".join(insights)

def _analyze_statistical_plotyyy(self):
    """Extract insights from statistical summary."""
    numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
    insights = []
    
    for col in numeric_df.columns:
        outliers = numeric_df[col][abs(numeric_df[col] - numeric_df[col].mean()) > 3*numeric_df[col].std()]
        if len(outliers) > 0:
            insights.append(f"- {col}: {len(outliers)} outliers detected")
    
    if not insights:
        insights.append("No significant outliers detected in numerical variables")
    
    return "\n".join(insights)

def generate_storyyyys(self, data_summary, correlation_results, cluster_results):
    """Generate enhanced narrative with visual insights."""
    try:
        # Get visual insights
        visual_insights = self._analyze_visualizations()
        
        # Add visual insights to the analysis
        insights = {
            "data_overview": {
                "size": f"{data_summary['total_rows']} rows × {data_summary['total_columns']} columns",
                "memory": data_summary['memory_usage'],
                "duplicates": data_summary['duplicates']
            },
            "missing_data": {col: count for col, count in data_summary['missing_values'].items() if count > 0},
            "correlations": correlation_results.get('correlation_matrix', {}),
            "clusters": len(set(cluster_results.get('cluster_labels', []))) if 'cluster_labels' in cluster_results else 0,
            "visual_insights": visual_insights
        }
        
        # Generate technical and creative analyses
        technical_analysis, creative_prompt = self._generate_dynamic_prompt(insights)
        creative_response = self._make_api_call(creative_prompt)
        
        if creative_response.status_code == 200:
            creative_story = creative_response.json()['choices'][0]['message']['content']
            combined_story = self._format_final_story(technical_analysis, creative_story, visual_insights)
            
            # Write to README with visual insights
            with open(self.readme_path, 'w', encoding='utf-8') as f:
                f.write(combined_story)
                for plot_type, plot_insights in visual_insights.items():
                    f.write(f"\n\n### {plot_type.title()} Analysis\n")
                    f.write(f"![{plot_type.title()} Analysis]({plot_type}_analysis.png)\n")
                    f.write(f"\n{plot_insights}\n")
            
            return combined_story
        else:
            raise Exception(f"API request failed with status code: {creative_response.status_code}")
    except Exception as e:
        print(f"Story generation error: {e}")
        return "# Data Analysis Story\n\nUnable to generate narrative"

def main():
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        analyzer = DataAnalyzer(csv_path)
        analyzer.analyze()
        print("Analysis complete. Check README.md and generated images.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()