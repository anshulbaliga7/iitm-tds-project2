# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0.0",
#   "seaborn>=0.12.0",
#   "matplotlib>=3.7.0",
#   "openai>=1.0.0",
#   "scikit-learn>=1.3.0",
#   "requests>=2.31.0",
#   "statsmodels>=0.14.0"
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
import statsmodels.api as sm

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
        Perform adaptive clustering analysis with dynamic technique selection.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        
        if len(numeric_df.columns) < 2:
            return {"error": "Not enough numeric columns for clustering"}
        
        # Handle missing values using iterative imputer for better accuracy
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=42)
        imputed_data = imputer.fit_transform(numeric_df)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)
        
        # Determine optimal number of clusters using silhouette analysis
        silhouette_scores = []
        K = range(2, 6)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
            labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            silhouette_scores.append(score)
        
        optimal_clusters = K[np.argmax(silhouette_scores)]
        
        # Try multiple clustering techniques
        clustering_results = {}
        
        # K-Means
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=n_init)
        kmeans_labels = kmeans.fit_predict(scaled_data)
        clustering_results['kmeans'] = {
            'labels': kmeans_labels,
            'score': silhouette_score(scaled_data, kmeans_labels)
        }
        
        # DBSCAN with adaptive eps
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn_dist = nn.fit(scaled_data).kneighbors(scaled_data)[0]
            eps = max(np.percentile(nn_dist[:, 1], 90), 0.1)  # Ensure minimum eps value
            
            dbscan = DBSCAN(eps=eps, min_samples=min(5, len(scaled_data) // 20))
            dbscan_labels = dbscan.fit_predict(scaled_data)
            if len(set(dbscan_labels)) > 1:  # Only calculate score if meaningful clusters found
                clustering_results['dbscan'] = {
                    'labels': dbscan_labels,
                    'score': silhouette_score(scaled_data, dbscan_labels)
                }
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
        
        # Use best performing algorithm for visualization
        best_algorithm = max(clustering_results.items(), key=lambda x: x[1]['score'])
        clusters = best_algorithm[1]['labels']
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        plt.figure(figsize=(6.4, 6.4))
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
        plt.title(f'Cluster Analysis using {best_algorithm[0].upper()}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        cluster_path = os.path.join(self.output_dir, 'cluster_analysis.png')
        plt.savefig(cluster_path, dpi=80, bbox_inches='tight')
        plt.close()
        
        return {
            'cluster_labels': clusters.tolist(),
            'cluster_visualization': cluster_path,
            'optimal_clusters': optimal_clusters,
            'algorithm_comparison': {k: v['score'] for k, v in clustering_results.items()}
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
        Perform comprehensive data analysis and generate story.
        """
        try:
            # Suppress sklearn warnings
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # Generate basic analyses
            data_summary = self.generate_data_summary()
            correlation_results = self.detect_correlations()
            cluster_results = self.cluster_analysis(n_init=10)
            
            # Generate advanced statistical analysis
            advanced_stats = self.advanced_statistical_analysis()
            
            # Add advanced analysis results to data summary
            data_summary['advanced_statistics'] = advanced_stats
            
            # Generate story with enhanced insights
            story = self.generate_story(data_summary, correlation_results, cluster_results)
            
            # Generate additional visualizations
            self.analyze_categorical_patterns()
            self.visualize_statistics()
            
            print(f"Analysis complete. Check {self.readme_path} and generated images.")
            return True
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False
    
    def _generate_dynamic_prompt(self, insights):
        """
        Generate a two-part prompt: technical analysis and creative narrative
        """
        # Extract metrics
        total_rows = insights['data_overview']['size']
        missing_data = len(insights['missing_data'])
        num_clusters = insights['clusters']
        
        # Extract advanced statistics if available
        advanced_stats = insights.get('advanced_statistics', {})
        
        # Build the technical analysis prompt
        technical_prompt = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a data scientist presenting a comprehensive analysis. 
                    Focus on statistical insights, patterns, and create ASCII/Markdown tables and visualizations."""
                },               
                {
                    "role": "user",
                    "content": f"""
                    Dataset Analysis Report:
                    
                    Overview:
                    - Dataset Size: {total_rows}
                    - Missing Data Points: {missing_data}
                    - Identified Clusters: {num_clusters}
                    
                    Advanced Statistical Analysis:
                    - Distribution Fitting Results: {json.dumps({k: v['best_fitting_distribution'] for k, v in advanced_stats.items()}, indent=2)}
                    - Stationarity Tests: {json.dumps({k: v['is_stationary'] for k, v in advanced_stats.items()}, indent=2)}
                    - Outlier Analysis: {json.dumps({k: {'iqr_outliers': v['iqr_outliers'], 'z_score_outliers': v['z_score_outliers']} for k, v in advanced_stats.items()}, indent=2)}
                    - Distribution Characteristics: {json.dumps({k: {'skewness': v['skewness'], 'kurtosis': v['kurtosis']} for k, v in advanced_stats.items()}, indent=2)}
                    
                    Correlation Patterns:
                    {json.dumps(insights.get('correlations', {}), indent=2)}
                    
                    Please provide a detailed technical analysis including:
                    1. Dataset characteristics in a formatted Markdown table
                    2. Statistical significance summary with ASCII box plots where relevant
                    3. Correlation matrix as a formatted Markdown table
                    4. Cluster analysis summary with text-based visualization
                    5. Missing data patterns in tabular format
                    6. Key metrics dashboard using ASCII/Unicode characters
                    7. Distribution fitting analysis and implications
                    8. Outlier analysis and impact assessment
                    9. Stationarity test results and their significance
                    10. Potential biases or limitations
                    11. Actionable recommendations based on advanced statistics
                    
                    Use these formatting elements:
                    - Create tables using Markdown |---|---| syntax accurately
                    - Use Unicode box-drawing characters for simple visualizations
                    - Format key metrics in highlighted blocks
                    - Use bullet points and numbered lists for clarity
                    - Include ASCII art charts where appropriate
                    
                    Format everything in Markdown with clear sections."""
                }
            ]
        }
        
        # Get technical analysis first
        tech_response = self._make_api_call(technical_prompt)
        if tech_response.status_code != 200:
            raise Exception("Failed to generate technical analysis")
        
        technical_analysis = tech_response.json()['choices'][0]['message']['content']
        
        # Now generate the creative narrative with enhanced insights
        creative_prompt = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a quantum historian from 2075. 
                    Based on the technical analysis provided, create an engaging narrative."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Technical Analysis:
                    {technical_analysis}
                    
                    Transform this into a quantum narrative following these directives:
                    1. Frame data points as temporal travelers
                    2. Present correlations as quantum entanglements
                    3. Describe clusters as convergence points
                    4. Interpret distribution patterns as temporal waves
                    5. Frame outliers as temporal anomalies
                    6. Describe stationarity as temporal stability
                    7. Maintain scientific accuracy while being creative
                    
                    Use the existing quantum template format:
                    {insights['quantum_template']}"""
                }
            ]
        }
        
        return technical_analysis, creative_prompt

    
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
        and violin plots for numeric columns, with enhanced styling and context.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return None
        
        # Set style for better readability
        plt.style.use('default')  # Reset to default style first
        sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid theme
        sns.set_palette("husl")
        
        # Create figure with two rows: box plots and violin plots
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, 
                                      figsize=(8, 10))
        
        # Box plots with enhanced styling
        sns.boxplot(data=numeric_df, ax=ax1, whis=1.5)
        ax1.set_title('Distribution of Numeric Variables\n(Box Plots)', pad=20)
        ax1.set_xlabel('Variables', labelpad=10)
        ax1.set_ylabel('Values', labelpad=10)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add explanation text for box plot
        box_text = "Box Plot Legend:\n⎯ Max\n▢ Q3\n— Median\n▢ Q1\n⎯ Min\n• Outliers"
        ax1.text(1.15, 0.5, box_text, transform=ax1.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 verticalalignment='center')
        
        # Violin plots with enhanced styling
        sns.violinplot(data=numeric_df, ax=ax2, inner='quartile')
        ax2.set_title('Distribution Shape of Numeric Variables\n(Violin Plots)', pad=20)
        ax2.set_xlabel('Variables', labelpad=10)
        ax2.set_ylabel('Values', labelpad=10)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add explanation text for violin plot
        violin_text = "Violin Plot Shows:\n- Distribution shape\n- Data density\n- Quartiles\n- Full data range"
        ax2.text(1.15, 0.5, violin_text, transform=ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 verticalalignment='center')
        
        plt.tight_layout()
        stats_path = os.path.join(self.output_dir, 'statistical_summary.png')
        plt.savefig(stats_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return stats_path
    
    def analyze_categorical_patterns(self):
        """
        Analyze and visualize patterns in categorical columns with enhanced context.
        """
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            return None
        
        # Set style for better readability
        plt.style.use('default')  # Reset to default style first
        sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid theme
        sns.set_palette("husl")
        
        # Select top 4 categorical columns
        plot_cols = cat_cols[:4]
        n_cols = len(plot_cols)
        
        if n_cols == 0:
            return None
        
        # Create 2x2 grid
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        for idx, col in enumerate(plot_cols):
            try:
                # Calculate row and column position
                row = idx // 2
                col_pos = idx % 2
                
                # Create subplot with specific position
                ax = fig.add_subplot(gs[row, col_pos])
                
                # Get value counts and calculate percentages
                value_counts = self.df[col].value_counts().head(5)
                total = value_counts.sum()
                percentages = (value_counts / total * 100).round(1)
                
                # Create bar plot
                bars = sns.barplot(x=value_counts.values, 
                                 y=[str(x)[:20] for x in value_counts.index],
                                 ax=ax)
                
                # Add percentage labels on bars
                for i, (v, p) in enumerate(zip(value_counts.values, percentages)):
                    bars.text(v, i, f' {p}%', va='center')
                
                # Enhance title and labels
                ax.set_title(f'Top 5 Categories in\n{col[:25]}...' if len(col) > 25 else f'Top 5 Categories in {col}',
                            pad=20)
                ax.set_xlabel('Count', labelpad=10)
                
                # Add total count in subtitle
                ax.text(0.5, -0.2, f'Total unique values: {self.df[col].nunique()}',
                       ha='center', transform=ax.transAxes, style='italic')
                
            except Exception as e:
                print(f"Warning: Could not plot column {col}: {str(e)}")
                if idx < len(plot_cols):
                    ax = fig.add_subplot(gs[row, col_pos])
                    ax.text(0.5, 0.5, f'Could not plot {col}\nError: {str(e)}',
                           ha='center', va='center', wrap=True)
        
        # Add overall title
        fig.suptitle('Distribution of Top Categories in Categorical Variables\n', 
                     fontsize=14, y=1.02)
        
        # Add explanation text
        fig.text(0.5, -0.05, 
                 'Note: Showing top 5 categories for each variable. Percentages indicate proportion of total values.',
                 ha='center', style='italic', wrap=True)
        
        cat_path = os.path.join(self.output_dir, 'categorical_analysis.png')
        plt.savefig(cat_path, dpi=100, bbox_inches='tight', facecolor='white')
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
    
    def advanced_statistical_analysis(self):
        """
        Perform advanced statistical analysis with hypothesis testing
        and distribution fitting.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        results = {}
        
        for column in numeric_df.columns:
            data = numeric_df[column].dropna()
            
            # Distribution fitting
            distributions = [
                stats.norm, stats.gamma, stats.lognorm, 
                stats.exponweib, stats.beta
            ]
            
            best_dist = None
            best_params = None
            best_kstest = float('inf')
            
            for dist in distributions:
                try:
                    params = dist.fit(data)
                    _, p_value = stats.kstest(data, dist.name, params)
                    if p_value > best_kstest:
                        best_dist = dist.name
                        best_params = params
                        best_kstest = p_value
                except:
                    continue
            
            # Outlier detection using IQR and Z-score
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            iqr_outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
            
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > 3).sum()
            
            # Stationarity test
            try:
                adf_stat, adf_p = sm.tsa.stattools.adfuller(data)[:2]
            except:
                adf_stat, adf_p = None, None
            
            results[column] = {
                'best_fitting_distribution': best_dist,
                'distribution_p_value': best_kstest,
                'iqr_outliers': iqr_outliers,
                'z_score_outliers': z_outliers,
                'is_stationary': adf_p < 0.05 if adf_p is not None else None,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        
        return results

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