# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0.0",
#   "seaborn>=0.12.0",
#   "matplotlib>=3.7.0",
#   "openai>=1.0.0",
#   "scikit-learn>=1.3.0",
#   "requests>=2.31.0"
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
        
        # Set up single README path
        self.readme_path = os.path.join(os.path.dirname(csv_path), 'README.md')
    
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
        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        
        if len(numeric_df.columns) < 2:
            return {"error": "Not enough numeric columns for correlation analysis"}
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        corr_path = os.path.join(os.path.dirname(self.readme_path), 'correlation_heatmap.png')
        plt.savefig(corr_path)
        plt.close()
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "correlation_heatmap": corr_path
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
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
        plt.title('Cluster Analysis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        cluster_path = os.path.join(os.path.dirname(self.readme_path), 'cluster_analysis.png')
        plt.savefig(cluster_path)
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
        Perform comprehensive data analysis and generate story.
        """
        try:
            # Suppress sklearn warnings
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # Generate all analyses
            data_summary = self.generate_data_summary()
            correlation_results = self.detect_correlations()
            cluster_results = self.cluster_analysis(n_init=10)
            
            # Generate story (this will write to README.md)
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
        # Extract metrics (keep existing code)
        total_rows = insights['data_overview']['size']
        missing_data = len(insights['missing_data'])
        num_clusters = insights['clusters']
        
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
                    
                    Statistical Analysis:
                    {json.dumps(insights.get('statistical_analysis', {}), indent=2)}
                    
                    Correlation Patterns:
                    {json.dumps(insights.get('correlations', {}), indent=2)}
                    
                    Please provide a detailed technical analysis including:
                    1. Dataset characteristics in a formatted Markdown table
                    2. Statistical significance summary with ASCII box plots where relevant
                    3. Correlation matrix as a formatted Markdown table
                    4. Cluster analysis summary with text-based visualization
                    5. Missing data patterns in tabular format
                    6. Key metrics dashboard using ASCII/Unicode characters
                    7. Potential biases or limitations
                    8. Actionable recommendations
                    
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
        
        # Now generate the creative narrative
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
                    4. Maintain scientific accuracy while being creative
                    
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
        and violin plots for numeric columns.
        
        Returns:
            str: Path to saved visualization
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return None
        
        # Create figure with two rows: box plots and violin plots
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, 
                                      figsize=(12, 8))
        
        # Box plots
        sns.boxplot(data=numeric_df, ax=ax1)
        ax1.set_title('Box Plots of Numeric Variables')
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plots
        sns.violinplot(data=numeric_df, ax=ax2)
        ax2.set_title('Violin Plots of Numeric Variables')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        stats_path = os.path.join(os.path.dirname(self.readme_path), 'statistical_summary.png')
        plt.savefig(stats_path)
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
        
        # Select top 6 categorical columns (if more exist)
        plot_cols = cat_cols[:6]
        n_cols = len(plot_cols)
        
        if n_cols == 0:
            return None
        
        # Create figure and subplots with proper dimensions
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(nrows=n_rows, ncols=min(2, n_cols), 
                                figsize=(15, 4 * n_rows))
        
        # Convert axes to array for consistent indexing
        if n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = np.array([axes])
        else:
            axes = axes.reshape(-1)
        
        for idx, col in enumerate(plot_cols):
            try:
                # Get value counts and limit to top 10
                value_counts = self.df[col].value_counts().head(10)
                
                # Create bar plot
                if len(value_counts) > 0:
                    sns.barplot(x=value_counts.values, 
                              y=[str(x)[:50] for x in value_counts.index],  # Truncate long labels
                              ax=axes[idx])
                    
                    axes[idx].set_title(f'Top {len(value_counts)} Values in {col}')
                    axes[idx].set_xlabel('Count')
                    
                    # Rotate labels if they're too long
                    if value_counts.index.str.len().max() > 20:
                        axes[idx].tick_params(axis='y', labelrotation=0)
            except Exception as e:
                print(f"Warning: Could not plot column {col}: {str(e)}")
                axes[idx].text(0.5, 0.5, f'Could not plot {col}',
                             ha='center', va='center')
        
        # Remove empty subplots if any
        for idx in range(len(plot_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        cat_path = os.path.join(os.path.dirname(self.readme_path), 'categorical_analysis.png')
        plt.savefig(cat_path)
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