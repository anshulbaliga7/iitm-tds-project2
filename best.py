# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "scipy"
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
        
        self.csv_path = csv_path
        self.output_dir = os.path.join(os.path.dirname(csv_path), 'analysis_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up OpenAI client
        openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDI3NDNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.i6MpRliZ3nPhSAQ_bOkOW-isk4R9iXZY3cM-3AuFi3o"
    
    def generate_data_summary(self):
        """
        Generate a comprehensive summary of the dataset.
        
        Returns:
            dict: A summary of dataset characteristics
        """
        try:
            # Basic dataset info
            summary = {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "columns": list(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "missing_values": self.df.isnull().sum().to_dict(),
                "duplicates": self.df.duplicated().sum()
            }
            
            # Numeric columns analysis
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
            
            # Categorical columns analysis
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                summary["categorical_summary"] = {
                    col: {
                        "unique_values": self.df[col].nunique(),
                        "top_values": self.df[col].value_counts().head(5).to_dict()
                    } for col in cat_cols
                }
            
            # Date columns detection and analysis
            date_cols = []
            for col in self.df.columns:
                try:
                    pd.to_datetime(self.df[col])
                    date_cols.append(col)
                except:
                    continue
            
            if date_cols:
                summary["date_columns"] = {
                    col: {
                        "min_date": str(pd.to_datetime(self.df[col]).min()),
                        "max_date": str(pd.to_datetime(self.df[col]).max())
                    } for col in date_cols
                }
            
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
        corr_path = 'correlation_heatmap.png'
        plt.savefig(corr_path)
        plt.close()
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "correlation_heatmap": corr_path
        }
    
    def cluster_analysis(self):
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
        kmeans = KMeans(n_clusters=3, random_state=42)
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
        
        cluster_path = 'cluster_analysis.png'
        plt.savefig(cluster_path)
        plt.close()
        
        return {
            "cluster_labels": clusters.tolist(),
            "cluster_visualization": cluster_path
        }
    
    def generate_story(self, data_summary, correlation_results, cluster_results):
        """
        Generate a narrative story about the data analysis.
        """
        try:
            # Prepare analysis insights
            insights = {
                "data_overview": {
                    "size": f"{data_summary['total_rows']} rows × {data_summary['total_columns']} columns",
                    "memory": data_summary['memory_usage'],
                    "duplicates": data_summary['duplicates']
                },
                "missing_data": {
                    col: count for col, count in data_summary['missing_values'].items() if count > 0
                },
                "correlations": correlation_results.get('correlation_matrix', {}),
                "clusters": len(set(cluster_results.get('cluster_labels', []))) if 'cluster_labels' in cluster_results else 0
            }
            
            # Generate dynamic prompt based on data characteristics
            prompt = self._generate_dynamic_prompt(insights)
            
            # Make API call with optimized data
            response = self._make_api_call(prompt)
            
            if response.status_code == 200:
                story = response.json()['choices'][0]['message']['content']
                
                # Save story with visualizations
                output_path = 'C:\\Users\\bali\\OneDrive - PESUNIVERSITY\\Documents\\iitmbs\\iitm-tds-project2\\README.md'
                with open(output_path, 'w') as f:
                    f.write(story)
                    
                    # Add visualization references if they exist
                    if os.path.exists('correlation_heatmap.png'):
                        f.write('\n\n## Correlation Analysis\n![Correlation Heatmap](correlation_heatmap.png)\n')
                    if os.path.exists('cluster_analysis.png'):
                        f.write('\n\n## Cluster Analysis\n![Cluster Analysis](cluster_analysis.png)\n')
                
                return story
            else:
                raise Exception(f"API request failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Story generation error: {e}")
            return "# Data Analysis Story\n\nUnable to generate narrative due to an error."
    
    def analyze(self):
        """
        Perform comprehensive data analysis and generate story.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Generate all analyses
        data_summary = self.generate_data_summary()
        correlation_results = self.detect_correlations()
        cluster_results = self.cluster_analysis()
        
        # Generate story
        story = self.generate_story(data_summary, correlation_results, cluster_results)
        
        # Generate additional visualizations
        self.visualize_distributions()  # Distribution plot
        self.visualize_statistics()     # Box/Violin plot
        
        # Save story with all visualizations
        output_path = os.path.join(self.output_dir, 'README.md')
        with open(output_path, 'w') as f:
            f.write(story)
            
            # Core visualizations
            if os.path.exists('correlation_heatmap.png'):
                f.write('\n\n## Correlation Analysis\n![Correlation Heatmap](correlation_heatmap.png)\n')
            if os.path.exists('cluster_analysis.png'):
                f.write('\n\n## Cluster Analysis\n![Cluster Analysis](cluster_analysis.png)\n')
            
            # Additional visualizations
            if os.path.exists('distribution_plot.png'):
                f.write('\n\n## Distribution Analysis\n![Distribution Analysis](distribution_plot.png)\n')
            if os.path.exists('statistical_summary.png'):
                f.write('\n\n## Statistical Summary\n![Statistical Summary](statistical_summary.png)\n')
    
    def _generate_dynamic_prompt(self, insights):
        """
        Generate a dynamic prompt based on data characteristics.
        
        Args:
            insights (dict): Dictionary containing analysis insights
        
        Returns:
            dict: OpenAI API request payload
        """
        # Extract key metrics
        total_rows = insights['data_overview']['size']
        missing_data = len(insights['missing_data'])
        num_clusters = insights['clusters']
        
        # Build the prompt template
        prompt = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a quantum historian and data chronicler from the year 2075, tasked with decoding the secret journeys of sentient data across the fabric of space-time.
                    Your mission: Construct a compelling narrative that merges imaginative storytelling with precise analytical insights, revealing the interconnected journeys of data across temporal dimensions.
                    
                    Transmission Protocol:
                    - Each data point is a living traveler with a unique quantum signature.
                    - Clusters are dimensional nexus points where timelines intersect, fostering innovation or historical echoes.
                    - Dimensional discontinuities signify missing or obscured data with implications for the broader quantum network.
                    - Narratives should seamlessly integrate visuals as storytelling artifacts while exposing the causal networks within the data."""
                },
                {
                    "role": "user",
                    "content": f"""Temporal Reconnaissance Mission:

                    Quantum Data Manifold Parameters:
                    - Total Temporal Vectors: {total_rows}
                    - Dimensional Discontinuities: {missing_data} quantum nodes with unresolved trajectories.
                    - Convergence Zones: {num_clusters} identified temporal clusters.

                    Mission Directives:
                    1. Construct a chronological narrative of data migrations, highlighting the unique paths of data travelers.
                    2. Decode the quantum correlations and causal relationships between interdimensional variables.
                    3. Analyze the significance of temporal convergence zones and their roles in the broader quantum tapestry.
                    4. Unveil hidden causality networks, showing how disparate data travelers influence each other and the timeline fabric.
                    5. Generate predictive insights about the future trajectories of data travelers, pushing beyond current computational limitations.

                    Transmission Requirements:
                    - Use Markdown to encrypt the quantum narrative.
                    - Blend imaginative storytelling with analytical clarity to engage diverse audiences.
                    - Integrate at least two key visuals (e.g., heatmaps, cluster diagrams) as pivotal elements of the narrative.
                    - Prioritize a revelatory approach that transcends conventional data analysis and explores the sentience of data."""
                }
            ]
        }
        return prompt

    
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
    
    def analyze_distributions(self):
        """
        Analyze and visualize distributions of numeric columns.
        
        Returns:
            str: Path to saved visualization
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return None
        
        # Create subplots for each numeric column
        n_cols = len(numeric_df.columns)
        fig, axes = plt.subplots(nrows=(n_cols+1)//2, ncols=2, figsize=(15, 4*((n_cols+1)//2)))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_df.columns):
            # Combine histogram and KDE plot
            sns.histplot(data=numeric_df[col].dropna(), kde=True, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Count')
        
        # Remove empty subplots if odd number of columns
        if n_cols % 2 != 0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        dist_path = os.path.join(self.output_dir, 'distribution_analysis.png')
        plt.savefig(dist_path)
        plt.close()
        
        return dist_path
    
    def analyze_temporal_patterns(self):
        """
        Analyze and visualize temporal patterns if date columns exist.
        
        Returns:
            str: Path to saved visualization
        """
        # Find date columns
        date_cols = []
        for col in self.df.columns:
            try:
                pd.to_datetime(self.df[col])
                date_cols.append(col)
            except:
                continue
        
        if not date_cols:
            return None
        
        # Select first date column and numeric columns
        date_col = date_cols[0]
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Create time series plot
        plt.figure(figsize=(12, 6))
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        
        # Plot time series for up to 3 numeric columns
        for col in numeric_cols[:3]:
            df_temp.groupby(df_temp[date_col].dt.to_period('M'))[col].mean().plot(label=col)
        
        plt.title('Temporal Patterns Analysis')
        plt.xlabel('Time')
        plt.ylabel('Average Value')
        plt.legend()
        plt.grid(True)
        
        time_path = os.path.join(self.output_dir, 'temporal_analysis.png')
        plt.savefig(time_path)
        plt.close()
        
        return time_path
    
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
        
        # Create subplots
        fig, axes = plt.subplots(nrows=(n_cols+1)//2, ncols=2, 
                                figsize=(15, 4*((n_cols+1)//2)))
        if n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(plot_cols):
            # Get value counts and plot
            value_counts = self.df[col].value_counts()
            sns.barplot(x=value_counts.values[:10], 
                       y=value_counts.index[:10], 
                       ax=axes[idx])
            axes[idx].set_title(f'Top 10 Values in {col}')
            axes[idx].set_xlabel('Count')
        
        # Remove empty subplots
        if n_cols % 2 != 0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        cat_path = os.path.join(self.output_dir, 'categorical_analysis.png')
        plt.savefig(cat_path)
        plt.close()
        
        return cat_path
    
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
        stats_path = os.path.join(self.output_dir, 'statistical_summary.png')
        plt.savefig(stats_path)
        plt.close()
        
        return stats_path
    
    def visualize_missing_data(self):
        """
        Create visualization of missing data patterns.
        
        Returns:
            str: Path to saved visualization
        """
        # Calculate missing values
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) == 0:
            return None
        
        # Create missing data visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.values, y=missing.index)
        plt.title('Missing Values by Column')
        plt.xlabel('Number of Missing Values')
        
        missing_path = os.path.join(self.output_dir, 'missing_data.png')
        plt.savefig(missing_path)
        plt.close()
        
        return missing_path
    
    def visualize_distributions(self):
        """
        Create distribution plots for numeric columns.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return
        
        # Create distribution plots
        plt.figure(figsize=(12, 6))
        for col in numeric_df.columns:
            sns.kdeplot(data=numeric_df[col].dropna(), label=col)
        
        plt.title('Distribution of Numeric Variables')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('distribution_plot.png')
        plt.close()
    
    def visualize_statistics(self):
        """
        Create box plots and violin plots for numeric columns.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) == 0:
            return
        
        # Create figure with box plots
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=numeric_df)
        plt.title('Statistical Summary of Numeric Variables')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('statistical_summary.png')
        plt.close()

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