"""
====================================================================
COMPLETE STUDENT SCORE ANALYZER & GRADE PREDICTOR
AI-201 Lab Project - Production Version (FIXED)
====================================================================
Team: Batool Binte Fazal (2024140), Rida Syed (2024540)

Features:
- OOP Design (6 classes)
- 3 ML Models (Linear Regression, Logistic Regression, KNN)
- Advanced NumPy operations (z-scores, percentiles, polynomial features)
- Pandas analysis (GroupBy, Q&A style)
- 4 Matplotlib visualizations
- Exception handling throughout
- Model persistence (Pickle)
- Streamlit interface
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ====================================================================
# CLASS 1: DATA CLEANER
# ====================================================================
class DataCleaner:
    """
    Handles data cleaning, validation, and encoding with exception handling
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.required_columns = ['math score', 'reading score', 'writing score']
    
    def clean_data(self, df):
        """Clean dataset: remove nulls, duplicates, validate columns"""
        try:
            print("🧹 Starting data cleaning...")
            
            initial_shape = df.shape
            
            # Validate required columns exist
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove null values
            df_clean = df.dropna()
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            
            # Reset index
            df_clean = df_clean.reset_index(drop=True)
            
            final_shape = df_clean.shape
            
            print(f"   Initial shape: {initial_shape}")
            print(f"   Final shape: {final_shape}")
            print(f"   Removed: {initial_shape[0] - final_shape[0]} rows")
            print("   ✅ Data cleaning complete!\n")
            
            return df_clean
            
        except ValueError as ve:
            print(f"   ❌ Validation Error: {ve}")
            raise
        except Exception as e:
            print(f"   ❌ Cleaning Error: {e}")
            raise
    
    def encode_categorical(self, df):
        """Encode categorical variables using LabelEncoder"""
        try:
            print("🔢 Encoding categorical variables...")
            
            categorical_cols = df.select_dtypes(include='object').columns
            
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded: {col}")
            
            print("   ✅ Encoding complete!\n")
            return df
            
        except Exception as e:
            print(f"   ❌ Encoding Error: {e}")
            raise


# ====================================================================
# CLASS 2: FEATURE ENGINEER (NumPy Heavy)
# ====================================================================
class FeatureEngineer:
    """
    Creates advanced features using NumPy operations
    """
    
    def __init__(self):
        self.features_created = []
    
    def create_all_features(self, df):
        """Create all engineered features"""
        try:
            print("⚙️ Engineering features with NumPy...")
            
            df = self.create_basic_features(df)
            df = self.create_performance_index(df)
            df = self.create_polynomial_features(df)
            df = self.detect_outliers_zscore(df)
            df, percentiles = self.calculate_percentile_ranks(df)
            df = self.calculate_risk_score(df)
            
            print(f"   ✅ Created {len(self.features_created)} new features!\n")
            return df, percentiles
            
        except Exception as e:
            print(f"   ❌ Feature Engineering Error: {e}")
            raise
    
    def create_basic_features(self, df):
        """Create basic derived features"""
        try:
            # Average score
            df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
            self.features_created.append('average_score')
            
            # Study efficiency (if study time exists)
            if 'study time' in df.columns:
                df['study_efficiency'] = df['average_score'] / (df['study time'].astype(float) + 1)
                df['study_engagement'] = df['study time'].astype(float) * (df['average_score'] / 100)
                self.features_created.extend(['study_efficiency', 'study_engagement'])
            
            print("   ✓ Basic features created")
            return df
            
        except Exception as e:
            raise Exception(f"Basic features error: {e}")
    
    def create_performance_index(self, df):
        """Performance Index using NumPy: 0.5*math + 0.3*reading + 0.2*writing"""
        try:
            math_scores = df['math score'].values.astype(float)
            reading_scores = df['reading score'].values.astype(float)
            writing_scores = df['writing score'].values.astype(float)
            
            # Weighted performance index using NumPy
            performance_index = (
                0.5 * math_scores +
                0.3 * reading_scores +
                0.2 * writing_scores
            )
            
            df['performance_index'] = performance_index
            self.features_created.append('performance_index')
            
            print("   ✓ Performance Index created")
            return df
            
        except Exception as e:
            raise Exception(f"Performance index error: {e}")
    
    def create_polynomial_features(self, df):
        """Create polynomial interaction features using NumPy"""
        try:
            if 'study time' in df.columns and 'parental level of education' in df.columns:
                study = df['study time'].values.astype(float)
                parent_edu = df['parental level of education'].values.astype(float)
                
                # Polynomial interactions
                df['study_parent_interaction'] = study * parent_edu
                df['study_squared'] = np.power(study, 2)
                
                self.features_created.extend(['study_parent_interaction', 'study_squared'])
                print("   ✓ Polynomial features created")
            
            return df
            
        except Exception as e:
            raise Exception(f"Polynomial features error: {e}")
    
    def detect_outliers_zscore(self, df):
        """Detect outliers using Z-score method (NumPy)"""
        try:
            subjects = ['math score', 'reading score', 'writing score']
            
            for subject in subjects:
                values = df[subject].values.astype(float)
                
                # Calculate z-scores using NumPy
                mean = np.mean(values)
                std = np.std(values)
                
                if std == 0:
                    z_scores = np.zeros_like(values)
                else:
                    z_scores = (values - mean) / std
                
                # Mark outliers (|z| > 2)
                df[f'{subject}_zscore'] = z_scores
                df[f'{subject}_outlier'] = np.abs(z_scores) > 2
            
            # Overall outlier flag
            df['is_outlier'] = (
                df['math score_outlier'] | 
                df['reading score_outlier'] | 
                df['writing score_outlier']
            )
            
            self.features_created.append('is_outlier')
            print("   ✓ Z-score outliers detected")
            return df
            
        except Exception as e:
            raise Exception(f"Outlier detection error: {e}")
    
    def calculate_percentile_ranks(self, df):
        """Calculate percentile rankings using NumPy"""
        try:
            scores = df['average_score'].values.astype(float)
            
            # Calculate percentile rank for each student
            percentile_ranks = np.array([
                np.sum(scores <= score) / len(scores) * 100 
                for score in scores
            ])
            
            df['percentile_rank'] = percentile_ranks
            
            # Calculate specific percentiles
            percentiles = {
                '25th': float(np.percentile(scores, 25)),
                '50th': float(np.percentile(scores, 50)),
                '75th': float(np.percentile(scores, 75)),
                '90th': float(np.percentile(scores, 90))
            }
            
            self.features_created.append('percentile_rank')
            print("   ✓ Percentile ranks calculated")
            return df, percentiles
            
        except Exception as e:
            raise Exception(f"Percentile calculation error: {e}")
    
    def calculate_risk_score(self, df):
        """Calculate custom risk score using NumPy aggregation"""
        try:
            avg_score = df['average_score'].values.astype(float)
            study_time = df['study time'].values.astype(float) if 'study time' in df.columns else np.ones(len(df))
            
            # Normalize using NumPy
            avg_norm = (avg_score - np.min(avg_score)) / (np.max(avg_score) - np.min(avg_score) + 1e-10)
            study_norm = (study_time - np.min(study_time)) / (np.max(study_time) - np.min(study_time) + 1e-10)
            
            # Risk score (higher = lower risk)
            risk_score = (0.7 * avg_norm + 0.3 * study_norm) * 100
            
            df['risk_score'] = risk_score
            
            # Risk categories using NumPy select
            risk_categories = np.select(
                [risk_score >= 70, risk_score >= 50, risk_score >= 30],
                ['Low Risk', 'Medium Risk', 'High Risk'],
                default='Critical Risk'
            )
            
            df['risk_category'] = risk_categories
            self.features_created.extend(['risk_score', 'risk_category'])
            
            print("   ✓ Risk scores calculated")
            return df
            
        except Exception as e:
            raise Exception(f"Risk score error: {e}")


# ====================================================================
# CLASS 3: VISUALIZER (Matplotlib - 4 Plots)
# ====================================================================
class Visualizer:
    """
    Creates 4 matplotlib visualizations with advanced styling
    """
    
    def __init__(self):
        plt.style.use('default')
        self.colors_gradient = plt.cm.viridis
    
    def plot_histogram(self, df, subject='math score'):
        """Plot 1: Histogram with gradient colors"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            scores = df[subject].values.astype(float)
            
            n, bins, patches = ax.hist(scores, bins=20, edgecolor='black', linewidth=1.2, alpha=0.85)
            
            # Gradient coloring
            fracs = n / n.max()
            norm = plt.Normalize(fracs.min(), fracs.max())
            for frac, patch in zip(fracs, patches):
                patch.set_facecolor(self.colors_gradient(norm(frac)))
            
            # Statistics
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2.5, 
                      label=f'Mean: {mean_score:.1f}')
            ax.axvline(median_score, color='orange', linestyle='--', linewidth=2.5, 
                      label=f'Median: {median_score:.1f}')
            
            ax.set_xlabel(subject.title(), fontsize=13, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
            ax.set_title(f'Score Distribution - {subject.title()}', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, shadow=True)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"❌ Histogram error: {e}")
            return None
    
    def plot_bar_chart(self, df):
        """Plot 2: Bar chart comparing subjects"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            subjects = ['math score', 'reading score', 'writing score']
            avg_scores = [df[s].astype(float).mean() for s in subjects]
            std_scores = [df[s].astype(float).std() for s in subjects]
            labels = ['📐 Math', '📚 Reading', '✍️ Writing']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = ax.bar(labels, avg_scores, color=colors, alpha=0.85, 
                         edgecolor='black', linewidth=2, yerr=std_scores, 
                         capsize=7, error_kw={'linewidth': 2})
            
            # Value labels on bars
            for bar, score in zip(bars, avg_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.1f}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            
            # Overall average line
            overall_avg = np.mean(avg_scores)
            ax.axhline(overall_avg, color='green', linestyle='--', linewidth=2, 
                      label=f'Overall Average: {overall_avg:.1f}')
            
            ax.set_ylabel('Average Score', fontsize=13, fontweight='bold')
            ax.set_title('Subject Performance Comparison', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, shadow=True)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"❌ Bar chart error: {e}")
            return None
    
    def plot_pie_chart(self, df, column='gender'):
        """Plot 4: Pie chart for demographics"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            counts = df[column].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            explode = [0.1 if i == 0 else 0 for i in range(len(counts))]
            
            # Donut pie chart
            wedges, texts, autotexts = ax1.pie(
                counts.values, labels=counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors[:len(counts)], explode=explode,
                shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'},
                pctdistance=0.85
            )
            
            # Make donut
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax1.add_artist(centre_circle)
            
            for autotext in autotexts:
                autotext.set_color('white')
            
            ax1.set_title(f'{column.title()} Distribution (Donut Chart)', 
                         fontsize=14, fontweight='bold')
            
            # Horizontal bar chart (alternative view)
            ax2.barh(counts.index, counts.values, color=colors[:len(counts)], 
                    alpha=0.85, edgecolor='black', linewidth=2)
            ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title(f'{column.title()} Distribution (Bar View)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add count labels
            for i, (idx, val) in enumerate(counts.items()):
                ax2.text(val, i, f' {val}', va='center', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"❌ Pie chart error: {e}")
            return None
    
    def plot_3d_scatter(self, df):
        """Plot 6: 3D scatter (ADVANCED)"""
        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get numeric data
            if 'study time' in df.columns:
                x = df['study time'].values.astype(float)
            else:
                x = np.random.randint(1, 6, len(df))
            
            if 'parental level of education' in df.columns:
                y = df['parental level of education'].values.astype(float)
            else:
                y = np.random.randint(0, 5, len(df))
            
            z = df['average_score'].values.astype(float)
            
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=100, alpha=0.7, 
                               edgecolors='black', linewidth=0.5)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Average Score', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
            
            # Annotate highest and lowest performers
            max_idx = np.argmax(z)
            min_idx = np.argmin(z)
            
            ax.text(x[max_idx], y[max_idx], z[max_idx], 
                   f'Highest\n{z[max_idx]:.1f}', fontsize=10, color='green', fontweight='bold')
            ax.text(x[min_idx], y[min_idx], z[min_idx], 
                   f'Lowest\n{z[min_idx]:.1f}', fontsize=10, color='red', fontweight='bold')
            
            # Calculate correlations
            corr_x = np.corrcoef(x, z)[0, 1]
            corr_y = np.corrcoef(y, z)[0, 1]
            
            # Stats box
            stats_text = f'Correlations:\nStudy Time: {corr_x:.3f}\nParent Edu: {corr_y:.3f}'
            ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Study Time', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_ylabel('Parental Education Level', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_zlabel('Average Score', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_title('3D Performance Analysis\nStudy Time × Parental Education × Score', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Enhanced grid
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"❌ 3D plot error: {e}")
            return None
    
    def generate_all_plots(self, df):
        """Generate all 4 plots"""
        try:
            print("📊 Generating visualizations...\n")
            plots = {}
            
            plots['histogram'] = self.plot_histogram(df)
            plots['bar_chart'] = self.plot_bar_chart(df)
            plots['pie_chart'] = self.plot_pie_chart(df)
            plots['3d_scatter'] = self.plot_3d_scatter(df)
            
            print("✅ All visualizations complete!\n")
            return plots
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return {}


# ====================================================================
# CLASS 4: MODEL TRAINER (3 Models Only)
# ====================================================================
class ModelTrainer:
    """
    Trains 3 models: Linear Regression, Logistic Regression, KNN
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
    
    def prepare_data(self, df, target='math score'):
        """Prepare features and target"""
        try:
            print("🔧 Preparing data for modeling...")
            
            # Select features (exclude target and derived columns)
            exclude_cols = ['math score', 'reading score', 'writing score', 
                          'average_score', 'performance_index', 'risk_score', 
                          'risk_category', 'percentile_rank', 'is_outlier']
            
            # Also exclude zscore and outlier columns
            exclude_cols.extend([col for col in df.columns if '_zscore' in col or '_outlier' in col])
            exclude_cols = [col for col in exclude_cols if col in df.columns]
            
            X = df.drop(columns=exclude_cols, errors='ignore')
            
            # Remove string columns and convert to numeric
            X = X.select_dtypes(include=[np.number])
            
            # Ensure all data is float
            X = X.astype(float)
            
            y = df[target].astype(float)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            print(f"   Features: {X.shape[1]}")
            print(f"   Samples: {X.shape[0]}")
            print("   ✅ Data prepared!\n")
            
            return X_scaled, y, X.columns.tolist()
            
        except Exception as e:
            print(f"   ❌ Data preparation error: {e}")
            raise
    
    def train_linear_regression(self, X, y):
        """Model 1: Linear Regression for score prediction"""
        try:
            print("🤖 Training Model 1: Linear Regression...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = model.score(X_test, y_test)
            
            self.models['linear_regression'] = model
            self.results['linear_regression'] = {'RMSE': float(rmse), 'R²': float(r2)}
            
            print(f"   RMSE: {rmse:.2f}")
            print(f"   R² Score: {r2:.3f}")
            print("   ✅ Linear Regression trained!\n")
            
            return model, rmse, r2
            
        except Exception as e:
            print(f"   ❌ Linear Regression error: {e}")
            raise
    
    def train_logistic_regression(self, X, y, threshold=50):
        """Model 2: Logistic Regression for pass/fail classification"""
        try:
            print("🤖 Training Model 2: Logistic Regression (Pass/Fail)...")
            
            # Create binary target (pass/fail)
            y_binary = (y >= threshold).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=0.2, random_state=42
            )
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            self.models['logistic_regression'] = model
            self.results['logistic_regression'] = {'Accuracy': float(accuracy), 'Threshold': threshold}
            
            print(f"   Accuracy: {accuracy*100:.2f}%")
            print(f"   Pass/Fail Threshold: {threshold}")
            print("   ✅ Logistic Regression trained!\n")
            
            return model, accuracy
            
        except Exception as e:
            print(f"   ❌ Logistic Regression error: {e}")
            raise
    
    def train_knn(self, X, y, threshold=50, n_neighbors=5):
        """Model 3: KNN for risk assessment"""
        try:
            print("🤖 Training Model 3: KNN (Risk Assessment)...")
            
            # Ensure y is numeric
            y_numeric = pd.to_numeric(y, errors='coerce')
            
            # Remove any NaN values
            valid_mask = ~y_numeric.isna()
            X_valid = X[valid_mask]
            y_valid = y_numeric[valid_mask]
            
            # Create risk categories based on score using numpy
            risk_labels = []
            for score in y_valid:
                if score < 50:
                    risk_labels.append('High Risk')
                elif score < 70:
                    risk_labels.append('Medium Risk')
                else:
                    risk_labels.append('Low Risk')
            
            y_risk = pd.Series(risk_labels, index=X_valid.index)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_risk, test_size=0.2, random_state=42
            )
            
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            self.models['knn'] = model
            self.results['knn'] = {'Accuracy': float(accuracy), 'K': n_neighbors}
            
            print(f"   Accuracy: {accuracy*100:.2f}%")
            print(f"   K Neighbors: {n_neighbors}")
            print("   ✅ KNN trained!\n")
            
            return model, accuracy
            
        except Exception as e:
            print(f"   ❌ KNN error: {e}")
            raise
    
    def train_all_models(self, df, target='math score'):
        """Train all 3 models"""
        try:
            X, y, feature_names = self.prepare_data(df, target)
            
            self.train_linear_regression(X, y)
            self.train_logistic_regression(X, y)
            self.train_knn(X, y)
            
            print("=" * 60)
            print("✅ ALL 3 MODELS TRAINED SUCCESSFULLY!")
            print("=" * 60 + "\n")
            
            return self.models, self.results, feature_names
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            raise


# ====================================================================
# CLASS 5: PREDICTOR
# ====================================================================
class Predictor:
    """
    Handles real-time predictions using trained models
    """
    
    def __init__(self, models, scaler):
        self.models = models
        self.scaler = scaler
    
    def predict_score(self, input_data):
        """Predict score using Linear Regression"""
        try:
            input_scaled = self.scaler.transform([input_data])
            prediction = self.models['linear_regression'].predict(input_scaled)[0]
            return round(float(prediction), 2)
        except Exception as e:
            print(f"❌ Score prediction error: {e}")
            return None
    
    def predict_pass_fail(self, input_data):
        """Predict pass/fail using Logistic Regression"""
        try:
            input_scaled = self.scaler.transform([input_data])
            prediction = self.models['logistic_regression'].predict(input_scaled)[0]
            return "Pass" if prediction == 1 else "Fail"
        except Exception as e:
            print(f"❌ Pass/fail prediction error: {e}")
            return None
    
    def predict_risk(self, input_data):
        """Predict risk using KNN"""
        try:
            input_scaled = self.scaler.transform([input_data])
            prediction = self.models['knn'].predict(input_scaled)[0]
            return prediction
        except Exception as e:
            print(f"❌ Risk prediction error: {e}")
            return None
    
    def get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


# ====================================================================
# CLASS 6: MODEL PERSISTENCE (Pickle)
# ====================================================================
class ModelPersistence:
    """
    Save and load trained models using Pickle
    """
    
    def __init__(self):
        self.model_dir = Path("saved_models")
        self.model_dir.mkdir(exist_ok=True)
    
    def save_models(self, models, scaler, feature_names):
        """Save all models and scaler"""
        try:
            print("💾 Saving models...")
            
            # Save each model
            for name, model in models.items():
                model_path = self.model_dir / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   ✓ Saved: {name}")
            
            # Save scaler
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print("   ✓ Saved: scaler")
            
            # Save feature names
            features_path = self.model_dir / "feature_names.pkl"
            with open(features_path, 'wb') as f:
                pickle.dump(feature_names, f)
            print("   ✓ Saved: feature names")
            
            print("✅ All models saved!\n")
            return True
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def load_models(self):
        """Load all saved models"""
        try:
            print("📥 Loading saved models...")
            
            models = {}
            
            # Load each model
            for model_file in self.model_dir.glob("*.pkl"):
                if model_file.stem not in ['scaler', 'feature_names']:
                    with open(model_file, 'rb') as f:
                        models[model_file.stem] = pickle.load(f)
                    print(f"   ✓ Loaded: {model_file.stem}")
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print("   ✓ Loaded: scaler")
            else:
                scaler = None
            
            # Load feature names
            features_path = self.model_dir / "feature_names.pkl"
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    feature_names = pickle.load(f)
                print("   ✓ Loaded: feature names")
            else:
                feature_names = None
            
            print("✅ All models loaded!\n")
            return models, scaler, feature_names
            
        except Exception as e:
            print(f"❌ Load error: {e}")
            return None, None, None
    
    def list_saved_models(self):
        """List all saved models"""
        try:
            models = [f.stem for f in self.model_dir.glob("*.pkl") 
                     if f.stem not in ['scaler', 'feature_names']]
            return models
        except Exception as e:
            print(f"❌ List error: {e}")
            return []


# ====================================================================
# PANDAS ANALYSIS MODULE
# ====================================================================
class PandasAnalyzer:
    """
    Advanced Pandas operations - Q&A style analysis
    """
    
    def __init__(self, df):
        self.df = df
    
    def answer_research_questions(self):
        """Answer research questions using Pandas (LA Jobs style)"""
        try:
            print("🔍 Pandas Q&A Analysis...\n")
            qa_results = {}
            
            # Q1: Which gender performs better?
            if 'gender' in self.df.columns:
                q1 = self.df.groupby('gender')['average_score'].mean()
                best_gender = q1.idxmax()
                qa_results['Q1: Best performing gender'] = f"{best_gender} (Avg: {q1[best_gender]:.2f})"
                print(f"   Q1: {best_gender} performs best with average {q1[best_gender]:.2f}")
            
            # Q2: Does test prep help?
            if 'test preparation course' in self.df.columns:
                q2 = self.df.groupby('test preparation course')['average_score'].mean()
                if len(q2) > 1:
                    improvement = q2.iloc[1] - q2.iloc[0]
                    qa_results['Q2: Test prep impact'] = f"+{improvement:.2f} points"
                    print(f"   Q2: Test prep improves scores by {improvement:.2f} points")
            
            # Q3: Study time correlation
            if 'study time' in self.df.columns:
                q3 = self.df[['study time', 'average_score']].corr().iloc[0, 1]
                qa_results['Q3: Study time correlation'] = f"{q3:.3f}"
                print(f"   Q3: Study time correlation: {q3:.3f}")
            
            # Q4: Best parental education level
            if 'parental level of education' in self.df.columns:
                q4 = self.df.groupby('parental level of education')['average_score'].mean()
                best_edu = q4.idxmax()
                qa_results['Q4: Best parental education'] = f"{best_edu} (Avg: {q4[best_edu]:.2f})"
                print(f"   Q4: Best parental education: {best_edu}")
            
            # Q5: Subject with highest scores
            subjects = ['math score', 'reading score', 'writing score']
            subject_means = {s: self.df[s].mean() for s in subjects}
            best_subject = max(subject_means, key=subject_means.get)
            qa_results['Q5: Highest scoring subject'] = f"{best_subject}: {subject_means[best_subject]:.2f}"
            print(f"   Q5: Highest scoring subject: {best_subject}\n")
            
            print("✅ Q&A Analysis complete!\n")
            return qa_results
            
        except Exception as e:
            print(f"❌ Q&A Analysis error: {e}")
            return {}
    
    def demographic_analysis(self):
        """Multi-level groupby analysis"""
        try:
            if 'gender' in self.df.columns and 'parental level of education' in self.df.columns:
                analysis = self.df.groupby(['gender', 'parental level of education']).agg({
                    'math score': 'mean',
                    'reading score': 'mean',
                    'writing score': 'mean',
                    'average_score': ['mean', 'count']
                }).round(2)
                
                return analysis
            return None
        except Exception as e:
            print(f"❌ Demographic analysis error: {e}")
            return None


# ====================================================================
# STREAMLIT APPLICATION
# ====================================================================
def main():
    """Main Streamlit application"""
    
    # Page config
    st.set_page_config(
        page_title="Student Score Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Title
    st.title("📊 Student Score Analyzer & Grade Predictor")
    st.markdown("**AI-201 Lab Project** | Team: Batool Binte Fazal (2024140), Rida Syed (2024540)")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Student Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} records")
            
            # Show raw data
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.head(10))
            
            # Data Cleaning
            st.header("1️⃣ Data Cleaning & Preprocessing")
            with st.spinner("Cleaning data..."):
                cleaner = DataCleaner()
                df_clean = cleaner.clean_data(df)
                df_encoded = cleaner.encode_categorical(df_clean)
                st.success(f"✅ Cleaned! Final shape: {df_encoded.shape}")
            
            # Feature Engineering
            st.header("2️⃣ Feature Engineering (NumPy)")
            with st.spinner("Creating features..."):
                engineer = FeatureEngineer()
                df_features, percentiles = engineer.create_all_features(df_encoded)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("25th Percentile", f"{percentiles['25th']:.1f}")
                col2.metric("50th Percentile", f"{percentiles['50th']:.1f}")
                col3.metric("75th Percentile", f"{percentiles['75th']:.1f}")
                col4.metric("90th Percentile", f"{percentiles['90th']:.1f}")
                
                st.success(f"✅ Created {len(engineer.features_created)} new features!")
            
            # Pandas Analysis
            st.header("3️⃣ Data Analysis (Pandas)")
            with st.spinner("Analyzing data..."):
                analyzer = PandasAnalyzer(df_features)
                qa_results = analyzer.answer_research_questions()
                
                if qa_results:
                    st.subheader("📊 Research Questions & Answers")
                    for question, answer in qa_results.items():
                        st.info(f"**{question}:** {answer}")
            
            # Visualizations
            st.header("4️⃣ Data Visualizations (Matplotlib)")
            with st.spinner("Generating plots..."):
                viz = Visualizer()
                plots = viz.generate_all_plots(df_features)
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogram", "📊 Bar Chart", "📊 Pie Chart", "🎨 3D Scatter"])
                
                with tab1:
                    st.subheader("Score Distribution")
                    if plots.get('histogram'):
                        st.pyplot(plots['histogram'])
                
                with tab2:
                    st.subheader("Subject Comparison")
                    if plots.get('bar_chart'):
                        st.pyplot(plots['bar_chart'])
                
                with tab3:
                    st.subheader("Demographics Distribution")
                    if plots.get('pie_chart'):
                        st.pyplot(plots['pie_chart'])
                
                with tab4:
                    st.subheader("⭐ 3D Performance Analysis (Advanced)")
                    if plots.get('3d_scatter'):
                        st.pyplot(plots['3d_scatter'])
            
            # Model Training
            st.header("5️⃣ Machine Learning Models")
            
            if st.button("🚀 Train All Models", type="primary"):
                with st.spinner("Training 3 models..."):
                    trainer = ModelTrainer()
                    models, results, feature_names = trainer.train_all_models(df_features)
                    
                    # Store in session state
                    st.session_state['models'] = models
                    st.session_state['scaler'] = trainer.scaler
                    st.session_state['feature_names'] = feature_names
                    st.session_state['results'] = results
                    
                    st.success("✅ All 3 models trained successfully!")
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Linear Regression RMSE", 
                                 f"{results['linear_regression']['RMSE']:.2f}")
                        st.metric("R² Score", 
                                 f"{results['linear_regression']['R²']:.3f}")
                    
                    with col2:
                        st.metric("Logistic Regression Accuracy", 
                                 f"{results['logistic_regression']['Accuracy']*100:.2f}%")
                    
                    with col3:
                        st.metric("KNN Accuracy", 
                                 f"{results['knn']['Accuracy']*100:.2f}%")
            
            # Model Persistence
            st.header("6️⃣ Model Persistence (Pickle)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Models"):
                    if 'models' in st.session_state:
                        persistence = ModelPersistence()
                        success = persistence.save_models(
                            st.session_state['models'],
                            st.session_state['scaler'],
                            st.session_state['feature_names']
                        )
                        if success:
                            st.success("✅ Models saved successfully!")
                    else:
                        st.warning("⚠️ Train models first!")
            
            with col2:
                if st.button("📥 Load Saved Models"):
                    persistence = ModelPersistence()
                    models, scaler, feature_names = persistence.load_models()
                    if models:
                        st.session_state['models'] = models
                        st.session_state['scaler'] = scaler
                        st.session_state['feature_names'] = feature_names
                        st.success("✅ Models loaded successfully!")
                    else:
                        st.error("❌ No saved models found!")
            
            # Predictions
            st.header("7️⃣ Make Predictions")
            
            if 'models' in st.session_state:
                st.subheader("Enter Student Information")
                
                # Create input form
                with st.form("prediction_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                        parental_edu = st.slider("Parental Education Level", 0, 5, 2)
                        study_time = st.slider("Study Time (hours/week)", 1, 10, 5)
                    
                    with col2:
                        test_prep = st.selectbox("Test Preparation Course", [0, 1], 
                                                format_func=lambda x: "Not Completed" if x == 0 else "Completed")
                        race = st.selectbox("Race / Ethnicity",[0, 1, 2, 3, 4],
                                            format_func=lambda x: ["Group A", "Group B", "Group C", "Group D", "Group E"][x])

                    
                    submitted = st.form_submit_button("🔮 Predict", type="primary")
                    
                    if submitted:
                        # Prepare input - match number of features used in training
                        input_data = [gender, parental_edu, study_time, test_prep, race]
                        
                        # Make predictions
                        predictor = Predictor(st.session_state['models'], st.session_state['scaler'])
                        
                        predicted_score = predictor.predict_score(input_data)
                        pass_fail = predictor.predict_pass_fail(input_data)
                        risk_level = predictor.predict_risk(input_data)
                        
                        if predicted_score:
                            grade = predictor.get_grade(predicted_score)
                            
                            # Show results
                            st.markdown("---")
                            st.subheader("🎯 Prediction Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Predicted Math Score", f"{predicted_score:.1f}")
                            col2.metric("Letter Grade", grade)
                            col3.metric("Pass/Fail", pass_fail, 
                                       delta="✅" if pass_fail == "Pass" else "❌")
                            col4.metric("Risk Level", risk_level if risk_level else "N/A")
            else:
                st.info("ℹ️ Train or load models first to make predictions!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)
    
    else:
        st.info("👈 Please upload a student dataset CSV file to begin analysis")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        st.markdown("""
        Your CSV should contain columns like:
        - `gender` (categorical)
        - `race/ethnicity` (categorical)
        - `parental level of education` (categorical)
        - `lunch` (categorical)
        - `test preparation course` (categorical)
        - `math score` (numeric 0-100)
        - `reading score` (numeric 0-100)
        - `writing score` (numeric 0-100)
        """)


# ====================================================================
# RUN APPLICATION
# ====================================================================
if __name__ == "__main__":
    main()