# Payment Default Prediction - DATATOUR 2025

**Team**: InsightX Divas  
**Members**: Christy Alotse, Aicha Ousseni, Danielle Fotsi  
**Cameroon Result**: 🥇 First Place (0.66 AUC) - 2nd: 0.65, 3rd: 0.64  
**Global Result**: 25th out of 96 teams (Top 3: 0.999 AUC)  
**Key Learning**: Feature engineering differentiates good models from exceptional ones

A binary classification project predicting payment defaults using LightGBM DART on 17.7 million samples. This repository documents Team InsightX Divas' systematic approach, national victory, international ranking, and critical lessons about feature engineering in competitive machine learning.

---

## Table of Contents

- [Competition Overview](#competition-overview)
- [Team](#team)
- [The Challenges](#the-challenges)
- [Our Approach](#our-approach)
- [Model Benchmark](#model-benchmark)
- [Results](#results)
- [What We Did Right](#what-we-did-right)
- [What We'll Improve](#what-well-improve)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [Future Competition Strategy](#future-competition-strategy)
- [Acknowledgments](#acknowledgments)

---

## Competition Overview

**Event**: DATATOUR 2025 - Pan-African Data Science Competition  
**Challenge**: Binary classification for payment default prediction  
**Dataset**: 17.7 million training samples, 60 features  
**Metric**: ROC AUC Score  

### Results

**National Phase (Cameroon)**:
- 🥇 **InsightX Divas: 0.66 AUC**
- 🥈 Second Place: 0.65 AUC
- 🥉 Third Place: 0.64 AUC

**International Phase (Global)**:
- **Ranking**: 25th out of 96 teams
- **Our Score**: 0.66 AUC
- **Top 3 Teams**: 0.999 AUC

**Analysis**: We won nationally with solid methodology but ranked 25th globally due to minimal feature engineering. Top teams achieved 0.999 AUC through extensive feature creation, not different algorithms.

---

## Team

**InsightX Divas**:

- **Christy Alotse**
- **Aicha Ousseni**
- **Danielle Fotsi**

Our collaborative approach enabled systematic decision-making and cross-validation of ideas.

---

## The Challenges

### Challenge 1: Severe Class Imbalance

**Problem**: Dataset extremely imbalanced (Class 0 >> Class 1). Training on imbalanced data produces useless models that predict "no default" for everyone.

**Solutions Considered**:
1. **Oversampling**: Generate synthetic minority samples
2. **Undersampling**: Reduce majority class to match minority

**Our Choice**: Undersampling

**Reasoning**: With 17.7M samples, oversampling would have doubled or tripled the dataset. Even after 70% memory optimization, our hardware couldn't handle additional synthetic samples.

**Trade-off**: Lost majority class information for computational feasibility.

### Challenge 2: Computational Constraints at Scale

**Problem**: Even after undersampling, 14+ million balanced samples meant hours per model training.

**Solution**: Created representative 120,000-sample subset for rapid model benchmarking. This allowed testing 10+ algorithms in reasonable time before committing to full-scale training.

---

## Our Approach

### 1. Memory Optimization (70% Reduction)

Working with 17.7 million samples required aggressive memory management to enable local training.

A custom memory optimization function (“squeezer”) was implemented to reduce the DataFrame footprint without altering the data itself.

What the optimization does

The process applies safe type downcasting:

Integer columns:
int64 → int32 / int16 / int8 when value ranges allow it

Float columns:
float64 → float32 when precision is sufficient

Categorical-like columns:
Converted to category dtype when appropriate

Each conversion is performed only after verifying the column’s minimum and maximum values to prevent overflow or precision issues.

**Impact**

~70% reduction in memory usage

Enabled local training on 14M+ balanced samples

Prevented RAM overflow during model benchmarking and final training

Reduced data loading time

**Important Note**

This optimization:

Does not modify values

Does not compress the dataset

Does not affect model logic

It strictly optimizes the in-memory representation of the data.


### 2. Validation Strategy (Three-Tier)

```
Original Dataset (17.7M imbalanced)
        ↓
   Global Split (80/20)
        ↓
    ┌────────────────┬──────────────┐
train_full (80%)    holdout (20%)
 (imbalanced)       (imbalanced)
    ↓
Undersampling
    ↓
Balanced (14M)
    ↓
Internal Split
    ↓
┌──────────┬──────────┐
train      validation
(balanced) (balanced)
```

**Key**: Holdout maintained original imbalanced distribution as proxy for real-world performance.

---

## Model Benchmark

Systematically tested 10+ algorithms on 120K sample using 5-fold stratified cross-validation:

### Algorithms Tested

**Tree-Based**:
- Balanced Random Forest
- Extra Trees
- HistGradient Boosting (calibrated)

**Boosting**:
- **LightGBM DART** ⭐ (Selected)
- LightGBM GOSS
- XGBoost (histogram)
- XGBRF
- CatBoost
- Easy Ensemble (AdaBoost)

**Ensemble**:
- **Stacking Classifier** 🏆 (Best but infeasible)
  - Base: Extra Trees, LightGBM DART, XGBRF, CatBoost
  - Meta: Logistic Regression

### Benchmark Results

| Rank | Model | Sample Performance | Memory | Decision |
|------|-------|-------------------|--------|----------|
| 1 | Stacking | Highest AUC | Very High | ❌ Exceeds RAM for 14M samples |
| 2 | LightGBM DART | Near Stacking | Efficient | ✅ **Selected** |
| 3 | CatBoost | Strong | Moderate | - |
| 4 | XGBoost | Good | Moderate | - |

**Final Decision**: LightGBM DART
- Performance nearly matched Stacking
- Memory efficient for 14M samples
- DART regularization prevents overfitting
- Ideal for financial risk with limited features

---

## Results

### Model Performance

**Algorithm**: LightGBM DART  
**Training**: 3000 iterations (converged ~2400)

| Split | Samples | ROC AUC | Notes |
|-------|---------|---------|-------|
| Validation | Balanced | 0.6593 | During training |
| Holdout | Imbalanced | 0.6605 | Real-world proxy |
| **Gap** | - | **0.0012** | Excellent generalization ✓ |

**Training Progression**:
```
Iteration  300: 0.6480
Iteration  600: 0.6527
Iteration  900: 0.6567
Iteration 1200: 0.6578
Iteration 1500: 0.6586
Iteration 1800: 0.6589
Iteration 2100: 0.6592
Iteration 2400: 0.6593 ← Convergence
Iteration 2700: 0.6593
Iteration 3000: 0.6593 ← Stable
```

### Competition Results

**Cameroon Phase**:
- 🥇 **InsightX Divas**: 0.66 AUC
- 🥈 Second Place: 0.65 AUC  
- 🥉 Third Place: 0.64 AUC
- **Achievement**: National champions, qualified for international phase

**International Phase**:
- **Global Rank**: 25th of 96 teams
- **Our Score**: 0.66 AUC
- **Top 3 Scores**: 0.999 AUC
- **Gap**: 0.33 AUC points

**Interpretation**:
- Top third globally (beat 71 teams)
- Solid methodology validated by national win
- Feature engineering gap revealed by global rankings
- Clear improvement path identified

---

## What We Did Right

### ✅ 1. Systematic Model Selection

- Benchmarked 10+ algorithms on representative sample
- Used 5-fold stratified cross-validation
- Prevented wasting compute on suboptimal models
- Made data-driven selection (LightGBM DART)

### ✅ 2. Strategic Compromises

- Undersampling over oversampling (hardware constraints)
- LightGBM DART over Stacking (RAM limitations)
- Feasible solutions that delivered results

### ✅ 3. Rigorous Validation

- Three-tier approach: train/validation/holdout
- Holdout maintained original class distribution
- Honest performance estimates (0.66 not inflated)

### ✅ 4. Memory Efficiency

- 70% reduction through type optimization
- Enabled local training without cloud costs
- Faster iteration cycles

### ✅ 5. DART Regularization

- 0.0012 gap between validation/holdout
- Dropout mechanism prevented overfitting
- No manual early stopping needed

### ✅ 6. Team Collaboration

- Christy's data processing expertise
- Aicha's analytical rigor
- Divided responsibilities effectively
- Cross-validated all major decisions

**Result**: National championship through solid methodology.

---

## What We'll Improve

### ❌ Critical Gap: Feature Engineering

**What we did**: Used 60 raw features with minimal transformation
- Memory optimization only
- Basic preprocessing
- No domain knowledge applied

**What top teams likely did**: Created hundreds of engineered features

### Missing Feature Categories

#### 1. Temporal Features
```python
# Time-based patterns
payment_7day_rolling_avg
payment_30day_rolling_avg
payment_90day_rolling_std
days_since_last_payment
consecutive_ontime_payments
payment_trend_direction_7day
payment_velocity
```

#### 2. Ratio Features
```python
# Financial ratios
payment_to_balance_ratio
debt_to_credit_limit_ratio
actual_vs_minimum_payment_ratio
payment_frequency_to_age_ratio
utilization_rate
debt_service_ratio
```

#### 3. Interaction Features
```python
# Feature combinations
account_age_x_payment_history_score
transaction_freq_x_avg_amount
credit_utilization_x_payment_consistency
payment_amount_x_account_tenure
balance_trend_x_payment_pattern
```

#### 4. Aggregation Features
```python
# Historical aggregations
total_payments_last_30d
total_payments_last_60d
total_payments_last_90d
max_consecutive_missed_payments
late_payment_percentage_30d
late_payment_percentage_90d
payment_amount_variance_30d
debt_accumulation_rate
```

#### 5. Domain-Specific Features
```python
# Financial expertise required
composite_risk_score
customer_segment_classification
payment_behavior_pattern_flag
default_probability_indicator
payment_stability_index
creditworthiness_score
```

**Expected Impact**: +0.15-0.25 AUC improvement (→ 0.85-0.90 range)

---

## Installation

### Prerequisites

- Python 3.8+
- 16GB+ RAM recommended
- pip package manager

### Setup

```bash
# Clone repository
git clone https://[https://github.com/DanielleFotsi/payment-default-prediction.git]
cd payment-default-prediction

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
lightgbm>=4.0.0
scikit-learn>=1.2.0
imbalanced-learn>=0.11.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
joblib>=1.3.0
```

---

## Usage

### Running the Pipeline

```bash
jupyter notebook Payment_Default_prediction.ipynb
```

**Notebook Sections**:
1. Dependency installation
2. Data loading
3. Memory optimization
4. EDA and class distribution
5. Class imbalance handling (undersampling)
6. Model benchmarking (10+ algorithms)
7. Final model training (LightGBM DART)
8. Test prediction
9. Submission file creation
10. Environment documentation

### Loading Trained Model

```python
import joblib

# Load model
model = joblib.load('model_lgbm_dart_final.pkl')

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

---

## Project Structure

```
.
├── Payment_Default_ Prediction.ipynb                  # Complete pipeline
├── model_lgbm_dart_final.pkl      # Trained model (45MB) in a folder called "Soumission" link[https://drive.google.com/drive/folders/1AxImyNtHX5X0p_zusvsUAbWsb-Yxb_ul?usp=drive_link]
├── README.md                       # This file
├── data/                           # link[https://drive.google.com/drive/folders/1gTa4g4PoGoCW1We_2BIFmuzJOF9LpUNB?usp=drive_link]
│   ├── train.parquet
│   ├── test.parquet
│   └── submission.parquet

```

---

## Key Learnings

### 1. National Success ≠ Global Competitiveness

**Our Experience**:
- Won Cameroon (0.66 vs 0.65 vs 0.64)
- Ranked 25th globally (top 3 at 0.999)

**Lesson**: Winning nationally qualifies you for international competition. Competing internationally reveals what's truly possible.

**Mindset**: Motivating, not discouraging. We know the path forward.

### 2. Feature Engineering Is Primary

**Our Split**: 5% features, 95% modeling/optimization  
**Top Teams**: ~60% features, 30% modeling, 10% optimization

**Formula**:
- Good methodology + Raw features = 0.66 AUC (National winner)
- Good methodology + Engineered features = 0.999 AUC (Global winner)

**Action**: Reverse our priorities. Features first, models second.

### 3. Methodology Gets You In, Features Win

Our methodology won nationally:
- Systematic benchmarking ✓
- Proper validation ✓
- Smart compromises ✓
- Team collaboration ✓

Globally, everyone has solid methodology. **Features differentiate winners**.

### 4. The Gap Is a Roadmap

25th of 96 teams means:
- Top third globally ✓
- Beat 71 teams ✓
- Clear examples (top 24) of better approaches ✓
- Specific improvement path identified ✓

**Perspective**: Not a failure—a foundation with clear next steps.

### 5. Team Composition Matters

**Current Strengths**:
- Christy: Data processing, memory optimization
- Aicha: Analytical rigor, validation
- Danielle: Model selection, benchmarking

**Next Addition**: Domain expert (financial professional who understands payment default predictors)

### 6. Constraints Drive Innovation

Memory limitations forced:
- 70% memory optimization (now a core skill)
- Undersampling strategy (taught us sampling trade-offs)
- LightGBM DART selection (learned DART's value)

**Result**: Better engineers through necessity.

---

## Reproducibility

To reproduce our 0.66 AUC result:

```bash
# Set seed
RANDOM_SEED = 42

# Run notebook
jupyter notebook Payment_Default_ Prediction.ipynb
```

**Note**: Exact reproduction requires same data split (seed 42) and library versions. LightGBM DART has inherent randomness; minor variations expected.

For exact results: Load `model_lgbm_dart_final.pkl`

---

## Technical Specifications

**Hardware**:
- RAM: 16GB
- CPU: 8 threads
- Training Time: ~4 hours (3000 iterations)

**Software**:
- Python: 3.9
- LightGBM: 4.0+
- Scikit-learn: 1.2+

**Model Size**: 45MB (joblib serialized)

---

## Acknowledgments

**Team InsightX Divas**:
- **Christy Alotse**: Data processing excellence and memory optimization
- **Aicha Ousseni**: Analytical rigor and validation strategy
- **Danielle Fotsi**: Model selection and systematic benchmarking

**Competition**:
- DATATOUR 2025 organizers for this learning opportunity
- LightGBM team for excellent documentation
- Data science community for shared knowledge

---

## Citation

```bibtex
@misc{insightxdivas2025,
  author = {Fotsi, Danielle and Alotse, Christy and Ousseni, Aicha},
  title = {Payment Default Prediction - DATATOUR 2025},
  year = {2025},
  publisher = {GitHub},
  team = {InsightX Divas},
  note = {1st place Cameroon (0.66 AUC), 25th globally of 96 teams}
}
```

---

## License

Available for educational purposes. Please cite if used.

---

## Contact

**Team InsightX Divas**:
- GitHub: [@DanielleFotsi](https://github.com/DanielleFotsi)
- LinkedIn: [linkedin.com/in/danielle-laura-nkonhawe-fotsi]
- Email: daniellefotsi@gmail.com

---

## Final Thoughts

Team InsightX Divas won Cameroon and ranked 25th globally. We proved we can:
- Handle massive datasets efficiently
- Make smart methodological choices
- Build generalizing models
- Win nationally
- Compete internationally

The gap to 0.999 AUC is clear: **feature engineering**.

Top teams used same algorithms. Better inputs.

**Next competition**: We'll engineer features first, model second.

We're not discouraged. We're motivated. We have a roadmap.

---

**Note** : The implementation notebook is written in French.

**⭐ Star this repository if you found our journey helpful!**

*Honest assessment of competitive ML: what worked, what didn't, why—and what's next.*
