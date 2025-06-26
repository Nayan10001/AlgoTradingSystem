
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from src.logger_setup import setup_logger
from datetime import datetime, timedelta 
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger('MLPredictor')

class MLPredictor:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.model_params = self.config.get('ml_predictor', {})
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.trained_features = []
        self.selected_features = []
        self.model_type = self.model_params.get('model_type', 'LogisticRegression') 
        
    def _create_advanced_features(self, df):
        """
        Create advanced technical features for better prediction.
        Ensures that operations that might produce all NaNs early on 
        are handled gracefully if the resulting series is too short.
        """
        data = df.copy()
        
        # Price-based features
        if 'close' in data.columns and len(data['close']) > 10: 
            data['price_change_1d'] = data['close'].pct_change(1)
            data['price_change_3d'] = data['close'].pct_change(3)
            data['price_change_5d'] = data['close'].pct_change(5)
            data['price_change_10d'] = data['close'].pct_change(10)
            
            data['volatility_5d'] = data['close'].pct_change().rolling(5).std()
            data['volatility_10d'] = data['close'].pct_change().rolling(10).std()
            data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
            
            data['price_vs_sma_5'] = (data['close'] - data['close'].rolling(5).mean()) / (data['close'].rolling(5).mean() + 1e-9)
            data['price_vs_sma_10'] = (data['close'] - data['close'].rolling(10).mean()) / (data['close'].rolling(10).mean() + 1e-9)
            data['price_vs_sma_20'] = (data['close'] - data['close'].rolling(20).mean()) / (data['close'].rolling(20).mean() + 1e-9)
            
            if 'high' in data.columns and 'low' in data.columns:
                data['hl_spread'] = (data['high'] - data['low']) / (data['close'] + 1e-9)
                data['hl_spread_ma_5'] = data['hl_spread'].rolling(5).mean()
                data['hl_spread_ma_10'] = data['hl_spread'].rolling(10).mean()
                
                data['close_position_in_range'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-9)
                data['close_position_ma_5'] = data['close_position_in_range'].rolling(5).mean()
        
        if 'volume' in data.columns and len(data['volume']) > 20: # Need enough data for rolling means
            data['volume_ma_5'] = data['volume'].rolling(5).mean()
            data['volume_ma_10'] = data['volume'].rolling(10).mean()
            data['volume_ma_20'] = data['volume'].rolling(20).mean()
            
            data['volume_ratio_5'] = data['volume'] / (data['volume_ma_5'] + 1e-9)
            data['volume_ratio_10'] = data['volume'] / (data['volume_ma_10'] + 1e-9)
            data['volume_ratio_20'] = data['volume'] / (data['volume_ma_20'] + 1e-9)
            
            if 'price_change_1d' in data.columns: # Requires price_change_1d
                data['volume_price_trend'] = data['volume'] * data['price_change_1d']
            
        if 'RSI' in data.columns and len(data['RSI']) > 5:
            data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
            data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
            data['RSI_momentum'] = data['RSI'].diff()
            data['RSI_ma_5'] = data['RSI'].rolling(5).mean()
            
        if 'MACD_hist' in data.columns:
            data['MACD_hist_momentum'] = data['MACD_hist'].diff()
            data['MACD_hist_positive'] = (data['MACD_hist'] > 0).astype(int)
            
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns and 'close' in data.columns and len(data['close']) > 20:
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / (data['close'] + 1e-9)
            data['BB_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'] + 1e-9)
            data['BB_squeeze'] = (data['BB_width'] < data['BB_width'].rolling(20).quantile(0.1)).astype(int)
            
        if 'close' in data.columns and len(data['close']) > 20:
            data['sma_5'] = data['close'].rolling(5).mean()
            data['sma_10'] = data['close'].rolling(10).mean()
            data['sma_20'] = data['close'].rolling(20).mean()
            
            if len(data['sma_5']) > 1: data['sma_trend_5'] = (data['sma_5'].diff() > 0).astype(int)
            if len(data['sma_10']) > 1: data['sma_trend_10'] = (data['sma_10'].diff() > 0).astype(int)
            if len(data['sma_20']) > 1: data['sma_trend_20'] = (data['sma_20'].diff() > 0).astype(int)
            
            if len(data['sma_5']) > 1 and len(data['sma_20']) > 1: # Check for shift(1)
                 data['golden_cross'] = ((data['sma_5'] > data['sma_20']) & 
                                   (data['sma_5'].shift(1) <= data['sma_20'].shift(1))).astype(int)
                 data['death_cross'] = ((data['sma_5'] < data['sma_20']) & 
                                  (data['sma_5'].shift(1) >= data['sma_20'].shift(1))).astype(int)
            
        if hasattr(data.index, 'dayofweek'):
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['is_monday'] = (data['day_of_week'] == 0).astype(int)
            data['is_friday'] = (data['day_of_week'] == 4).astype(int)
            
        return data
    
    def _prepare_data(self, df_with_indicators):
        if df_with_indicators is None or df_with_indicators.empty:
            logger.warning("Input DataFrame for ML is empty.")
            return None, None

        data = self._create_advanced_features(df_with_indicators)
        
        basic_features = ['RSI', 'MACD_hist', 'VOLUME_RATIO', 'BB_position'] 
        advanced_features = [
            'price_change_1d', 'price_change_3d', 'price_change_5d', 'price_change_10d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20',
            'hl_spread', 'hl_spread_ma_5', 'hl_spread_ma_10',
            'close_position_in_range', 'close_position_ma_5',
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 
            'volume_price_trend',
            'RSI_oversold', 'RSI_overbought', 'RSI_momentum', 'RSI_ma_5',
            'MACD_hist_momentum', 'MACD_hist_positive',
            'BB_width', 'BB_squeeze',
            'sma_trend_5', 'sma_trend_10', 'sma_trend_20',
            'golden_cross', 'death_cross',
            'is_monday', 'is_friday'
        ]
        
        features_list_config = self.model_params.get('features', []) # Get features from config
        if not features_list_config: # If config is empty or 'features' not present, use defaults
            features_list = basic_features + advanced_features
        else:
            features_list = features_list_config


        available_features = [f for f in features_list if f in data.columns and not data[f].isnull().all()]
        if len(available_features) < len(features_list):
            missing_features = set(features_list) - set(available_features)
            logger.warning(f"Some features are not available or all NaNs: {missing_features}")
        
        if not available_features:
            logger.error("No features available for training after filtering.")
            return None, None
            
        target_horizon = self.model_params.get('target_horizon', 1)
        
        price_col = 'adj_close' if 'adj_close' in data.columns and not data['adj_close'].isnull().all() else 'close'
        if price_col not in data.columns or data[price_col].isnull().all():
            logger.error(f"Price column '{price_col}' not found or all NaNs in data.")
            return None, None

        data['future_price'] = data[price_col].shift(-target_horizon)
        data['target_simple'] = (data['future_price'] > data[price_col]).astype(int)
        
        threshold = self.model_params.get('prediction_threshold_pct', 0.01) # 1% threshold from config
        price_change_pct = (data['future_price'] - data[price_col]) / (data[price_col] + 1e-9)
        
        # Target classification: 0 for Down, 1 for Neutral, 2 for Up
        data['target_threshold'] = 1 # Default to Neutral
        data.loc[price_change_pct > threshold, 'target_threshold'] = 2  # Up
        data.loc[price_change_pct < -threshold, 'target_threshold'] = 0 # Down

        # Choose target based on config, default to simple binary
        target_strategy = self.model_params.get('target_strategy', 'simple_binary')
        if target_strategy == 'threshold_ternary':
            target_col = 'target_threshold'
        else: # Default 'simple_binary'
            target_col = 'target_simple'
        
        relevant_cols = available_features + [target_col]
        ml_data = data[relevant_cols].copy()
        
        ml_data.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinities
        ml_data = ml_data.fillna(method='ffill').fillna(method='bfill')
        ml_data.dropna(inplace=True)

        if ml_data.empty:
            logger.warning("No data remains after NaN handling for ML.")
            return None, None

        X = ml_data[available_features]
        y = ml_data[target_col]
        
        self.trained_features = available_features # Store all features before selection
        return X, y

    def train_model(self, df_with_indicators, symbol_for_log=""):
        if not self.model_params.get('enabled', False):
            logger.info(f"ML Predictor is disabled in config for {symbol_for_log}.")
            return None

        X, y = self._prepare_data(df_with_indicators)
        if X is None or y is None or X.empty or y.empty:
            logger.error(f"ML data preparation failed for {symbol_for_log}.")
            return None

        class_distribution = y.value_counts(normalize=True)
        logger.info(f"Class distribution for {symbol_for_log}: {class_distribution.to_dict()}")
        
        if y.nunique() < 2:
            logger.error(f"Target variable for {symbol_for_log} has only one class. Model cannot be trained.")
            return None
            
        if class_distribution.min() < 0.05 and y.nunique() > 1: # Less than 5% minority class
            logger.warning(f"Severe class imbalance detected for {symbol_for_log}. Minimum class proportion: {class_distribution.min():.3f}.")

        test_size = self.model_params.get('test_size', 0.2)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if y.value_counts().min() > 1 else None # Stratify if at least 2 samples per class
            )
        except ValueError as e:
            logger.warning(f"Stratification failed for {symbol_for_log}: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if X_train.empty or X_test.empty:
            logger.error(f"Training or testing set is empty for {symbol_for_log} after split.")
            return None

        scaler_type = self.model_params.get('scaler', 'RobustScaler')
        if scaler_type == 'StandardScaler':
            self.scaler = StandardScaler()
        else: # Default to RobustScaler
            self.scaler = RobustScaler()
            
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        n_features_to_select = min(self.model_params.get('max_features', 10), X_train_scaled.shape[1]) # Default 10
        if n_features_to_select > 0 and X_train_scaled.shape[1] > n_features_to_select :
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [self.trained_features[i] for i in selected_indices]
        else: # Use all features
            self.feature_selector = None
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            self.selected_features = self.trained_features
        
        logger.info(f"Selected features for {symbol_for_log} ({len(self.selected_features)}): {self.selected_features}")
        if not self.selected_features:
            logger.error(f"No features were selected for {symbol_for_log}. Cannot train model.")
            return None

        # Model selection
        if self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get('rf_n_estimators', 100),
                max_depth=self.model_params.get('rf_max_depth', None),
                random_state=42,
                class_weight='balanced' if y.nunique() == 2 else None # Only for binary
            )
        else: # Default LogisticRegression
            self.model = LogisticRegression(
                random_state=42, solver='liblinear',
                C=self.model_params.get('lr_regularization_strength', 1.0), # Renamed config param
                class_weight='balanced' if y.nunique() == 2 else None, # Only for binary
                max_iter=1000
            )

        self.model.fit(X_train_selected, y_train)
        
        y_pred = self.model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        auc_score = np.nan
        if hasattr(self.model, "predict_proba") and y.nunique() == 2: # AUC for binary classification
            try:
                y_pred_proba = self.model.predict_proba(X_test_selected)
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate AUC for {symbol_for_log}: {e}")
        
        cv_scores = cross_val_score(self.model, X_train_selected, y_train, cv=min(5, y_train.value_counts().min()), scoring='accuracy') # Ensure cv folds <= smallest class count
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        logger.info(f"Model trained for {symbol_for_log} using {self.model_type}:")
        logger.info(f"  Test Accuracy: {accuracy:.4f}")
        if not np.isnan(auc_score): logger.info(f"  AUC Score: {auc_score:.4f}")
        logger.info(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")

        top_features_str = "N/A"
        if hasattr(self.model, 'coef_') and self.model_type == 'LogisticRegression': # Logistic Regression coefficients
            feature_importance = dict(zip(self.selected_features, abs(self.model.coef_[0])))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features_str = str(top_features)
            logger.info(f"Top 5 important LR features for {symbol_for_log}: {top_features}")
        elif hasattr(self.model, 'feature_importances_') and self.model_type == 'RandomForestClassifier': # RF feature importances
            feature_importance = dict(zip(self.selected_features, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features_str = str(top_features)
            logger.info(f"Top 5 important RF features for {symbol_for_log}: {top_features}")


        training_results = {
            'Log_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': symbol_for_log,
            'Prediction_For_Date': 'Training_Evaluation', # This is meta-info about the training run
            'Model_Type': self.model_type,
            'Test_Accuracy': f"{accuracy:.4f}",
            'AUC_Score': f"{auc_score:.4f}" if not np.isnan(auc_score) else "N/A",
            'CV_Accuracy_Mean': f"{cv_mean:.4f}",
            'CV_Accuracy_Std': f"{cv_std:.4f}",
            'Features_Selected': ", ".join(self.selected_features),
            'Total_Training_Features': len(self.trained_features), # All features before selection
            'Selected_Features_Count': len(self.selected_features),
            'Train_Size': len(X_train),
            'Test_Size': len(X_test),
            'Class_Distribution': str(class_distribution.to_dict()),
            'Top_Features': top_features_str[:250] # Truncate if too long for GSheet cell
        }
        
        return training_results

    def predict(self, df_with_indicators_latest_row, symbol_for_log=""):
        if not self.model_params.get('enabled', False) or self.model is None or self.scaler is None:
            logger.info(f"ML Predictor disabled, not trained, or scaler not found for {symbol_for_log}.")
            return None
        
        if df_with_indicators_latest_row is None or df_with_indicators_latest_row.empty:
            logger.warning(f"No latest data row provided for prediction for {symbol_for_log}.")
            return None
        
        if isinstance(df_with_indicators_latest_row, pd.Series):
            latest_data = df_with_indicators_latest_row.to_frame().T
        else:
            latest_data = df_with_indicators_latest_row.iloc[[-1]].copy() # Ensure it's the last row and a copy

        latest_data_enhanced = self._create_advanced_features(latest_data)
        
        # Ensure all self.selected_features are present in latest_data_enhanced
        # These were the features used for training after scaling and KBest selection
        missing_for_pred = [f for f in self.selected_features if f not in latest_data_enhanced.columns]
        if missing_for_pred:
            logger.error(f"Missing selected features for prediction for {symbol_for_log}: {missing_for_pred}")
            return None
            
        # Use self.trained_features for scaling, then self.selected_features for feature_selector
        # X_live must have columns in the order of self.trained_features
        X_live = pd.DataFrame(columns=self.trained_features, index=latest_data_enhanced.index)
        for col in self.trained_features:
            if col in latest_data_enhanced.columns:
                X_live[col] = latest_data_enhanced[col]
            else: # Should not happen if self.trained_features was derived from data properly
                logger.error(f"Critical: Feature '{col}' from trained_features not in latest_data_enhanced for {symbol_for_log}.")
                X_live[col] = 0 # Or some imputation

        if X_live.isnull().values.any():
            logger.warning(f"NaN values found in features for prediction for {symbol_for_log}. Filling with 0.")
            X_live = X_live.fillna(0) # Simple imputation for prediction

        X_live_scaled = self.scaler.transform(X_live) # Scale using all trained features
        
        if self.feature_selector: # Apply feature selection if it was used in training
            X_live_selected = self.feature_selector.transform(X_live_scaled)
        else:
            X_live_selected = X_live_scaled
        
        prediction_val = self.model.predict(X_live_selected)[0]
        
        confidence_up = np.nan
        confidence_down = np.nan
        max_confidence = np.nan
        
        if hasattr(self.model, "predict_proba"):
            confidence_probs = self.model.predict_proba(X_live_selected)[0]
            target_strategy = self.model_params.get('target_strategy', 'simple_binary')

            if target_strategy == 'threshold_ternary': # 0: Down, 1: Neutral, 2: Up
                confidence_down = confidence_probs[0]
                confidence_neutral = confidence_probs[1]
                confidence_up = confidence_probs[2]
                max_confidence = max(confidence_probs)
                if prediction_val == 0: predicted_movement = "DOWN"
                elif prediction_val == 1: predicted_movement = "NEUTRAL"
                else: predicted_movement = "UP"
            else: # Simple binary: 0: Down, 1: Up
                confidence_down = confidence_probs[0]
                confidence_up = confidence_probs[1]
                max_confidence = max(confidence_probs)
                predicted_movement = "UP" if prediction_val == 1 else "DOWN"
        else: # Models without predict_proba (e.g. SVM default)
            predicted_movement = "UP" if prediction_val == 1 else "DOWN" # Or based on ternary if applicable

        risk_level = "HIGH" if (not np.isnan(max_confidence) and max_confidence < 0.6) else \
                     "MEDIUM" if (not np.isnan(max_confidence) and max_confidence < 0.75) else \
                     "LOW" if not np.isnan(max_confidence) else "UNKNOWN"
        
        logger.info(f"Enhanced ML Prediction for {symbol_for_log} ({self.model_type}):")
        logger.info(f"  Movement: {predicted_movement}")
        if not np.isnan(confidence_up): logger.info(f"  Confidence UP: {confidence_up:.3f}")
        if not np.isnan(confidence_down): logger.info(f"  Confidence DOWN: {confidence_down:.3f}")
        if target_strategy == 'threshold_ternary':
            if not np.isnan(confidence_neutral): logger.info(f"  Confidence NEUTRAL: {confidence_neutral:.3f}")
        logger.info(f"  Risk Level: {risk_level}")

        prediction_date_str = "N/A"
        if latest_data_enhanced.index[0] and isinstance(latest_data_enhanced.index[0], (pd.Timestamp, datetime)):
            try:
                prediction_date_str = (latest_data_enhanced.index[0] + timedelta(days=self.model_params.get('target_horizon', 1))).strftime('%Y-%m-%d')
            except Exception:
                pass # Keep N/A if date operation fails

        prediction_results = {
            'Log_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': symbol_for_log,
            'Prediction_For_Date': prediction_date_str,
            'Predicted_Movement': predicted_movement,
            'Confidence_UP': f"{confidence_up:.3f}" if not np.isnan(confidence_up) else "N/A",
            'Confidence_DOWN': f"{confidence_down:.3f}" if not np.isnan(confidence_down) else "N/A",
            'Confidence_NEUTRAL': f"{confidence_neutral:.3f}" if target_strategy == 'threshold_ternary' and not np.isnan(confidence_neutral) else "N/A",
            'Max_Confidence': f"{max_confidence:.3f}" if not np.isnan(max_confidence) else "N/A",
            'Risk_Level': risk_level,
            'Features_Used_Count': len(self.selected_features),
            'Model_Type': self.model_type,
            'Target_Strategy': target_strategy
        }
        
        return prediction_results


def predict_multiple_stocks(symbols, config, data_fetcher, indicator_calculator_func, gsheet_manager=None):
    """
    Train and predict for multiple stocks, returning consolidated results.
    Optionally logs to Google Sheets.
    """
    all_results = {
        'training_evaluations': [],
        'predictions': [],
        'summary': {}
    }
    
    successful_predictions = 0
    ml_config = config.get('ml_predictor', {})
    indicator_params = config.get('indicators', {})
    
    for symbol in symbols:
        logger.info(f"--- Processing {symbol} for ML Prediction ---")
        
        try:
            fetch_period = ml_config.get('data_fetch_period_ml', "2y") # e.g., "2y", "500d"
            fetch_interval = ml_config.get('data_fetch_interval_ml', "1d")
            hist_data = data_fetcher.fetch_stock_data(symbol, period=fetch_period, interval=fetch_interval)
            
            if hist_data.empty or len(hist_data) < ml_config.get('min_data_points_for_ml', 60): # Min 60 data points
                logger.warning(f"Not enough data fetched for {symbol} (got {len(hist_data)}). Skipping ML.")
                continue
                
            df_with_indicators = indicator_calculator_func(hist_data, **indicator_params)
            
            if df_with_indicators.empty:
                logger.warning(f"Indicator calculation failed for {symbol}. Skipping ML.")
                continue
            
            predictor = MLPredictor(config=config) # Pass full config
            
            training_result = predictor.train_model(df_with_indicators, symbol_for_log=symbol)
            if training_result:
                all_results['training_evaluations'].append(training_result)
                
                last_row_for_pred = df_with_indicators.iloc[[-1]] # Ensure it's the latest row
                prediction_result = predictor.predict(last_row_for_pred, symbol_for_log=symbol)
                
                if prediction_result:
                    all_results['predictions'].append(prediction_result)
                    successful_predictions += 1
            else:
                logger.warning(f"ML Model training failed for {symbol}.")
                    
        except Exception as e:
            logger.error(f"Error processing {symbol} for ML: {str(e)}", exc_info=True)
            continue
    
    if all_results['predictions']:
        up_predictions = sum(1 for p in all_results['predictions'] if p['Predicted_Movement'] == 'UP')
        down_predictions = sum(1 for p in all_results['predictions'] if p['Predicted_Movement'] == 'DOWN')
        neutral_predictions = len(all_results['predictions']) - up_predictions - down_predictions

        confidences = [float(p['Max_Confidence']) for p in all_results['predictions'] if p['Max_Confidence'] != "N/A"]
        avg_confidence = np.mean(confidences) if confidences else np.nan
        
        all_results['summary'] = {
            'Run_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total_Stocks_Processed': len(symbols),
            'Successful_Predictions': successful_predictions,
            'Up_Predictions': up_predictions,
            'Down_Predictions': down_predictions,
            'Neutral_Predictions': neutral_predictions,
            'Average_Max_Confidence': f"{avg_confidence:.3f}" if not np.isnan(avg_confidence) else "N/A",
            'Model_Type_Used': ml_config.get('model_type', 'LogisticRegression'),
            'Target_Strategy_Used': ml_config.get('target_strategy', 'simple_binary')
        }

    # Log to Google Sheets if manager is provided
    if gsheet_manager and ml_config.get('enabled', False):
        logger.info("Logging ML results to Google Sheets...")
        if all_results['training_evaluations']:
            df_training_eval = pd.DataFrame(all_results['training_evaluations'])
            # Convert datetime objects to string for GSheet compatibility
            if 'Log_Timestamp' in df_training_eval.columns:
                 df_training_eval['Log_Timestamp'] = df_training_eval['Log_Timestamp'].astype(str)

            gsheet_manager.log_trades_from_dataframe(
                df_training_eval, 
                sheet_name=ml_config.get('gsheet_ml_training_log', 'ML_Training_Log'),
                clear_sheet=ml_config.get('gsheet_clear_ml_training_log', False)
            )
        if all_results['predictions']:
            df_predictions = pd.DataFrame(all_results['predictions'])
            # Convert datetime objects to string
            if 'Log_Timestamp' in df_predictions.columns:
                df_predictions['Log_Timestamp'] = df_predictions['Log_Timestamp'].astype(str)

            gsheet_manager.log_trades_from_dataframe(
                df_predictions, 
                sheet_name=ml_config.get('gsheet_ml_predictions_log', 'ML_Predictions_Log'), # Use a different sheet name
                clear_sheet=ml_config.get('gsheet_clear_ml_predictions_log', False)
            )
        if all_results['summary']:
            gsheet_manager.update_key_value_sheet(
                all_results['summary'],
                sheet_name=ml_config.get('gsheet_ml_summary', 'ML_Summary'),
                clear_sheet=False # Append to summary
            )
    
    return all_results


# if __name__ == '__main__':
#     from data_handler import DataFetcher
#     from indicators_calculator import insertion_to_df as calculate_indicators
#     import yaml
#     import os
#     from dotenv import load_dotenv
#     from gsheet_manager import GoogleSheetsManager


#     load_dotenv()
#     logger.info("Running")

#     # Load config
#     try:
#         with open('config.yaml', 'r') as f:
#             config = yaml.safe_load(f)
#     except FileNotFoundError:
#         logger.error("config.yaml not found. ML Predictor test cannot run with full configuration.")
#         config = {} # Fallback to empty config

#     if not config or 'ml_predictor' not in config:
#         logger.warning("ml_predictor section not in config.yaml or config is empty. Using minimal default for testing.")
#         config.setdefault('ml_predictor', {
#             'enabled': True, 'model_type': 'LogisticRegression', 'target_horizon': 1,
#             'test_size': 0.2, 'max_features': 10, 'lr_regularization_strength': 1.0,
#             'prediction_threshold_pct': 0.01, 'target_strategy': 'simple_binary',
#             'features': ['RSI', 'MACD_hist', 'VOLUME_RATIO', 'BB_position', 'price_change_1d', 'volatility_5d'] # Example minimal features
#         })
#         config.setdefault('indicators', { # Default indicator params needed by features
#             'rsi_period': 14, 'sma_short_period': 10, 'sma_long_period': 30, # Adjusted for smaller feature set
#             'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
#             'bb_window': 20, 'bb_std_dev': 2, 'volume_ma_window': 20
#         })
    
#     logger.setLevel(config.get('logging_level', 'INFO'))
#     stocks_to_predict = config.get('stocks_to_track') 

#     logger.info(f"Testing MLPredictor for: {stocks_to_predict}")
    
#     fetcher = DataFetcher()
    
#     # Initialize GoogleSheetsManager if configured
#     gsheet_manager = None
#     if config.get('google_sheets', {}).get('enabled_for_ml', False) and \
#        os.getenv('GOOGLE_SHEET_ID') and os.getenv('GOOGLE_SHEET_ID') != "YOUR_SPREADSHEET_ID_HERE" and \
#        os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE') and os.path.exists(os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')):
#         try:
#             gsheet_manager = GoogleSheetsManager()
#             logger.info("GoogleSheetsManager initialized for ML test.")
#             # Create ML specific worksheets if they don't exist
#             ml_worksheet_configs = {
#                 config.get('ml_predictor',{}).get('gsheet_ml_training_log', 'ML_Training_Log'): None, # No specific headers needed here, will take from df
#                 config.get('ml_predictor',{}).get('gsheet_ml_predictions_log', 'ML_Predictions_Log'): None,
#                 config.get('ml_predictor',{}).get('gsheet_ml_summary', 'ML_Summary'): ["Metric", "Value", "Last_Updated"]
#             }
#             gsheet_manager.create_project_worksheets(worksheet_configs=ml_worksheet_configs)

#         except Exception as e:
#             logger.error(f"Failed to initialize GoogleSheetsManager for ML test: {e}")
#             gsheet_manager = None
#     else:
#         logger.info("Google Sheets for ML is not enabled or configured. Skipping GSheet logging for ML test.")


#     results = predict_multiple_stocks(
#         symbols=stocks_to_predict,
#         config=config,
#         data_fetcher=fetcher,
#         indicator_calculator_func=calculate_indicators,
#         gsheet_manager=gsheet_manager
#     )
    
#     logger.info("\n=== ML PREDICTION SUMMARY ===")
#     if results.get('summary'):
#         for key, value in results['summary'].items():
#             logger.info(f"  {key}: {value}")
#     else:
#         logger.info("  No summary generated.")
    
#     logger.info("\n=== INDIVIDUAL ML PREDICTIONS ===")
#     if results.get('predictions'):
#         for prediction in results['predictions']:
#             logger.info(f"  Stock: {prediction['Symbol']}")
#             logger.info(f"    Prediction Date: {prediction['Prediction_For_Date']}")
#             logger.info(f"    Predicted Movement: {prediction['Predicted_Movement']}")
#             logger.info(f"    Max Confidence: {prediction['Max_Confidence']}")
#             logger.info(f"    Risk Level: {prediction['Risk_Level']}")
#             logger.info("    ---")
#     else:
#         logger.info("  No predictions generated.")

#     logger.info("\n=== TRAINING EVALUATIONS ===")
#     if results.get('training_evaluations'):
#         for eval_result in results['training_evaluations']:
#             logger.info(f"  Stock: {eval_result['Symbol']}")
#             logger.info(f"    Model Type: {eval_result['Model_Type']}")
#             logger.info(f"    Test Accuracy: {eval_result['Test_Accuracy']}")
#             logger.info(f"    AUC Score: {eval_result['AUC_Score']}")
#             logger.info(f"    Selected Features: {eval_result['Selected_Features_Count']} of {eval_result['Total_Training_Features']}")
#             logger.info(f"    Top Features: {eval_result.get('Top_Features', 'N/A')}")
#             logger.info("    ---")
#     else:
#         logger.info("  No training evaluations generated.")

#     logger.info("closed")