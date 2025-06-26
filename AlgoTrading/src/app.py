# streamlit_app.py
import streamlit as st
import pandas as pd
import yaml
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import asyncio
import traceback 
import logging 
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app_logic import (
    load_configuration, save_configuration,
    initialize_gsheet_manager,
    run_backtesting_logic, run_ml_prediction_logic,
    fetch_data_from_gsheet
)

# Import Telegram functionality
from src.alert_messenger import TelegramManager 

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AlgoAnalytics Dashboard")
st.markdown("""
    <h1 style='text-align: center; font-size: 3em;'>AlgoAnalytics Dashboard</h1>
""", unsafe_allow_html=True)
load_dotenv()

# --- Telegram Message Formatting Helpers ---
def format_backtest_summary_for_telegram(summary_data, trades_df=None):
    """Format backtest summary for Telegram message."""
    if not summary_data:
        return "Backtest summary data not available."
    
    lines = [
        f"Backtest Run Timestamp: {summary_data.get('Run_Timestamp', 'N/A')}",
        f"Strategy: {summary_data.get('Strategy_Name', 'N/A')}",
        f"Stocks: {', '.join(summary_data.get('Symbols_Tested', ['N/A']))}",
        f"Trades: {summary_data.get('Total_Completed_Trades', 'N/A')}",
        f"PnL Sum: {float(summary_data.get('Overall_PnL_Sum_Pct', 0)):.2f}%",
        f"Win Rate: {float(summary_data.get('Overall_Win_Rate_Pct', 0)):.2f}%",
        f"Avg PnL/Trade: {float(summary_data.get('Avg_PnL_Per_Trade_Pct', 0)):.2f}%",
        f"Sharpe: {float(summary_data.get('Sharpe_Ratio', 0.0)):.2f}", 
        f"Max Drawdown: {float(summary_data.get('Max_Drawdown_Pct', 0.0)):.2f}%",
    ]
    if trades_df is not None and not trades_df.empty:
        lines.append(f"Total Trades Logged: {len(trades_df)}")
    
    notes = summary_data.get('Notes', '')
    if notes:
        lines.append(f"\n Notes: {notes}")
    return "\n".join(lines)

def format_ml_summary_for_telegram(ml_results):
    """Format ML prediction summary for Telegram message."""
    if not ml_results:
        return "ML prediction results not available."

    summary = ml_results.get('summary', {})
    if not summary:
        num_preds = len(ml_results.get('predictions', []))
        num_evals = len(ml_results.get('training_evaluations', []))
        if num_preds == 0 and num_evals == 0:
             return "ML prediction results contained no summary, predictions, or evaluations."
        else:
             fallback_lines = [" ML Prediction Partial Data:"]
             if num_preds > 0: fallback_lines.append(f"  - Predictions found: {num_preds}")
             if num_evals > 0: fallback_lines.append(f"  - Evaluations found: {num_evals}")
             fallback_lines.append("  - Core summary dictionary was missing.")
             return "\n".join(fallback_lines)

    predictions = ml_results.get('predictions', [])
    evaluations = ml_results.get('training_evaluations', [])

    lines = [
        f"ML Prediction Summary (Dashboard)",
        f"Model Type: {summary.get('Model_Type_Used', 'N/A')}",
        f"Target Strategy: {summary.get('Target_Strategy_Used', 'N/A')}",
        f"Stocks Processed: {summary.get('Total_Stocks_Processed', 'N/A')}",
        f"Successful Preds: {summary.get('Successful_Predictions', 'N/A')}",
        f"Avg Max Conf: {float(summary.get('Average_Max_Confidence', 0.0)):.2f}"
    ]

    if predictions:
        lines.append("\n Top Predictions (Sample):")
        for pred in predictions[:min(3, len(predictions))]:
            conf_str = f"{float(pred.get('Max_Confidence', 0.0)):.2f}"
            signal_str = pred.get('Final_Signal_Generated', 'N/A')
            lines.append(f"  - {pred.get('Symbol', 'N/S')}: {pred.get('Predicted_Movement', 'N/A')} "
                         f"(Conf: {conf_str}, Signal: {signal_str})")
    
    if evaluations:
        valid_evals_acc = []
        valid_evals_auc = []
        for e in evaluations:
            if isinstance(e, dict):
                try:
                    acc = e.get('Test_Accuracy')
                    if acc is not None: valid_evals_acc.append(float(acc))
                except (ValueError, TypeError): pass 
                try:
                    auc = e.get('AUC_Score')
                    if auc is not None: valid_evals_auc.append(float(auc))
                except (ValueError, TypeError): pass
        
        avg_test_accuracy = sum(valid_evals_acc) / len(valid_evals_acc) if valid_evals_acc else 0.0
        avg_auc = sum(valid_evals_auc) / len(valid_evals_auc) if valid_evals_auc else 0.0
        
        lines.append("\n Avg Training Performance:")
        lines.append(f"  Avg Test Accuracy: {avg_test_accuracy:.2f}")
        lines.append(f"  Avg AUC Score: {avg_auc:.2f}")

    return "\n".join(lines)

# --- Async Telegram Helper Functions ---
async def initialize_telegram_manager_async():
    """Initialize Telegram Manager with health check."""
    try:
        # TelegramManager() will use the event loop set by run_async_task for its client
        tg_manager = TelegramManager() 
        if await tg_manager.health_check():
            return tg_manager
        else:
            st.error("Telegram health check failed. Notifications may be unreliable.")
            return None
    except ValueError as ve:
        st.error(f"Telegram Configuration Error: {ve}. Check .env variables.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Telegram Manager: {type(e).__name__} - {e}")
        logger.error(f"DEBUG: initialize_telegram_manager_async error: {traceback.format_exc()}")
        return None

async def send_telegram_message_via_manager_async(tg_manager, message, use_summary=False, prefix=""):
    """Send message to Telegram asynchronously using a provided manager instance."""
    if tg_manager:
        try:
            success = await tg_manager.send(message, use_summary=use_summary, prefix=prefix)
            return success
        except Exception as e:
            st.error(f"Telegram: Failed to send message - {type(e).__name__}: {e}")
            logger.error(f"DEBUG: send_telegram_message_via_manager_async error: {traceback.format_exc()}")
            return False
    return False

async def send_telegram_error_via_manager_async(tg_manager, error_details, context=""):
    """Send error alert to Telegram asynchronously using a provided manager instance."""
    if tg_manager:
        try:
            success = await tg_manager.send_error_alert(error_details, context=context)
            return success
        except Exception as e:
            st.error(f"Telegram: Failed to send error alert - {type(e).__name__}: {e}")
            logger.error(f"DEBUG: send_telegram_error_via_manager_async error: {traceback.format_exc()}")
            return False
    return False

# --- Synchronous Wrappers for Streamlit ---

# REMOVE the module-level global_loop definition that was here.

def get_session_event_loop():
    """
    Retrieves or creates an asyncio event loop stored in Streamlit's session state.
    This ensures that the same loop is used across Streamlit script reruns for a given session.
    """
    if 'session_event_loop' not in st.session_state:
        logger.info("Creating new event loop for Streamlit session.")
        st.session_state.session_event_loop = asyncio.new_event_loop()
    return st.session_state.session_event_loop

def run_async_task(async_function, *args, **kwargs):
    """
    Generic synchronous wrapper for running an async task using a session-persistent event loop.
    WARNING: This approach can be fragile, especially if the async library (Telethon)
             starts background tasks. A 'loop is already running' error might occur.
             A more robust solution involves running asyncio logic in a separate thread.
    """
    loop = get_session_event_loop()
    
    # Ensure the session loop is set as the current event loop for this thread's context.
    # This is important because Telethon's TelegramClient() will internally call
    # asyncio.get_event_loop() to associate itself with a loop.
    original_policy = asyncio.get_event_loop_policy()
    current_thread_loop = None
    try:
        current_thread_loop = asyncio.get_event_loop()
        # If the current thread's loop is different from our session loop, set ours.
        if current_thread_loop is not loop:
            asyncio.set_event_loop(loop)
    except RuntimeError: # No current event loop in this thread
        asyncio.set_event_loop(loop)

    result = None
    try:
        if loop.is_running():
            # This is a significant problem if it occurs.
            # It means Telethon (or another part of the async code) has background tasks
            # running on this loop, and run_until_complete cannot be called again.
            logger.error("Attempted to run task on an already running event loop. This indicates a potential architectural issue for Streamlit integration.")
            st.error("Async Conflict: The system is busy with another background task. Please try again. If the problem persists, the Telegram connection might need to be reset or the app restarted.")
            return None

        # logger.info(f"Running async task {async_function.__name__} on loop {id(loop)}")
        result = loop.run_until_complete(async_function(*args, **kwargs))
        # logger.info(f"Finished async task {async_function.__name__}. Loop is_running: {loop.is_running()}")

    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e).lower() or \
           "event loop is closed" in str(e).lower():
            logger.error(f"Event loop was closed or shutdown: {e}. Resetting session loop and Telegram manager.")
            st.error("Critical Async Error: Event loop closed unexpectedly. Telegram connection lost. Please Test/Reconnect Telegram.")
            # Clean up the old, closed loop from session state
            if 'session_event_loop' in st.session_state:
                del st.session_state.session_event_loop 
            # Force re-initialization of Telegram manager on next action
            if 'telegram_manager_instance' in st.session_state:
                st.session_state.telegram_manager_instance = None
            st.session_state.telegram_connection_active = False
            # The next call to get_session_event_loop() will create a new one.
        elif "this event loop is already running" in str(e).lower():
            logger.error(f"RuntimeError: Event loop is already running: {e}. Task: {async_function.__name__}")
            st.error("Async Conflict: Operation could not start as the system is busy. This might happen with concurrent Telegram operations. Please try again.")
        else:
            logger.error(f"Unhandled RuntimeError in run_async_task for {async_function.__name__}: {traceback.format_exc()}")
            st.error(f"An unexpected async runtime error occurred: {e}")
    except Exception as e:
        logger.error(f"Exception in run_async_task for {async_function.__name__}: {traceback.format_exc()}")
        st.error(f"An unexpected error occurred during async execution: {e}")
    finally:
        if current_thread_loop is not None and current_thread_loop is not loop : 
            asyncio.set_event_loop(current_thread_loop)
        elif current_thread_loop is None and loop: 
            pass

    return result


#Initialize Session State 
def init_session_state():
    if 'app_config' not in st.session_state:
        st.session_state.app_config = load_configuration()
    if 'gsheet_manager' not in st.session_state:
        st.session_state.gsheet_manager = None
    
    if 'telegram_manager_instance' not in st.session_state:
        st.session_state.telegram_manager_instance = None 
    if 'telegram_connection_active' not in st.session_state:
        st.session_state.telegram_connection_active = False
    if 'telegram_auto_notifications_enabled' not in st.session_state:
        st.session_state.telegram_auto_notifications_enabled = True

    if 'backtest_summary' not in st.session_state:
        st.session_state.backtest_summary = None
    if 'backtest_trades' not in st.session_state:
        st.session_state.backtest_trades = pd.DataFrame()
    if 'ml_predictions' not in st.session_state:
        st.session_state.ml_predictions = None
    if 'active_stocks' not in st.session_state:
        st.session_state.active_stocks = st.session_state.app_config.get('stocks_to_track', [])
    if 'config_updated' not in st.session_state:
        st.session_state.config_updated = False
    if 'execution_completed' not in st.session_state:
        st.session_state.execution_completed = False

init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls & Configuration")

    # Config Management
    st.subheader("Configuration")
    uploaded_config = st.file_uploader("Upload config.yaml", type=['yaml', 'yml'])
    if uploaded_config and not st.session_state.config_updated:
        try:
            new_config_data = yaml.safe_load(uploaded_config)
            if new_config_data:
                st.session_state.app_config = new_config_data
                save_configuration(st.session_state.app_config)
                st.session_state.config_updated = True
                st.success("Config uploaded and saved.")
                st.rerun()
        except yaml.YAMLError as e: st.error(f"Invalid YAML file: {e}")
        except Exception as e: st.error(f"Error processing config file: {e}")
    if not uploaded_config: st.session_state.config_updated = False
    if st.button("Reload Default Config"):
        st.session_state.app_config = load_configuration()
        st.session_state.active_stocks = st.session_state.app_config.get('stocks_to_track', [])
        st.success("Default config reloaded.")
        st.rerun()
    
    # Stock Selection
    st.subheader("Stock Selection")
    stocks_text = ", ".join(st.session_state.active_stocks)
    new_stocks_text = st.text_area("Stocks to Track (comma-separated):", value=stocks_text, height=100)
    if st.button("Update Active Stocks"):
        new_stocks = [s.strip().upper() for s in new_stocks_text.split(',') if s.strip()]
        if new_stocks != st.session_state.active_stocks:
            st.session_state.active_stocks = new_stocks
            current_config = st.session_state.app_config
            current_config['stocks_to_track'] = st.session_state.active_stocks
            save_configuration(current_config)
            st.success(f"Active stocks updated to: {st.session_state.active_stocks}")
            st.rerun()

    st.markdown("---")
    
    # Telegram Connection
    # Load Font Awesome for Telegram icon (once at top of app)
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)

    # --- Telegram Notifications Section ---
    st.markdown("""
    <h3 style='display: flex; align-items: center; gap: 10px;'>
        <i class="fa-brands fa-telegram" style="color:#229ED9;"></i>
        Telegram Notifications
    </h3>
    """, unsafe_allow_html=True)
    if st.button("connect Telegram", key="connect_telegram_btn"):
        with st.spinner("Attempting to connect to Telegram & run health check..."):
            # If a manager instance exists, try to close it first to release resources,
            # especially if its loop was from a previous, potentially problematic state.
            if st.session_state.telegram_manager_instance:
                logger.info("Closing existing Telegram manager before reconnecting.")
                st.session_state.telegram_manager_instance = None
                st.session_state.telegram_connection_active = False
            
            manager_instance = run_async_task(initialize_telegram_manager_async)
            if manager_instance:
                st.session_state.telegram_manager_instance = manager_instance
                st.session_state.telegram_connection_active = True
            else:
                st.session_state.telegram_manager_instance = None
                st.session_state.telegram_connection_active = False
                st.error("Telegram connection/health check failed. See messages above.")
        st.rerun()

    if st.session_state.telegram_connection_active:
        st.success("✅ Telegram: Connected")
    else:
        st.warning("⚠️ Telegram: Disconnected")

    auto_notify_current_val = st.session_state.get('telegram_auto_notifications_enabled', True)
    auto_notify_checkbox = st.checkbox(
        "📨 Send Auto-Notifications", 
        value=auto_notify_current_val,
        help="Automatically send results/errors to Telegram after execution if connected.",
        disabled=not st.session_state.telegram_connection_active
    )
    if auto_notify_checkbox != auto_notify_current_val:
        st.session_state.telegram_auto_notifications_enabled = auto_notify_checkbox
        st.rerun()

    st.markdown("---")
    
    # GSheet Connection
    if st.session_state.app_config.get('google_sheets', {}).get('enabled_for_app', False):
        if st.session_state.gsheet_manager is None:
            if st.button("🔗 Connect Google Sheets"):
                with st.spinner("Connecting to Google Sheets..."):
                    st.session_state.gsheet_manager = initialize_gsheet_manager(st.session_state.app_config)
                if st.session_state.gsheet_manager: st.success("Google Sheets Connected!")
                else: st.error("Failed to connect to Google Sheets.")
                st.rerun()
        else:
            st.success("🔗 GSheets: Connected")
    else:
        st.info("Google Sheets is disabled in config.")
    
    st.markdown("---")

    # Run Controls
    st.subheader("Run Modules")
    run_bt_cb = st.checkbox("Backtester", value=st.session_state.app_config.get('run_backtester', True))
    run_ml_cb = st.checkbox(" ML Predictor", value=st.session_state.app_config.get('run_ml_predictor', True))

    if st.button(" Run Modules", type="primary"):
        st.session_state.app_config['run_backtester'] = run_bt_cb
        st.session_state.app_config['run_ml_predictor'] = run_ml_cb
        save_configuration(st.session_state.app_config)

        st.session_state.backtest_summary = None
        st.session_state.backtest_trades = pd.DataFrame()
        st.session_state.ml_predictions = None
        st.session_state.execution_completed = False

        active_tg_manager = st.session_state.telegram_manager_instance
        send_notifications = st.session_state.telegram_connection_active and \
                             st.session_state.telegram_auto_notifications_enabled and \
                             active_tg_manager is not None


        if not st.session_state.active_stocks:
            st.warning("No stocks selected to run. Please add stocks in the sidebar.")
            if send_notifications:
                run_async_task(send_telegram_message_via_manager_async, 
                               active_tg_manager,
                               "⚠️ Execution attempted but no stocks selected!",
                               prefix="📱 Dashboard Warning: ")
        else:
            execution_success_overall = True
            
            if send_notifications:
                start_msg_parts = [f"⏳ Starting execution for: {', '.join(st.session_state.active_stocks)}"]
                if run_bt_cb: start_msg_parts.append("Backtester: ✅")
                if run_ml_cb: start_msg_parts.append("ML Predictor: ✅")
                run_async_task(send_telegram_message_via_manager_async, active_tg_manager, "\n".join(start_msg_parts), prefix="📱 Exec: ")
            
            if run_bt_cb:
                with st.spinner("Executing Backtester..."):
                    try:
                        bt_results = run_backtesting_logic(st.session_state.app_config, st.session_state.gsheet_manager, st.session_state.active_stocks)
                        if bt_results and "error" not in bt_results:
                            st.session_state.backtest_summary = bt_results.get('summary')
                            st.session_state.backtest_trades = bt_results.get('all_trades', pd.DataFrame())
                            st.success("Backtester executed successfully!")
                            if send_notifications and st.session_state.backtest_summary:
                                bt_report = format_backtest_summary_for_telegram(st.session_state.backtest_summary, st.session_state.backtest_trades)
                                run_async_task(send_telegram_message_via_manager_async, active_tg_manager, bt_report, use_summary=True, prefix="📊 BT Report (App):\n")
                        else:
                            err_msg = bt_results.get('error', "Unknown backtester error.") if isinstance(bt_results, dict) else "Backtester returned invalid data."
                            st.error(f"Backtester Error: {err_msg}")
                            execution_success_overall = False
                            if send_notifications:
                                run_async_task(send_telegram_error_via_manager_async, active_tg_manager, f"Backtester Error: {err_msg}", context="Dashboard BT")
                    except Exception as e_bt:
                        tb_str = traceback.format_exc()
                        st.error(f"Backtester execution CRASHED: {str(e_bt)}")
                        logger.error(f"Backtester CRASH: {tb_str}")
                        execution_success_overall = False
                        if send_notifications:
                            run_async_task(send_telegram_error_via_manager_async, active_tg_manager, f"Backtester CRASH: {str(e_bt)}\n{tb_str}", context="Dashboard BT Crash")
            
            if run_ml_cb:
                with st.spinner("Executing ML Predictor..."):
                    try:
                        ml_res = run_ml_prediction_logic(st.session_state.app_config, st.session_state.gsheet_manager, st.session_state.active_stocks)
                        if ml_res and "error" not in ml_res:
                            st.session_state.ml_predictions = ml_res
                            st.success("ML Predictor executed successfully!")
                            if send_notifications and st.session_state.ml_predictions:
                                ml_report = format_ml_summary_for_telegram(st.session_state.ml_predictions)
                                run_async_task(send_telegram_message_via_manager_async, active_tg_manager, ml_report, use_summary=True, prefix="🤖 ML Report (App):\n")
                        else:
                            err_msg = ml_res.get('error', "Unknown ML predictor error.") if isinstance(ml_res, dict) else "ML Predictor returned invalid data."
                            st.error(f"ML Predictor Error: {err_msg}")
                            execution_success_overall = False
                            if send_notifications:
                                run_async_task(send_telegram_error_via_manager_async, active_tg_manager, f"ML Predictor Error: {err_msg}", context="Dashboard ML")
                    except Exception as e_ml:
                        tb_str = traceback.format_exc()
                        st.error(f"ML Predictor execution CRASHED: {str(e_ml)}")
                        logger.error(f"ML Predictor CRASH: {tb_str}")
                        execution_success_overall = False
                        if send_notifications:
                           run_async_task(send_telegram_error_via_manager_async, active_tg_manager, f"ML Predictor CRASH: {str(e_ml)}\n{tb_str}", context="Dashboard ML Crash")
            
            st.session_state.execution_completed = True
            
            final_msg_text = "✅Executed successfully!" if execution_success_overall else "⚠️ Execution completed with some errors."
            if execution_success_overall: st.success(final_msg_text)
            else: st.error(final_msg_text)

            if send_notifications:
                run_async_task(send_telegram_message_via_manager_async, active_tg_manager, final_msg_text, prefix="Exec Complete: ")
            
            st.rerun()

# --- Main Dashboard Area ---
st.header("Dashboard & Results")

# --- Backtesting Dashboard ---
if st.session_state.app_config.get('run_backtester', True):
    st.subheader("Backtesting Analysis")
    if st.session_state.backtest_summary:
        summary = st.session_state.backtest_summary
        trades_df = st.session_state.backtest_trades

        st.markdown("### Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", summary.get('Total_Completed_Trades', 'N/A'))
        col2.metric("Total PnL (%)", f"{float(summary.get('Overall_PnL_Sum_Pct', 0)):.2f}%")
        col3.metric("Win Rate (%)", f"{float(summary.get('Overall_Win_Rate_Pct', 0)):.2f}%")
        col4.metric("Avg PnL / Trade (%)", f"{float(summary.get('Avg_PnL_Per_Trade_Pct', 0)):.2f}%")

        st.caption(f" Run Timestamp: {summary.get('Run_Timestamp', 'N/A')}")
        st.caption(f" Notes: {summary.get('Notes', 'N/A')}")

        if trades_df is not None and not trades_df.empty:
            st.markdown("###  Trade-Level Analysis")
            tab1_bt, tab2_bt, tab3_bt = st.tabs([" Trades Table", " Cumulative PnL", " PnL Distribution"])
            with tab1_bt:
                st.dataframe(trades_df.style.format({"Entry_Price": "{:.2f}", "Exit_Price": "{:.2f}", "PnL_Percent": "{:.2f}%", "Holding_Period": "{:.0f} days"}), use_container_width=True)
            with tab2_bt:
                if 'PnL_Percent' in trades_df.columns:
                    sort_col = 'Exit_Date' if 'Exit_Date' in trades_df.columns else trades_df.index
                    trades_df_sorted = trades_df.sort_values(by=sort_col).copy()
                    trades_df_sorted['Cumulative_PnL_Percent'] = trades_df_sorted['PnL_Percent'].cumsum()
                    fig_cum = px.line(trades_df_sorted, x=trades_df_sorted.index, y='Cumulative_PnL_Percent', markers=True, title=" Cumulative PnL (%) Over Trades")
                    st.plotly_chart(fig_cum, use_container_width=True)
                else: st.warning("PnL_Percent column missing for Cumulative PnL chart.")
            with tab3_bt:
                if 'PnL_Percent' in trades_df.columns:
                    fig_hist = px.histogram(trades_df, x="PnL_Percent", nbins=20, title=" Distribution of Trade PnL (%)")
                    st.plotly_chart(fig_hist, use_container_width=True)
                else: st.warning("PnL_Percent column missing for PnL Distribution chart.")
        elif st.session_state.execution_completed:
            st.info("No trades data from the last backtest run.")
    elif st.session_state.execution_completed:
        st.info("Backtest did not produce a summary or failed.")
    else:
        st.info("Run the backtester to view analysis.")
    st.markdown("---")

#Ml-board
if st.session_state.app_config.get('run_ml_predictor', False):
    st.subheader("Machine Learning Predictions Analysis")
    if st.session_state.ml_predictions and "error" not in st.session_state.ml_predictions:
        ml_summary = st.session_state.ml_predictions.get('summary', {})
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Stocks Processed", ml_summary.get('Total_Stocks_Processed', 'N/A'))
        col2.metric("Successful Predictions", ml_summary.get('Successful_Predictions', 'N/A'))
        col3.metric("Avg. Max Confidence", ml_summary.get('Average_Max_Confidence', 'N/A'))
        
        st.text(f"Model Type: {ml_summary.get('Model_Type_Used', 'N/A')}")
        st.text(f"Target Strategy: {ml_summary.get('Target_Strategy_Used', 'N/A')}")
        
        predictions_data = st.session_state.ml_predictions.get('predictions', [])
        if predictions_data:
            preds_df = pd.DataFrame(predictions_data)
            
            # visualization tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Predictions Overview", " Confidence Analysis", " Performance Metrics", "Data Table"])
            
            with tab1:
                if 'Predicted_Movement' in preds_df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        movement_counts = preds_df['Predicted_Movement'].value_counts()
                        fig_pred_dist = px.pie(
                            values=movement_counts.values, 
                            names=movement_counts.index,
                            title="Distribution of Predicted Movements",
                            color_discrete_map={
                                'Up': '#00CC96', 
                                'Down': '#EF553B', 
                                'Neutral': '#FFA15A'
                            }
                        )
                        fig_pred_dist.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pred_dist, use_container_width=True)
                    
                    with col2:
                        # Bar chart version
                        fig_bar = px.bar(
                            movement_counts, 
                            x=movement_counts.index, 
                            y=movement_counts.values,
                            title="Prediction Counts by Movement",
                            color=movement_counts.index,
                            color_discrete_map={
                                'Up': '#00CC96', 
                                'Down': '#EF553B', 
                                'Neutral': '#FFA15A'
                            }
                        )
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab2:
                # Confidence analysis with scatter plots
                if 'Max_Confidence' in preds_df.columns:
                    # Convert 'Max_Confidence' to numeric
                    preds_df['Max_Confidence'] = pd.to_numeric(preds_df['Max_Confidence'], errors='coerce')
                    preds_df = preds_df.dropna(subset=['Max_Confidence'])

                    # Scatter plot: Confidence vs Symbol
                    fig_confidence_scatter = px.scatter(
                        preds_df,
                        x='Symbol' if 'Symbol' in preds_df.columns else range(len(preds_df)),
                        y='Max_Confidence',
                        color_discrete_map={
                            'Up': '#00CC96', 
                            'Down': '#EF553B', 
                            'Neutral': '#FFA15A'
                        }
                    )
                    fig_confidence_scatter.update_layout(
                        xaxis_title="Stock Symbol",
                        yaxis_title="Max Confidence",
                        height=500
                    )
                    if 'Symbol' in preds_df.columns:
                        fig_confidence_scatter.update_layout(xaxis=dict(tickangle=45))
                    st.plotly_chart(fig_confidence_scatter, use_container_width=True)
                    
                    # Confidence distribution histogram
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_conf_hist = px.histogram(
                            preds_df,
                            x='Max_Confidence',
                            nbins=20,
                            title="Confidence Score Distribution",
                            color_discrete_sequence=['#636EFA']
                        )
                        st.plotly_chart(fig_conf_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot of confidence by movement
                        if 'Predicted_Movement' in preds_df.columns:
                            fig_conf_box = px.box(
                                preds_df,
                                x='Predicted_Movement',
                                y='Max_Confidence',
                                title="Confidence by Predicted Movement",
                                color='Predicted_Movement',
                                color_discrete_map={
                                    'Up': '#00CC96', 
                                    'Down': '#EF553B', 
                                    'Neutral': '#FFA15A'
                                }
                            )
                            fig_conf_box.update_layout(showlegend=False)
                            st.plotly_chart(fig_conf_box, use_container_width=True)
            
            with tab3:
                # Training evaluations with enhanced scatter plots
                training_evals = st.session_state.ml_predictions.get('training_evaluations', [])
                if training_evals:
                    evals_df = pd.DataFrame(training_evals)
                    
                    if not evals_df.empty:
                        # Multi-metric scatter plot
                        if all(col in evals_df.columns for col in ['Test_Accuracy', 'AUC_Score']):
                            # Convert numeric columns to ensure they're numeric
                            numeric_cols = ['Test_Accuracy', 'AUC_Score', 'CV_Accuracy_Mean']
                            for col in numeric_cols:
                                if col in evals_df.columns:
                                    evals_df[col] = pd.to_numeric(evals_df[col], errors='coerce')
                            
                            # Only use size parameter if CV_Accuracy_Mean is numeric and has valid data
                            size_param = None
                            if 'CV_Accuracy_Mean' in evals_df.columns and evals_df['CV_Accuracy_Mean'].notna().any():
                                size_param = 'CV_Accuracy_Mean'
                            
                            fig_metrics_scatter = px.scatter(
                                evals_df,
                                x='Test_Accuracy',
                                y='AUC_Score',
                                color='Model_Type' if 'Model_Type' in evals_df.columns else None,
                                size=size_param,
                                hover_name='Symbol' if 'Symbol' in evals_df.columns else None,
                                title="Model Performance: Test Accuracy vs AUC Score",
                                hover_data=['Selected_Features_Count'] if 'Selected_Features_Count' in evals_df.columns else None
                            )
                            fig_metrics_scatter.update_layout(height=500)
                            st.plotly_chart(fig_metrics_scatter, use_container_width=True)
                        
                        # Performance metrics comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'Test_Accuracy' in evals_df.columns:
                                fig_acc_dist = px.histogram(
                                    evals_df,
                                    x='Test_Accuracy',
                                    title="Test Accuracy Distribution",
                                    nbins=15,
                                    color_discrete_sequence=['#AB63FA']
                                )
                                st.plotly_chart(fig_acc_dist, use_container_width=True)
                        
                        with col2:
                            if 'Selected_Features_Count' in evals_df.columns:
                                fig_features = px.scatter(
                                    evals_df,
                                    x='Selected_Features_Count',
                                    y='Test_Accuracy' if 'Test_Accuracy' in evals_df.columns else 'AUC_Score',
                                    color='Model_Type' if 'Model_Type' in evals_df.columns else None,
                                    title="Features vs Performance",
                                    hover_name='Symbol' if 'Symbol' in evals_df.columns else None
                                )
                                st.plotly_chart(fig_features, use_container_width=True)
                        
                        # Model comparison radar chart (if multiple models)
                        if 'Model_Type' in evals_df.columns and len(evals_df['Model_Type'].unique()) > 1:
                            # Ensure numeric columns are properly converted
                            numeric_cols = ['Test_Accuracy', 'AUC_Score', 'CV_Accuracy_Mean']
                            evals_df_numeric = evals_df.copy()
                            for col in numeric_cols:
                                if col in evals_df_numeric.columns:
                                    evals_df_numeric[col] = pd.to_numeric(evals_df_numeric[col], errors='coerce')
                            
                            model_avg = evals_df_numeric.groupby('Model_Type').agg({
                                'Test_Accuracy': 'mean',
                                'AUC_Score': 'mean',
                                'CV_Accuracy_Mean': 'mean'
                            }).reset_index()
                            
                            fig_radar = go.Figure()
                            
                            for model in model_avg['Model_Type'].unique():
                                model_data = model_avg[model_avg['Model_Type'] == model].iloc[0]
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[model_data['Test_Accuracy'], model_data['AUC_Score'], model_data['CV_Accuracy_Mean']],
                                    theta=['Test Accuracy', 'AUC Score', 'CV Accuracy'],
                                    fill='toself',
                                    name=model
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                title="Model Performance Comparison",
                                height=500
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab4:
                # Enhanced data table with filtering options
                st.subheader("Predictions Data")
                
                # Add filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'Predicted_Movement' in preds_df.columns:
                        movement_filter = st.multiselect(
                            "Filter by Predicted Movement",
                            options=preds_df['Predicted_Movement'].unique(),
                            default=preds_df['Predicted_Movement'].unique()
                        )
                        preds_df = preds_df[preds_df['Predicted_Movement'].isin(movement_filter)]
                
                with col2:
                    if 'Max_Confidence' in preds_df.columns:
                        conf_min = st.slider(
                            "Minimum Confidence",
                            min_value=float(preds_df['Max_Confidence'].min()),
                            max_value=float(preds_df['Max_Confidence'].max()),
                            value=float(preds_df['Max_Confidence'].min())
                        )
                        preds_df = preds_df[preds_df['Max_Confidence'] >= conf_min]
                
                # Display filtered dataframe
                st.dataframe(
                    preds_df,
                    use_container_width=True,
                    height=400
                )
                
                # Training evaluations table
                if training_evals:
                    st.subheader("Training Evaluations")
                    evals_df = pd.DataFrame(training_evals)
                    
                    if not evals_df.empty and all(col in evals_df.columns for col in ['Symbol', 'Model_Type']):
                        display_cols = ['Symbol', 'Model_Type']
                        for col in ['Test_Accuracy', 'AUC_Score', 'CV_Accuracy_Mean', 'Selected_Features_Count']:
                            if col in evals_df.columns:
                                display_cols.append(col)
                        
                        # Color-code accuracy columns
                        styled_df = evals_df[display_cols].style.background_gradient(
                            subset=[col for col in ['Test_Accuracy', 'AUC_Score', 'CV_Accuracy_Mean'] if col in display_cols],
                            cmap='RdYlGn'
                        )
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        st.dataframe(evals_df, use_container_width=True)
        
        else:
            st.warning("No prediction data available to visualize.")
    
    else:
        st.info("Run the ML Predictor or load results to view analysis.")
# --- Tab for Config Editing ---
with st.expander("⚙️ Advanced Configuration Editor (YAML)"):
    if st.session_state.app_config:
        try:
            config_str_display = yaml.dump(st.session_state.app_config, sort_keys=False, indent=2)
            edited_config_str_display = st.text_area("Current Configuration (YAML):", value=config_str_display, height=400, key="yaml_editor")
            if st.button("Apply & Save Edited YAML Configuration"):
                try:
                    new_config_data_yaml = yaml.safe_load(edited_config_str_display)
                    if new_config_data_yaml != st.session_state.app_config:
                        st.session_state.app_config = new_config_data_yaml
                        save_configuration(st.session_state.app_config)
                        st.success("YAML Configuration updated and saved.")
                        st.rerun()
                    else: st.info("No changes detected in YAML configuration.")
                except yaml.YAMLError as e: st.error(f"Invalid YAML: {e}")
        except Exception as e_yaml: st.error(f"Error with YAML display/edit: {e_yaml}")

st.sidebar.markdown("---")
st.sidebar.caption(f"Trading Engine UI v0.2")

if st.sidebar.checkbox("Load Data from Google Sheets (if not run locally)"):
    if st.session_state.gsheet_manager:
        if st.sidebar.button("Fetch Backtest Data (GSheets)"):
            try:
                st.info("GSheet BT data fetching logic from your original file.")
            except Exception as e: st.error(f"Error fetching backtest data: {str(e)}")
        if st.sidebar.button("Fetch ML Data (GSheets)"):
            try:
                st.info("GSheet ML data fetching logic from your original file.")
            except Exception as e: st.error(f"Error fetching ML data: {str(e)}")
    else:
        st.sidebar.warning("Google Sheets not connected.")