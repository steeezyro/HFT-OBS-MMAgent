"""HTML reporting system for HFT market making strategies."""

import base64
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, Template
from plotly.subplots import make_subplots

from .backtest import Backtester


class ReportGenerator:
    """
    HTML report generator for market making strategy performance.
    
    Generates comprehensive tear sheets with:
    - Performance overview
    - Risk metrics
    - Trade analytics
    - Order book analytics
    - Latency analysis
    - RL training curves
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # HTML template
        self.template = self._create_template()
    
    def generate_report(
        self,
        backtest_results: Dict[str, Any],
        portfolio_history: pd.DataFrame,
        trades: List[Dict],
        latency_data: Optional[pd.DataFrame] = None,
        training_data: Optional[pd.DataFrame] = None,
        title: str = "HFT Market Making Strategy Report",
        subtitle: str = "Performance Analysis & Risk Metrics",
    ) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            backtest_results: Dictionary with backtest metrics
            portfolio_history: DataFrame with portfolio evolution
            trades: List of trade dictionaries
            latency_data: Optional latency measurements
            training_data: Optional RL training metrics
            title: Report title
            subtitle: Report subtitle
            
        Returns:
            Path to generated HTML report
        """
        # Generate all plots
        plots = self._generate_all_plots(
            backtest_results, portfolio_history, trades, latency_data, training_data
        )
        
        # Prepare data for template
        template_data = {
            'title': title,
            'subtitle': subtitle,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self._format_metrics(backtest_results),
            'plots': plots,
            'trade_summary': self._generate_trade_summary(trades),
            'risk_analysis': self._generate_risk_analysis(portfolio_history),
        }
        
        # Render HTML
        html_content = self.template.render(**template_data)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hft_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated: {filepath}")
        return filepath
    
    def _generate_all_plots(
        self,
        backtest_results: Dict[str, Any],
        portfolio_history: pd.DataFrame,
        trades: List[Dict],
        latency_data: Optional[pd.DataFrame],
        training_data: Optional[pd.DataFrame],
    ) -> Dict[str, str]:
        """Generate all plots and return as base64 encoded strings."""
        plots = {}
        
        # 1. Portfolio performance
        plots['portfolio_performance'] = self._plot_portfolio_performance(portfolio_history)
        
        # 2. Drawdown analysis
        plots['drawdown'] = self._plot_drawdown_analysis(portfolio_history)
        
        # 3. Returns distribution
        plots['returns_distribution'] = self._plot_returns_distribution(portfolio_history)
        
        # 4. Trade analytics
        if trades:
            plots['trade_analytics'] = self._plot_trade_analytics(trades)
            plots['trade_heatmap'] = self._plot_trade_heatmap(trades)
        
        # 5. Risk metrics over time
        plots['risk_metrics'] = self._plot_risk_metrics_over_time(portfolio_history)
        
        # 6. Inventory management
        plots['inventory'] = self._plot_inventory_analysis(portfolio_history)
        
        # 7. Latency analysis
        if latency_data is not None:
            plots['latency_distribution'] = self._plot_latency_distribution(latency_data)
            plots['latency_trend'] = self._plot_latency_trend(latency_data)
        
        # 8. RL training curves
        if training_data is not None:
            plots['training_curves'] = self._plot_training_curves(training_data)
            plots['learning_progress'] = self._plot_learning_progress(training_data)
        
        return plots
    
    def _plot_portfolio_performance(self, portfolio_history: pd.DataFrame) -> str:
        """Generate portfolio performance plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative PnL', 'Returns'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Cumulative PnL
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'],
                y=portfolio_history['total_pnl'],
                mode='lines',
                name='Total PnL',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Returns
        returns = portfolio_history['total_pnl'].pct_change()
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'],
                y=returns,
                mode='lines',
                name='Returns',
                line=dict(color='#ff7f0e', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Returns", row=2, col=1)
        
        return self._fig_to_base64(fig)
    
    def _plot_drawdown_analysis(self, portfolio_history: pd.DataFrame) -> str:
        """Generate drawdown analysis plot."""
        pnl = portfolio_history['total_pnl']
        cummax = pnl.cummax()
        drawdown = cummax - pnl
        
        fig = go.Figure()
        
        # Drawdown area
        fig.add_trace(go.Scatter(
            x=portfolio_history['timestamp'],
            y=-drawdown,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        # Add maximum drawdown line
        max_dd_idx = drawdown.idxmax()
        max_dd_value = drawdown.max()
        
        fig.add_hline(
            y=-max_dd_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max DD: ${max_dd_value:.2f}"
        )
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Time",
            yaxis_title="Drawdown ($)",
            height=400
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_returns_distribution(self, portfolio_history: pd.DataFrame) -> str:
        """Generate returns distribution plot."""
        returns = portfolio_history['total_pnl'].pct_change().dropna()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Q-Q plot
        import scipy.stats as stats
        qq_data = stats.probplot(returns, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='blue', size=4)
            ),
            row=1, col=2
        )
        
        # Q-Q line
        slope, intercept = qq_data[1]
        x_line = np.array([qq_data[0][0].min(), qq_data[0][0].max()])
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Returns Distribution Analysis",
            height=400
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_trade_analytics(self, trades: List[Dict]) -> str:
        """Generate trade analytics plot."""
        trades_df = pd.DataFrame(trades)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trade Size Distribution', 'PnL per Trade', 
                          'Trades by Hour', 'Cumulative Trade Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Trade size distribution
        trade_sizes = trades_df['quantity'].abs()
        fig.add_trace(
            go.Histogram(x=trade_sizes, nbinsx=30, name='Trade Sizes'),
            row=1, col=1
        )
        
        # PnL per trade
        pnl_per_trade = trades_df['pnl_after'].diff().dropna()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pnl_per_trade))),
                y=pnl_per_trade,
                mode='markers',
                name='PnL per Trade',
                marker=dict(
                    color=pnl_per_trade,
                    colorscale='RdYlGn',
                    size=6
                )
            ),
            row=1, col=2
        )
        
        # Trades by hour
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        trades_by_hour = trades_df.groupby('hour').size()
        fig.add_trace(
            go.Bar(x=trades_by_hour.index, y=trades_by_hour.values, name='Trades by Hour'),
            row=2, col=1
        )
        
        # Cumulative trade count
        cumulative_trades = np.arange(1, len(trades_df) + 1)
        fig.add_trace(
            go.Scatter(
                x=trades_df['timestamp'],
                y=cumulative_trades,
                mode='lines',
                name='Cumulative Trades'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Trade Analytics Dashboard",
            height=600,
            showlegend=False
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_trade_heatmap(self, trades: List[Dict]) -> str:
        """Generate trade timing heatmap."""
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day'] = trades_df['timestamp'].dt.day_name()
        
        # Create heatmap data
        heatmap_data = trades_df.groupby(['day', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex([day for day in day_order if day in heatmap_data.index])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title="Trade Activity Heatmap",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_risk_metrics_over_time(self, portfolio_history: pd.DataFrame) -> str:
        """Generate risk metrics over time plot."""
        # Calculate rolling metrics
        window = min(100, len(portfolio_history) // 10)
        returns = portfolio_history['total_pnl'].pct_change().dropna()
        
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Volatility', 'Rolling Sharpe Ratio'),
            vertical_spacing=0.1
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'][window:],
                y=rolling_vol.iloc[window:],
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        
        # Rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'][window:],
                y=rolling_sharpe.iloc[window:],
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Risk Metrics Over Time",
            height=500
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_inventory_analysis(self, portfolio_history: pd.DataFrame) -> str:
        """Generate inventory analysis plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Inventory Level', 'Inventory Distribution'),
            vertical_spacing=0.1
        )
        
        # Inventory over time
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'],
                y=portfolio_history['inventory'],
                mode='lines',
                name='Inventory',
                line=dict(color='purple')
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Inventory distribution
        fig.add_trace(
            go.Histogram(
                x=portfolio_history['inventory'],
                nbinsx=50,
                name='Inventory Distribution',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Inventory Management Analysis",
            height=500
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_latency_distribution(self, latency_data: pd.DataFrame) -> str:
        """Generate latency distribution plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Latency Distribution', 'Latency Percentiles'),
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=latency_data['latency_us'],
                nbinsx=50,
                name='Latency Distribution'
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=latency_data['latency_us'],
                name='Latency Box Plot',
                boxpoints='outliers'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Latency Analysis",
            height=400
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_latency_trend(self, latency_data: pd.DataFrame) -> str:
        """Generate latency trend plot."""
        # Calculate rolling statistics
        window = min(1000, len(latency_data) // 10)
        rolling_mean = latency_data['latency_us'].rolling(window).mean()
        rolling_p50 = latency_data['latency_us'].rolling(window).quantile(0.5)
        rolling_p95 = latency_data['latency_us'].rolling(window).quantile(0.95)
        rolling_p99 = latency_data['latency_us'].rolling(window).quantile(0.99)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=latency_data.index,
            y=rolling_mean,
            mode='lines',
            name='Mean Latency',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=latency_data.index,
            y=rolling_p50,
            mode='lines',
            name='P50 Latency',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=latency_data.index,
            y=rolling_p95,
            mode='lines',
            name='P95 Latency',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=latency_data.index,
            y=rolling_p99,
            mode='lines',
            name='P99 Latency',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Latency Trend Analysis",
            xaxis_title="Time",
            yaxis_title="Latency (μs)",
            height=400
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_training_curves(self, training_data: pd.DataFrame) -> str:
        """Generate RL training curves plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Reward', 'Episode Length', 'Learning Rate', 'Loss'),
        )
        
        # Episode reward
        fig.add_trace(
            go.Scatter(
                x=training_data['episode'],
                y=training_data['episode_reward'],
                mode='lines',
                name='Episode Reward'
            ),
            row=1, col=1
        )
        
        # Episode length
        fig.add_trace(
            go.Scatter(
                x=training_data['episode'],
                y=training_data['episode_length'],
                mode='lines',
                name='Episode Length'
            ),
            row=1, col=2
        )
        
        # Learning rate
        if 'learning_rate' in training_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=training_data['episode'],
                    y=training_data['learning_rate'],
                    mode='lines',
                    name='Learning Rate'
                ),
                row=2, col=1
            )
        
        # Loss
        if 'loss' in training_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=training_data['episode'],
                    y=training_data['loss'],
                    mode='lines',
                    name='Loss'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="RL Training Progress",
            height=600,
            showlegend=False
        )
        
        return self._fig_to_base64(fig)
    
    def _plot_learning_progress(self, training_data: pd.DataFrame) -> str:
        """Generate learning progress analysis."""
        # Calculate moving averages
        window = min(100, len(training_data) // 10)
        ma_reward = training_data['episode_reward'].rolling(window).mean()
        
        fig = go.Figure()
        
        # Raw rewards
        fig.add_trace(go.Scatter(
            x=training_data['episode'],
            y=training_data['episode_reward'],
            mode='lines',
            name='Episode Reward',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=training_data['episode'],
            y=ma_reward,
            mode='lines',
            name=f'MA-{window} Reward',
            line=dict(color='darkblue', width=2)
        ))
        
        fig.update_layout(
            title="Learning Progress Analysis",
            xaxis_title="Episode",
            yaxis_title="Reward",
            height=400
        )
        
        return self._fig_to_base64(fig)
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Format metrics for display."""
        formatted = {}
        
        formatters = {
            'total_return': lambda x: f"{x:.2%}",
            'annualized_return': lambda x: f"{x:.2%}",
            'volatility': lambda x: f"{x:.2%}",
            'sharpe_ratio': lambda x: f"{x:.3f}",
            'sortino_ratio': lambda x: f"{x:.3f}",
            'max_drawdown': lambda x: f"{x:.2%}",
            'calmar_ratio': lambda x: f"{x:.3f}",
            'win_rate': lambda x: f"{x:.2%}",
            'profit_factor': lambda x: f"{x:.2f}",
        }
        
        for key, value in metrics.items():
            if key in formatters:
                formatted[key] = formatters[key](value)
            elif isinstance(value, (int, float)):
                formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def _generate_trade_summary(self, trades: List[Dict]) -> Dict[str, Any]:
        """Generate trade summary statistics."""
        if not trades:
            return {}
        
        trades_df = pd.DataFrame(trades)
        
        return {
            'total_trades': len(trades),
            'avg_trade_size': trades_df['quantity'].abs().mean(),
            'largest_trade': trades_df['quantity'].abs().max(),
            'total_volume': trades_df['quantity'].abs().sum(),
            'avg_price': trades_df['price'].mean(),
            'price_range': f"{trades_df['price'].min():.2f} - {trades_df['price'].max():.2f}",
        }
    
    def _generate_risk_analysis(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Generate risk analysis summary."""
        if portfolio_history.empty:
            return {}
        
        returns = portfolio_history['total_pnl'].pct_change().dropna()
        
        return {
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
        }
    
    def _fig_to_base64(self, fig) -> str:
        """Convert Plotly figure to base64 string."""
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def _create_template(self) -> Template:
        """Create Jinja2 template for HTML report."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }
        .header h2 {
            color: #666;
            margin: 10px 0 0 0;
            font-weight: normal;
        }
        .generated-at {
            color: #999;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .metric-card .value {
            font-size: 1.8em;
            font-weight: bold;
            margin: 0;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .plot-container {
            text-align: center;
            margin-bottom: 30px;
            background: white;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .summary-table th,
        .summary-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .summary-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <h2>{{ subtitle }}</h2>
            <div class="generated-at">Generated on {{ generated_at }}</div>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                {% for key, value in metrics.items() %}
                <div class="metric-card">
                    <h3>{{ key.replace('_', ' ').title() }}</h3>
                    <div class="value">{{ value }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        {% if plots.portfolio_performance %}
        <div class="section">
            <h2>Portfolio Performance</h2>
            <div class="plot-container">
                <img src="{{ plots.portfolio_performance }}" alt="Portfolio Performance">
            </div>
        </div>
        {% endif %}
        
        {% if plots.drawdown %}
        <div class="section">
            <h2>Risk Analysis</h2>
            <div class="two-column">
                <div class="plot-container">
                    <img src="{{ plots.drawdown }}" alt="Drawdown Analysis">
                </div>
                {% if plots.returns_distribution %}
                <div class="plot-container">
                    <img src="{{ plots.returns_distribution }}" alt="Returns Distribution">
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        {% if plots.trade_analytics %}
        <div class="section">
            <h2>Trade Analytics</h2>
            <div class="plot-container">
                <img src="{{ plots.trade_analytics }}" alt="Trade Analytics">
            </div>
            {% if plots.trade_heatmap %}
            <div class="plot-container">
                <img src="{{ plots.trade_heatmap }}" alt="Trade Timing Heatmap">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if plots.inventory %}
        <div class="section">
            <h2>Inventory Management</h2>
            <div class="plot-container">
                <img src="{{ plots.inventory }}" alt="Inventory Analysis">
            </div>
        </div>
        {% endif %}
        
        {% if plots.latency_distribution %}
        <div class="section">
            <h2>Latency Analysis</h2>
            <div class="two-column">
                <div class="plot-container">
                    <img src="{{ plots.latency_distribution }}" alt="Latency Distribution">
                </div>
                {% if plots.latency_trend %}
                <div class="plot-container">
                    <img src="{{ plots.latency_trend }}" alt="Latency Trend">
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        {% if plots.training_curves %}
        <div class="section">
            <h2>RL Training Analysis</h2>
            <div class="plot-container">
                <img src="{{ plots.training_curves }}" alt="Training Curves">
            </div>
            {% if plots.learning_progress %}
            <div class="plot-container">
                <img src="{{ plots.learning_progress }}" alt="Learning Progress">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="footer">
            <p>Generated by HFT Order-Book Simulator + Market-Making RL Agent</p>
            <p>🤖 Powered by Claude Code</p>
        </div>
    </div>
</body>
</html>
        """
        
        return Template(template_str)


def main():
    """Example usage of report generator."""
    # This would typically be called with real data
    print("Report generator ready. Use generate_report() method with your data.")


if __name__ == '__main__':
    main()