"""
revive-my-lover Dashboard — 可视化 AI 互动决策过程

展示：
1. 🎲 渴望曲线 — Poisson 概率随时间变化
2. 🧠 状态分布 — Bayesian 推断的用户状态
3. 📊 发送历史 — 什么时候发了/没发
4. 📈 信息增益 — 每次决策的 gain 值

运行：
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# Import revive-my-lover
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from revive_my_lover import PoissonLove
from revive_my_lover.bayesian import StateEstimator, State, BayesianLearner
from revive_my_lover.core.models import Action


# Page config
st.set_page_config(
    page_title="💘 revive-my-lover Dashboard",
    page_icon="💘",
    layout="wide",
)

st.title("💘 revive-my-lover Dashboard")
st.markdown("**Math models that make AI engagement feel human**")


# Sidebar controls
st.sidebar.header("⚙️ 参数设置")

# Poisson parameters
st.sidebar.subheader("🎲 Poisson 参数")
lam = st.sidebar.slider("λ (触达率)", 0.05, 0.50, 0.15, 0.01)
check_interval = st.sidebar.slider("检查间隔 (分钟)", 10, 60, 30, 5)
growth_factor = st.sidebar.slider("增长因子", 0.01, 0.20, 0.08, 0.01)

# Simulation parameters
st.sidebar.subheader("⏱️ 模拟设置")
sim_hours = st.sidebar.slider("模拟时长 (小时)", 12, 168, 48, 12)
seed = st.sidebar.number_input("随机种子", 0, 1000, 42)

# Run simulation
@st.cache_data
def run_simulation(lam, check_interval, growth_factor, sim_hours, seed):
    """Run a full simulation and return results."""
    love = PoissonLove(seed=seed)
    
    # Override config
    love._engine.config.engagement.lambda_rate = lam
    love._engine.config.engagement.check_interval_minutes = check_interval
    love._engine.config.engagement.growth_factor = growth_factor
    
    results = []
    now = datetime.now()
    
    total_ticks = int(sim_hours * 60 / check_interval)
    for i in range(total_ticks):
        tick_time = now + timedelta(minutes=i * check_interval)
        result = love.tick(now=tick_time)
        
        # Simulate user behavior
        hour = tick_time.hour
        if 9 <= hour < 17:  # Work hours
            love.record_reply(reply_speed=0.2, reply_length=0.2) if random.random() < 0.3 else None
        elif 18 <= hour < 22:  # Evening
            love.record_reply(reply_speed=0.8, reply_length=0.7) if random.random() < 0.6 else None
        elif 0 <= hour < 7:  # Night
            pass  # No replies
        
        if result.should_send:
            love.record_send()
        
        results.append({
            'time': tick_time,
            'hour': hour,
            'probability': result.probability,
            'should_send': result.should_send,
            'stage': result.stage,
            'user_state': result.user_state,
            'utility': result.send_utility,
            'info_gain': result.info_gain,
        })
    
    return pd.DataFrame(results)


# Run simulation
with st.spinner("正在模拟..."):
    df = run_simulation(lam, check_interval, growth_factor, sim_hours, seed)

# Layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_checks = len(df)
    st.metric("总检查次数", total_checks)

with col2:
    sent_count = df['should_send'].sum()
    st.metric("发送次数", sent_count)

with col3:
    max_prob = df['probability'].max()
    st.metric("最高渴望度", f"{max_prob:.0%}")

with col4:
    avg_utility = df['utility'].mean()
    st.metric("平均效用", f"{avg_utility:.2f}")


# Charts
st.markdown("---")

# 1. Longing Curve
st.subheader("🎲 渴望曲线 (Poisson Probability)")

fig1 = go.Figure()

# Add probability line
fig1.add_trace(go.Scatter(
    x=df['time'],
    y=df['probability'],
    mode='lines',
    name='渴望度',
    line=dict(color='#FF6B6B', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 107, 107, 0.1)',
))

# Add send markers
sent_df = df[df['should_send']]
fig1.add_trace(go.Scatter(
    x=sent_df['time'],
    y=sent_df['probability'],
    mode='markers',
    name='发送',
    marker=dict(color='#4ECDC4', size=10, symbol='star'),
))

fig1.update_layout(
    xaxis_title="时间",
    yaxis_title="渴望度",
    yaxis_range=[0, 1],
    height=300,
    showlegend=True,
)

st.plotly_chart(fig1, use_container_width=True)


# 2. State Distribution
st.subheader("🧠 用户状态分布 (Bayesian)")

# Count states
state_counts = df['user_state'].value_counts()

fig2 = go.Figure(data=[go.Pie(
    labels=state_counts.index,
    values=state_counts.values,
    hole=.3,
    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
)])

fig2.update_layout(
    height=300,
    showlegend=True,
)

st.plotly_chart(fig2, use_container_width=True)


# 3. Hourly Pattern
st.subheader("⏰ 按小时分布")

hourly = df.groupby('hour').agg({
    'should_send': 'sum',
    'probability': 'mean',
    'utility': 'mean',
}).reset_index()

fig3 = make_subplots(specs=[[{"secondary_y": True}]])

fig3.add_trace(
    go.Bar(x=hourly['hour'], y=hourly['should_send'], name="发送次数", marker_color='#4ECDC4'),
    secondary_y=False,
)

fig3.add_trace(
    go.Scatter(x=hourly['hour'], y=hourly['probability'], name="平均渴望度", line=dict(color='#FF6B6B', width=2)),
    secondary_y=True,
)

fig3.update_layout(
    xaxis_title="小时",
    height=300,
)

fig3.update_yaxes(title_text="发送次数", secondary_y=False)
fig3.update_yaxes(title_text="渴望度", range=[0, 1], secondary_y=True)

st.plotly_chart(fig3, use_container_width=True)


# 4. Decision Log
st.subheader("📋 决策日志")

# Show recent decisions
recent = df.tail(20).copy()
recent['时间'] = recent['time'].dt.strftime('%H:%M')
recent['决策'] = recent['should_send'].apply(lambda x: '✅ 发送' if x else '❌ 等待')
recent['状态'] = recent['user_state']
recent['渴望度'] = recent['probability'].apply(lambda x: f"{x:.0%}")
recent['效用'] = recent['utility'].apply(lambda x: f"{x:.2f}")

st.dataframe(
    recent[['时间', '决策', '状态', '渴望度', '效用']],
    use_container_width=True,
    hide_index=True,
)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        💘 revive-my-lover v0.9.0 |
        <a href="https://github.com/pearthink123/revive-my-lover">GitHub</a> |
        Math models that make AI engagement feel human
    </div>
    """,
    unsafe_allow_html=True,
)
