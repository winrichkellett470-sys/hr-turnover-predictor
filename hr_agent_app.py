"""
hr_agent_app.py —— 员工离职预测智能体（HR Talent Retention Advisor）

功能：
    1. 输入员工九项特征 → 预测离职概率
    2. 分析最影响该员工的关键因素
    3. 综合得分判断是否为"需要留住的优秀员工"
    4. 给出针对性的管理建议

启动方式：
    streamlit run hr_agent_app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

# 中文字体
# 加载中文字体（本地和云端都适配）
import matplotlib.font_manager as fm

font_candidates = [
    'NotoSansCJKsc-Regular.otf',   # 仓库内的字体（云端用）
    'NotoSansSC-Regular.ttf',      # 备用字体名
]
zh_font = None
for font_file in font_candidates:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)
        zh_font = fm.FontProperties(fname=font_file).get_name()
        break

if zh_font:
    # 云端加载成功
    matplotlib.rcParams['font.sans-serif'] = [zh_font, 'SimHei', 'Microsoft YaHei']
else:
    # 本地回退：用系统字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================================================
# 页面配置
# ==============================================================
st.set_page_config(
    page_title="员工离职预警智能助手",
    page_icon="🧑‍💼",
    layout="wide",
)


# ==============================================================
# 模型加载（带缓存，只加载一次；云端首次运行会自动训练）
# ==============================================================
@st.cache_resource
def load_artifacts():
    required = ['champion_model.pkl', 'scaler.pkl', 'feature_info.pkl']
    missing = [f for f in required if not os.path.exists(f)]

    # 本地没有模型文件 → 自动触发一次训练（部署到云端的必经路径）
    if missing:
        if not os.path.exists('HR.csv'):
            st.error(
                "❌ 找不到 HR.csv 数据文件，无法训练模型。\n\n"
                "请确保 HR.csv 与 hr_agent_app.py 在同一目录下。"
            )
            st.stop()

        with st.spinner("⚙️ 首次启动，正在训练 Stacking 模型... 约 1-3 分钟，请耐心等待"):
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, 'train_model.py'],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"训练失败：\n{result.stderr}")
                st.stop()
        st.success("✅ 模型训练完成！即将进入应用...")

    with open('champion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, scaler, info


model, scaler, info = load_artifacts()
FEATURE_NAMES = info['feature_names']            # 18 列独热编码后的列名
DEPARTMENTS = info['departments']                # 部门原始列表
SALARY_LEVELS = info['salary_levels']
REF = info['reference_stats']                    # 行业基准


# ==============================================================
# 工具函数
# ==============================================================
def build_input_row(raw: dict) -> pd.DataFrame:
    """把单个员工的原始输入转成模型可接受的 DataFrame (1 × 18)"""
    # 先准备基础字段
    row = {
        'satisfaction_level': raw['satisfaction'],
        'last_evaluation': raw['evaluation'],
        'number_project': raw['projects'],
        'average_montly_hours': raw['hours'],
        'time_spend_company': raw['tenure'],
        'Work_accident': int(raw['accident']),
        'promotion_last_5years': int(raw['promoted']),
    }
    # 独热编码：部门与薪资
    for dep in DEPARTMENTS:
        if dep != DEPARTMENTS[0]:   # drop_first=True 跳过第一个
            row[f'position_{dep}'] = int(raw['department'] == dep)
    for sal in SALARY_LEVELS:
        if sal != SALARY_LEVELS[0]:
            row[f'salary_{sal}'] = int(raw['salary'] == sal)

    # 严格按 FEATURE_NAMES 的列顺序对齐
    aligned = {col: row.get(col, 0) for col in FEATURE_NAMES}
    return pd.DataFrame([aligned], columns=FEATURE_NAMES)


def predict_leave_prob(row_df: pd.DataFrame) -> float:
    """返回离职概率 ∈ [0, 1]"""
    row_s = scaler.transform(row_df)
    prob = model.predict_proba(row_s)[0, 1]
    return float(prob)


def compute_talent_score(raw: dict) -> dict:
    """
    综合得分制：衡量员工"优秀度"（满分 100）
    加权组合：绩效评估 40% + 项目数 20% + 工时 20% + 在职年数 20%
    返回：{ 'score': float, 'level': str, 'breakdown': dict }
    """
    # ---- 各维度归一化到 0-1 ----
    # 绩效评估：原始 0-1，直接用
    eval_score = raw['evaluation']

    # 项目数：4-5 为理想区间，2 和 7 都扣分（U 型倒置）
    n_proj = raw['projects']
    if n_proj in (4, 5):
        proj_score = 1.0
    elif n_proj == 3:
        proj_score = 0.75
    elif n_proj == 6:
        proj_score = 0.70
    elif n_proj == 2:
        proj_score = 0.30
    elif n_proj == 7:
        proj_score = 0.40
    else:
        proj_score = 0.5

    # 月均工时：180-240 是合理区间，高于或低于扣分
    h = raw['hours']
    if 180 <= h <= 240:
        hours_score = 1.0
    elif 150 <= h < 180 or 240 < h <= 260:
        hours_score = 0.8
    elif 140 <= h < 150 or 260 < h <= 280:
        hours_score = 0.6
    else:
        hours_score = 0.35

    # 在职年数：3 年以上算成熟员工，但 10 年封顶
    tenure = raw['tenure']
    if tenure <= 2:
        tenure_score = 0.4
    elif tenure == 3:
        tenure_score = 0.7
    elif 4 <= tenure <= 6:
        tenure_score = 1.0
    elif tenure >= 7:
        tenure_score = 0.9
    else:
        tenure_score = 0.5

    # ---- 加权合成 ----
    weights = {'evaluation': 0.40, 'projects': 0.20, 'hours': 0.20, 'tenure': 0.20}
    score = (
        weights['evaluation'] * eval_score +
        weights['projects'] * proj_score +
        weights['hours'] * hours_score +
        weights['tenure'] * tenure_score
    ) * 100

    # ---- 等级判定 ----
    if score >= 80:
        level = "核心人才"
    elif score >= 65:
        level = "骨干员工"
    elif score >= 50:
        level = "普通员工"
    else:
        level = "待提升员工"

    return {
        'score': round(score, 1),
        'level': level,
        'breakdown': {
            '绩效评估': round(eval_score * 100, 1),
            '项目贡献': round(proj_score * 100, 1),
            '工作投入': round(hours_score * 100, 1),
            '资历深度': round(tenure_score * 100, 1),
        },
        'weights': weights,
    }


def analyze_risk_factors(raw: dict, prob: float) -> list:
    """
    根据员工特征识别出最关键的风险因素，返回有序列表
    每项格式：{'factor': str, 'severity': 'high/medium/low', 'description': str}
    """
    factors = []
    s = raw['satisfaction']
    h = raw['hours']
    p = raw['projects']
    t = raw['tenure']
    e = raw['evaluation']

    # ---- 工作满意度 ----
    if s < 0.2:
        factors.append({
            'factor': '工作满意度极低',
            'severity': 'high',
            'value': f'{s:.2f}（行业均值 {REF["satisfaction_mean"]:.2f}）',
            'desc': '员工已进入"心理离职"状态，很可能已在寻找下家。这是最紧急的预警信号。',
            'weight': 10 if s < 0.15 else 9,
        })
    elif 0.35 <= s <= 0.46:
        factors.append({
            'factor': '工作满意度偏低',
            'severity': 'medium',
            'value': f'{s:.2f}（行业均值 {REF["satisfaction_mean"]:.2f}）',
            'desc': '处于"中等偏低"离职集群，常伴随职业倦怠感。',
            'weight': 7,
        })

    # ---- 工作负荷 ----
    if h > 260:
        factors.append({
            'factor': '月均工时过高',
            'severity': 'high',
            'value': f'{h} 小时（健康值 180-240）',
            'desc': '长期超负荷工作，职业倦怠风险极高。此类员工常因身心耗竭而突然离职。',
            'weight': 8,
        })
    elif h < 150:
        factors.append({
            'factor': '月均工时偏低',
            'severity': 'medium',
            'value': f'{h} 小时（健康值 180-240）',
            'desc': '可能处于被边缘化状态，缺少参与感与成就感。',
            'weight': 6,
        })

    if p >= 6:
        factors.append({
            'factor': '项目数量过多',
            'severity': 'high',
            'value': f'{p} 个项目',
            'desc': '任务过载，分身乏术，难以在任何一个项目上做出深度贡献，心理压力大。',
            'weight': 8,
        })
    elif p == 2:
        factors.append({
            'factor': '项目参与度不足',
            'severity': 'medium',
            'value': f'仅 {p} 个项目',
            'desc': '任务量不足，可能被认为"不受重用"，主动离职意愿上升。',
            'weight': 6,
        })

    # ---- 在职年数：3-6 年高危期 ----
    if 4 <= t <= 6:
        factors.append({
            'factor': '处于离职高发工龄段',
            'severity': 'medium',
            'value': f'在职 {t} 年',
            'desc': '3-6 年工龄是员工离职的高发窗口期，常在此时寻求新的发展机会。',
            'weight': 5,
        })

    # ---- 晋升 ----
    if raw['promoted'] == 0 and t >= 5:
        factors.append({
            'factor': '长期未获晋升',
            'severity': 'medium',
            'value': f'在职 {t} 年仍未晋升',
            'desc': '在同岗位停留过久，发展受阻是离职的重要推力。',
            'weight': 5,
        })

    # ---- 高绩效 + 低薪（被挖角风险） ----
    if e >= 0.8 and raw['salary'] == 'low':
        factors.append({
            'factor': '高绩效低薪资',
            'severity': 'high',
            'value': f'绩效 {e:.2f}，薪资档位：低',
            'desc': '市场议价能力强但薪酬匹配度低，是竞争对手挖角的首选目标。',
            'weight': 9,
        })

    # ---- 如果没识别到任何风险，说明员工处于健康状态 ----
    if not factors:
        factors.append({
            'factor': '各项指标均在健康区间',
            'severity': 'low',
            'value': '—',
            'desc': '员工处于良好状态，无明显离职风险信号。',
            'weight': 0,
        })

    # 按权重降序
    factors.sort(key=lambda x: x['weight'], reverse=True)
    return factors


def risk_level(prob: float) -> tuple:
    """根据概率划分风险等级，返回 (等级, 颜色, emoji)"""
    if prob >= 0.5:
        return ("高风险", "#D7263D", "🔴")
    elif prob >= 0.2:
        return ("中风险", "#F6AE2D", "🟡")
    else:
        return ("低风险", "#2A9D8F", "🟢")


def generate_recommendation(prob: float, talent: dict, factors: list) -> str:
    """根据预测结果生成管理建议"""
    is_valuable = talent['level'] in ("核心人才", "骨干员工")
    high_risk = prob >= 0.5

    if high_risk and is_valuable:
        return (
            f"🚨 **紧急挽留建议**：该员工是企业的**{talent['level']}**"
            f"（综合得分 {talent['score']}），同时预测离职概率高达 **{prob:.1%}**。"
            f"属于「高价值 + 高风险」象限，必须立即介入。\n\n"
            f"**建议行动**：\n"
            f"- 本周内安排直属上级或 HRBP 进行一对一深度沟通\n"
            f"- 针对上述关键风险因素提供具体改善方案（如调整工作负荷、薪酬 review）\n"
            f"- 评估是否有晋升或岗位轮换机会\n"
            f"- 必要时升级到 HR VP 或事业部总经理层面介入"
        )
    elif high_risk and not is_valuable:
        return (
            f"⚠️ **常规关注建议**：该员工预测离职概率为 **{prob:.1%}**（属高风险），"
            f"综合得分 {talent['score']}，属于「一般员工 + 高风险」象限。\n\n"
            f"**建议行动**：\n"
            f"- 由直属上级进行常规关怀沟通，了解真实想法\n"
            f"- 评估是否有合适的岗位调整方向或技能培训机会\n"
            f"- 若员工去意已决，可从容做好交接安排，同步启动招聘流程"
        )
    elif not high_risk and is_valuable:
        return (
            f"✅ **维护关系建议**：该员工是企业的**{talent['level']}**"
            f"（综合得分 {talent['score']}），当前离职概率较低（**{prob:.1%}**），"
            f"处于「高价值 + 稳定」象限。\n\n"
            f"**建议行动**：\n"
            f"- 保持现有激励与关怀机制\n"
            f"- 纳入高潜人才库，定期提供发展机会（项目负责、内部导师等）\n"
            f"- 年度薪酬调整时给予倾斜，强化归属感"
        )
    else:
        return (
            f"👍 **稳定员工**：该员工综合得分 {talent['score']}（{talent['level']}），"
            f"离职概率较低（**{prob:.1%}**）。无需特别干预，按常规管理即可。"
        )


# ==============================================================
# 侧边栏 —— 输入区
# ==============================================================
st.sidebar.title("📋 员工信息录入")
st.sidebar.markdown("请填写该员工的各项指标，右侧将实时给出分析结果。")

# —— 一键载入典型画像 ——
st.sidebar.markdown("#### ⚡ 快速测试（可选）")
st.sidebar.caption("点击下方按钮，一键载入典型员工画像：")

PRESET_PROFILES = {
    '边缘化的冷淡者': {
        'satisfaction': 0.10, 'evaluation': 0.50,
        'projects': 2, 'hours': 140, 'tenure': 3,
        'department': 'sales', 'salary': 'low',
        'accident': False, 'promoted': False,
    },
    '过载的疲惫者': {
        'satisfaction': 0.40, 'evaluation': 0.85,
        'projects': 6, 'hours': 280, 'tenure': 4,
        'department': 'technical', 'salary': 'medium',
        'accident': False, 'promoted': False,
    },
    '被挖角的明星': {
        'satisfaction': 0.80, 'evaluation': 0.92,
        'projects': 5, 'hours': 240, 'tenure': 5,
        'department': 'technical', 'salary': 'low',
        'accident': False, 'promoted': False,
    },
    '健康的核心员工': {
        'satisfaction': 0.75, 'evaluation': 0.85,
        'projects': 4, 'hours': 210, 'tenure': 5,
        'department': 'technical', 'salary': 'high',
        'accident': False, 'promoted': True,
    },
}

preset_cols = st.sidebar.columns(2)
for i, (name, profile) in enumerate(PRESET_PROFILES.items()):
    col = preset_cols[i % 2]
    if col.button(name, use_container_width=True, key=f'preset_{i}'):
        for k, v in profile.items():
            st.session_state[f'input_{k}'] = v
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### 主观评价维度")
satisfaction = st.sidebar.slider(
    "工作满意度", 0.0, 1.0,
    st.session_state.get('input_satisfaction', 0.60), 0.01,
    help="来自员工自评问卷，0 = 非常不满意，1 = 非常满意"
)
evaluation = st.sidebar.slider(
    "最新绩效评估", 0.0, 1.0,
    st.session_state.get('input_evaluation', 0.75), 0.01,
    help="来自上级主管打分，0 = 很差，1 = 优秀"
)

st.sidebar.markdown("#### 工作负荷维度")
projects = st.sidebar.select_slider(
    "参与项目数", options=list(range(2, 8)),
    value=st.session_state.get('input_projects', 4),
    help="当前正在进行或近期参与的项目数"
)
hours = st.sidebar.slider(
    "月均工作时长（小时）", 90, 320,
    st.session_state.get('input_hours', 200), 1,
    help="过去三个月平均每月工作小时数"
)

st.sidebar.markdown("#### 组织归属维度")
tenure = st.sidebar.select_slider(
    "在职年数", options=list(range(2, 11)),
    value=st.session_state.get('input_tenure', 3),
)
_dep_default = st.session_state.get('input_department',
                                     'sales' if 'sales' in DEPARTMENTS else DEPARTMENTS[0])
department = st.sidebar.selectbox(
    "所属部门", DEPARTMENTS,
    index=DEPARTMENTS.index(_dep_default) if _dep_default in DEPARTMENTS else 0
)
salary = st.sidebar.selectbox(
    "薪资水平", SALARY_LEVELS,
    index=SALARY_LEVELS.index(st.session_state.get('input_salary', 'low'))
)

st.sidebar.markdown("#### 事件维度")
accident = st.sidebar.checkbox("过去发生过工作事故",
                                value=st.session_state.get('input_accident', False))
promoted = st.sidebar.checkbox("过去 5 年内获得过晋升",
                                value=st.session_state.get('input_promoted', False))

predict_btn = st.sidebar.button("🔍 分析该员工", type="primary", use_container_width=True)


# ==============================================================
# 主界面
# ==============================================================
st.title("🧑‍💼 员工离职预警智能助手")
st.caption(
    "基于 Stacking 集成模型（AUC = 0.9926）· "
    "集成 Random Forest + XGBoost + GBDT，以 Logistic Regression 为元学习器"
)

# 尚未点击时显示使用说明
if not predict_btn:
    st.info("👈 请在左侧填写员工信息，点击 **分析该员工** 按钮开始。")

    st.markdown("### 这个智能体能为 HR 做什么？")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        "#### 🎯 精准预警\n"
        "基于 14,999 份员工样本训练，离职预测查全率 96%、精确率 98%。"
    )
    col2.markdown(
        "#### 🔍 归因分析\n"
        "自动识别影响该员工的关键风险因素，不只是「高风险」三个字，而是告诉你「为什么」。"
    )
    col3.markdown(
        "#### 💎 人才价值判断\n"
        "综合绩效、项目、工时、资历四维度评分，区分「高价值 vs. 一般员工」，帮助 HR 聚焦重点。"
    )

    st.markdown("### 关于综合得分")
    st.markdown("""
    综合得分由以下四个维度加权合成，满分 100：
    
    | 维度 | 权重 | 含义 |
    | --- | --- | --- |
    | 绩效评估 | 40% | 上级主管打分 |
    | 项目贡献 | 20% | 项目数量是否处于理想区间（4-5 个为最佳） |
    | 工作投入 | 20% | 月均工时是否健康（180-240 小时为最佳） |
    | 资历深度 | 20% | 是否已成为成熟员工（4-6 年资深价值最高） |
    
    **等级划分**：≥ 80 核心人才 · 65-80 骨干员工 · 50-65 普通员工 · < 50 待提升员工
    """)
    st.stop()


# ========== 点击按钮后的完整分析 ==========
raw = dict(
    satisfaction=satisfaction, evaluation=evaluation,
    projects=projects, hours=hours, tenure=tenure,
    department=department, salary=salary,
    accident=accident, promoted=promoted,
)

row_df = build_input_row(raw)
prob = predict_leave_prob(row_df)
talent = compute_talent_score(raw)
factors = analyze_risk_factors(raw, prob)
rlevel, rcolor, remoji = risk_level(prob)
recommendation = generate_recommendation(prob, talent, factors)


# ---------- 顶部指标卡 ----------
col1, col2, col3 = st.columns([1.3, 1.3, 1.4])

with col1:
    st.markdown("##### 离职概率预测")
    st.markdown(
        f"<h1 style='color:{rcolor}; margin:0'>{prob:.1%}</h1>"
        f"<p style='font-size:20px; color:{rcolor}; margin:0'>{remoji} {rlevel}</p>",
        unsafe_allow_html=True
    )
    st.progress(prob)

with col2:
    st.markdown("##### 人才综合得分")
    score = talent['score']
    score_color = "#2A9D8F" if score >= 65 else ("#F6AE2D" if score >= 50 else "#D7263D")
    st.markdown(
        f"<h1 style='color:{score_color}; margin:0'>{score:.1f}</h1>"
        f"<p style='font-size:20px; color:{score_color}; margin:0'>💎 {talent['level']}</p>",
        unsafe_allow_html=True
    )
    st.progress(score / 100)

with col3:
    st.markdown("##### 人才-风险象限定位")
    q_high_risk = prob >= 0.5
    q_valuable = talent['level'] in ('核心人才', '骨干员工')
    if q_high_risk and q_valuable:
        q = "🚨 高价值 + 高风险（紧急挽留）"; qcolor = "#D7263D"
    elif q_high_risk and not q_valuable:
        q = "⚠️ 一般员工 + 高风险（常规关注）"; qcolor = "#F6AE2D"
    elif not q_high_risk and q_valuable:
        q = "✅ 高价值 + 稳定（重点维护）"; qcolor = "#2A9D8F"
    else:
        q = "👍 一般员工 + 稳定（常规管理）"; qcolor = "#577590"
    st.markdown(
        f"<p style='font-size:18px; color:{qcolor}; font-weight:bold; "
        f"padding-top:28px'>{q}</p>",
        unsafe_allow_html=True
    )


st.markdown("---")

# ---------- 关键风险因素 ----------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown("### 🔍 影响该员工的关键因素")
    if factors[0]['severity'] == 'low':
        st.success(factors[0]['desc'])
    else:
        for i, f in enumerate(factors[:5], 1):
            sev_color = {'high': '#D7263D', 'medium': '#F6AE2D', 'low': '#2A9D8F'}[f['severity']]
            sev_label = {'high': '高', 'medium': '中', 'low': '低'}[f['severity']]
            st.markdown(
                f"<div style='padding:10px; border-left:4px solid {sev_color}; "
                f"background-color:#F8F9FA; margin-bottom:8px;'>"
                f"<b>#{i} {f['factor']}</b> "
                f"<span style='color:{sev_color}; font-weight:bold;'>[风险:{sev_label}]</span><br>"
                f"<small>📊 {f['value']}</small><br>"
                f"<small style='color:#555'>💡 {f['desc']}</small>"
                f"</div>",
                unsafe_allow_html=True
            )

with col_right:
    st.markdown("### 💎 人才综合得分构成")
    fig, ax = plt.subplots(figsize=(6, 4))
    bd = talent['breakdown']
    w = talent['weights']
    dims = list(bd.keys())
    scores = list(bd.values())
    weights_pct = [w[k] * 100 for k in ['evaluation', 'projects', 'hours', 'tenure']]

    colors = ['#2A9D8F' if s >= 70 else ('#F6AE2D' if s >= 50 else '#D7263D') for s in scores]
    bars = ax.barh(dims, scores, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel('得分')
    ax.set_title(f'综合得分: {talent["score"]:.1f}  ({talent["level"]})', fontsize=12)

    for bar, s, wp in zip(bars, scores, weights_pct):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{s:.0f}  (权重 {wp:.0f}%)', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


st.markdown("---")

# ---------- 管理建议 ----------
st.markdown("### 📋 管理建议")
st.markdown(recommendation)


# ---------- 折叠：输入回顾 + 原始概率 ----------
with st.expander("🔬 查看原始输入与模型细节"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**原始输入：**")
        st.json({
            '工作满意度': satisfaction,
            '最新绩效评估': evaluation,
            '参与项目数': projects,
            '月均工作时长': hours,
            '在职年数': tenure,
            '所属部门': department,
            '薪资水平': salary,
            '工作事故': accident,
            '过去5年是否晋升': promoted,
        })
    with col_b:
        st.markdown("**模型输出：**")
        st.json({
            '离职概率': f'{prob:.4f}',
            '风险等级': rlevel,
            '综合得分': talent['score'],
            '人才等级': talent['level'],
            '识别到的风险因素数': len([f for f in factors if f['severity'] != 'low']),
        })

st.markdown(
    "<br><small style='color:#888'>"
    "※ 本系统基于 HR 历史数据训练，预测结果仅供参考，最终决策请结合实际管理经验综合判断。"
    "</small>",
    unsafe_allow_html=True
)
