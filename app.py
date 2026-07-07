import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import datetime

st.set_page_config(
    page_title="Shipping Route Optimizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container{padding-top:0rem;padding-bottom:0rem}
    header{visibility:hidden}
    .stMetric{background:#f8f9fa;border-radius:10px;padding:12px;border:1px solid #e9ecef}
    .stMetric label{color:#6c757d;font-size:13px}
    .topbar{background:#ffffff;border-bottom:1px solid #e9ecef;padding:14px 32px}
    .topbar-title{font-size:17px;font-weight:600;color:#1a1a2e;letter-spacing:-0.01em}
    .topbar-sub{font-size:12px;color:#6c757d;margin-top:2px}
    .ai-box{background:#f8f9fa;border-left:3px solid #1a1a2e;border-radius:0 8px 8px 0;padding:14px 16px;font-size:14px;color:#495057;line-height:1.7;margin-top:12px}
    .risk-high{color:#dc3545;font-weight:600}
    .risk-medium{color:#fd7e14;font-weight:600}
    .risk-low{color:#28a745;font-weight:600}
    .stButton button{background:#1a1a2e;color:#ffffff;border:none;border-radius:8px;padding:10px 24px;font-size:14px;font-weight:500;width:100%}
    .stButton button:hover{background:#2d2d44;color:#ffffff}
    footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

ORS_KEY  = st.secrets["ORS_KEY"]
GROQ_KEY = st.secrets["GROQ_KEY"]

@st.cache_data
def load_data():
    with open("data/cities.json") as f:
        cities = json.load(f)
    route_graph = {}
    for i in range(1, 5):
        with open(f"data/route_graph_{i}.json") as f:
            chunk = json.load(f)
            route_graph.update(chunk)
            del chunk
    route_stats   = pd.read_parquet("data/route_stats.parquet")
    monthly_stats = pd.read_parquet("data/monthly_stats.parquet")
    city_stats    = pd.read_parquet("data/city_stats.parquet")
    return cities, route_graph, route_stats, monthly_stats, city_stats

@st.cache_resource
def load_model():
    model    = joblib.load("data/rf_model.pkl")
    encoders = joblib.load("data/encoders.pkl")
    return model, encoders

cities, route_graph, route_stats, monthly_stats, city_stats = load_data()
rf_model, encoders = load_model()
le_truck   = encoders["truck"]
le_weather = encoders["weather"]
le_cargo   = encoders["cargo"]
features   = encoders["features"]
city_list  = list(cities.keys())

llm = ChatGroq(
    api_key=GROQ_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0)

cargo_truck_rules = {
    "Fresh Produce": ["Refrigerated"],
    "Machinery":     ["Heavy", "Flatbed"],
    "Chemicals":     ["Heavy"],
    "Electronics":   ["Light", "Medium"],
    "Dry Goods":     ["Light", "Medium", "Heavy", "Flatbed"],
    "Auto Parts":    ["Medium", "Heavy", "Flatbed"],
}
speed_factors     = {"Clear":1.0,"Rain":0.85,"Fog":0.80,"Snow":0.65,"Storm":0.55,"Windy":0.90}
truck_multipliers = {"Light":0.95,"Medium":1.0,"Heavy":1.25,"Refrigerated":1.15,"Flatbed":1.20}
fuel_rates        = {"Light":0.25,"Medium":0.35,"Heavy":0.50,"Refrigerated":0.45,"Flatbed":0.42}
month_names       = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

@st.cache_resource
def build_graph(_cities, _route_graph):
    G = nx.DiGraph()
    for city, coords in _cities.items():
        G.add_node(city, lat=coords["lat"], lon=coords["lon"])
    for key, val in _route_graph.items():
        G.add_edge(val["origin"], val["dest"],
                   distance_km=val["best_route"]["distance_km"],
                   duration_hrs=val["best_route"]["duration_hrs"])
    return G

G = build_graph(cities, route_graph)

def find_best_truck_route(origin, dest, truck_type, weight_kg, weather):
    G_w = G.copy()
    tf  = truck_multipliers[truck_type]
    wf  = 1 / speed_factors[weather]
    wt  = 1 + (weight_kg / 20000) * 0.15
    for u, v, d in G_w.edges(data=True):
        G_w[u][v]["adj"] = round(d["duration_hrs"] * tf * wf * wt, 2)
    try:
        path = nx.dijkstra_path(G_w, origin, dest, weight="adj")
        dur  = nx.dijkstra_path_length(G_w, origin, dest, weight="adj")
        dist = sum(G[path[i]][path[i+1]]["distance_km"] for i in range(len(path)-1))
        return {"path":path, "total_hrs":round(dur,2), "total_km":round(dist,2), "hops":len(path)-1}
    except:
        return None

def predict_risk(truck, weather, cargo, weight, month, quarter, dist):
    X = pd.DataFrame([[
        le_truck.transform([truck])[0],
        le_weather.transform([weather])[0],
        le_cargo.transform([cargo])[0],
        weight, month, quarter, dist
    ]], columns=features)
    return round(rf_model.predict_proba(X)[0][1] * 100, 1)

def get_explanation(route, truck, cargo, weight, weather, risk):
    try:
        system_msg = SystemMessage(content=(
            "You are an expert logistics advisor. "
            "Give a concise 3-sentence route explanation covering: "
            "why this route was chosen, key risk factors, "
            "and practical driver advice."))
        human_msg  = HumanMessage(content=(
            f"Origin:{route['path'][0]} Destination:{route['path'][-1]} "
            f"Path:{' - '.join(route['path'])} Distance:{route['total_km']}km "
            f"Duration:{route['total_hrs']}hrs Truck:{truck} Cargo:{cargo} "
            f"Weight:{weight}kg Weather:{weather} DelayRisk:{risk}%"))
        r = llm.invoke([system_msg, human_msg])
        return r.content
    except Exception as e:
        return f"Route: {' → '.join(route['path'])} | {route['total_km']}km | {route['total_hrs']}hrs | Risk: {risk}%"

def parse_nl_query(query):
    city_list_str = ", ".join(city_list)
    try:
        system_content = (
            "You are a JSON extractor. Return ONLY a flat JSON object with "
            "EXACTLY these keys: origin, destination, truck_type, cargo_type, "
            "weight_kg, weather. "
            f"origin must be one of: {city_list_str}. "
            f"destination must be one of: {city_list_str}. "
            "truck_type: one of Light, Medium, Heavy, Refrigerated, Flatbed. "
            "cargo_type: one of Electronics, Fresh Produce, Machinery, Dry Goods, Chemicals, Auto Parts. "
            "weight_kg: number between 500 and 20000. "
            "weather: one of Clear, Rain, Fog, Snow, Storm, Windy. "
            "Infer truck_type from cargo if not stated. "
            "Use Clear if weather not mentioned. "
            "Use 5000 if weight not mentioned. "
            "Return ONLY raw JSON, no markdown, no explanation."
        )
        r    = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=query)])
        text = r.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(text)
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None

def draw_route_map(route):
    mid_lat = np.mean([cities[c]["lat"] for c in route["path"]])
    mid_lon = np.mean([cities[c]["lon"] for c in route["path"]])
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=5,
                   tiles="CartoDB positron")
    for i in range(len(route["path"])-1):
        o, d = route["path"][i], route["path"][i+1]
        key  = f"{o}->{d}"
        if key in route_graph:
            coords = route_graph[key]["best_route"]["geometry"]
            folium.PolyLine([[c[1],c[0]] for c in coords],
                            color="#1a1a2e", weight=4, opacity=0.8).add_to(m)
    for i, city in enumerate(route["path"]):
        color = "darkblue" if i==0 else "red" if i==len(route["path"])-1 else "orange"
        folium.Marker(
            [cities[city]["lat"], cities[city]["lon"]],
            popup=city,
            icon=folium.Icon(color=color)
        ).add_to(m)
    return m

def show_route_results(route, truck, cargo, weight, weather):
    now     = datetime.datetime.now()
    month   = now.month
    quarter = (month-1)//3+1
    risk    = predict_risk(truck, weather, cargo, weight, month, quarter, route["total_km"])
    fuel    = round(route["total_km"] * fuel_rates[truck] * 1.5, 2)
    level   = "HIGH" if risk>60 else "MEDIUM" if risk>30 else "LOW"
    rc      = "risk-high" if risk>60 else "risk-medium" if risk>30 else "risk-low"

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Distance",   f"{route['total_km']} km")
    c2.metric("Duration",   f"{route['total_hrs']} hrs")
    c3.metric("Delay Risk", f"{risk}%")
    c4.metric("Fuel Cost",  f"${fuel}")
    st.markdown(
        f"**Route:** {' → '.join(route['path'])} | "
        f"{route['hops']} hop(s) | "
        f"<span class='{rc}'>{level} risk</span>",
        unsafe_allow_html=True)

    col_map, col_ai = st.columns([1.6, 1])
    with col_map:
        st_folium(draw_route_map(route), height=350, width=700)
    with col_ai:
        if "explanation" not in st.session_state:
            st.session_state.explanation = None
        if st.session_state.explanation is None:
            with st.spinner("Generating AI explanation..."):
                st.session_state.explanation = get_explanation(
                    route, truck, cargo, weight, weather, risk)
        st.markdown(
            f'<div class="ai-box">{st.session_state.explanation}</div>',
            unsafe_allow_html=True)
    return risk

# HEADER
st.markdown("""
<div class="topbar">
  <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
      <div class="topbar-title">Shipping Route Optimizer</div>
      <div class="topbar-sub">AI-powered truck routing · 20 US freight hubs  </div>
    </div>
    <div style="font-size:12px;color:#6c757d;text-align:right">
      Dijkstra · OpenRouteService HGV · Groq LLaMA 3.3<br>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

page = st.radio("Navigation",
                ["Route Planner","Route Analytics","Risk Intelligence","Fleet Analytics"],
                horizontal=True, label_visibility="collapsed")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# SESSION STATE INIT
if "route_result"  not in st.session_state: st.session_state.route_result  = None
if "route_truck"   not in st.session_state: st.session_state.route_truck   = None
if "route_cargo"   not in st.session_state: st.session_state.route_cargo   = None
if "route_weight"  not in st.session_state: st.session_state.route_weight  = None
if "route_weather" not in st.session_state: st.session_state.route_weather = None
if "explanation"   not in st.session_state: st.session_state.explanation   = None

# PAGE 1 - ROUTE PLANNER
if page == "Route Planner":
    nl_query = st.text_input("Search",
        placeholder="Describe your shipment — e.g. Ship 12000kg of machinery from Chicago to Miami, stormy weather",
        label_visibility="collapsed")

    if nl_query:
        has_city = any(c.lower() in nl_query.lower() for c in city_list)
        is_garbage = (len(nl_query.strip()) < 15 or
                      nl_query.startswith("ghp_") or
                      nl_query.startswith("sk-") or
                      nl_query.startswith("eyJ") or
                      not has_city)
        if is_garbage:
            st.error("Please describe a shipment with origin and destination cities. Example: Ship machinery from Chicago to Miami.")
        else:
          with st.spinner("Parsing with AI..."):
            parsed = parse_nl_query(nl_query)
          if parsed:
            if (parsed.get("origin") not in city_list or
                    parsed.get("destination") not in city_list or
                    parsed.get("origin") == parsed.get("destination")):
                st.error("Could not identify valid cities. Please describe a shipment between two US freight cities.")
            else:
                st.success(f"Parsed: {parsed['origin']} → {parsed['destination']} | {parsed['truck_type']} | {parsed['cargo_type']} | {parsed['weight_kg']}kg | {parsed['weather']}")
                route = find_best_truck_route(
                    parsed["origin"], parsed["destination"],
                    parsed["truck_type"], parsed["weight_kg"],
                    parsed["weather"])
            if route:
                if (st.session_state.route_result != route or
                        st.session_state.route_truck != parsed["truck_type"]):
                    st.session_state.route_result  = route
                    st.session_state.route_truck   = parsed["truck_type"]
                    st.session_state.route_cargo   = parsed["cargo_type"]
                    st.session_state.route_weight  = parsed["weight_kg"]
                    st.session_state.route_weather = parsed["weather"]
                    st.session_state.explanation   = None
                show_route_results(
                    st.session_state.route_result,
                    st.session_state.route_truck,
                    st.session_state.route_cargo,
                    st.session_state.route_weight,
                    st.session_state.route_weather)

    st.markdown("---")
    st.markdown("**Or fill in manually:**")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1: origin  = st.selectbox("Origin",      city_list, index=0)
    with col2: dest    = st.selectbox("Destination", [c for c in city_list if c != origin])
    with col3: cargo   = st.selectbox("Cargo",       list(cargo_truck_rules.keys()))
    with col4: truck   = st.selectbox("Truck",       cargo_truck_rules[cargo])
    with col5: weather = st.selectbox("Weather",     ["Clear","Rain","Fog","Snow","Storm","Windy"])
    with col6: weight  = st.number_input("Weight (kg)", 500, 20000, 5000, 500)

    if st.button("Find best route"):
        route = find_best_truck_route(origin, dest, truck, weight, weather)
        if route:
            st.session_state.route_result  = route
            st.session_state.route_truck   = truck
            st.session_state.route_cargo   = cargo
            st.session_state.route_weight  = weight
            st.session_state.route_weather = weather
            st.session_state.explanation   = None
        else:
            st.error("No route found.")

    if st.session_state.route_result and not nl_query:
        show_route_results(
            st.session_state.route_result,
            st.session_state.route_truck,
            st.session_state.route_cargo,
            st.session_state.route_weight,
            st.session_state.route_weather)

# PAGE 2 - ROUTE ANALYTICS
elif page == "Route Analytics":
    st.subheader("Route Analytics — 2025")
    st.markdown("Historical shipping patterns-synthetic trips.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total trips",   "500,000")
    c2.metric("City pairs",    "380")
    c3.metric("Busiest route",
              route_stats.nlargest(1,"trip_count").iloc[0]["origin_city"] + " → " +
              route_stats.nlargest(1,"trip_count").iloc[0]["dest_city"])
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 10 busiest routes**")
        top = route_stats.nlargest(10,"trip_count").copy()
        top["route"] = top["origin_city"] + " → " + top["dest_city"]
        fig = px.bar(top, x="trip_count", y="route", orientation="h",
                     color="trip_count",
                     color_continuous_scale=[[0,"#e9ecef"],[1,"#1a1a2e"]],
                     labels={"trip_count":"Trips","route":""})
        fig.update_layout(showlegend=False, height=350,
                          margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="white", paper_bgcolor="white",
                          coloraxis_showscale=False)
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("**Monthly trip volume**")
        monthly = monthly_stats.groupby("month")["trip_count"].sum().reset_index()
        monthly["month_name"] = monthly["month"].map(month_names)
        fig2 = px.area(monthly, x="month_name", y="trip_count",
                       color_discrete_sequence=["#1a1a2e"],
                       labels={"trip_count":"Trips","month_name":""})
        fig2.update_traces(fill="tozeroy", fillcolor="rgba(26,26,46,0.08)")
        fig2.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="white", paper_bgcolor="white")
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Monthly volume by truck type**")
    fig3 = px.bar(monthly_stats, x="month", y="trip_count", color="truck_type",
                  barmode="stack",
                  labels={"trip_count":"Trips","month":"Month","truck_type":"Truck"},
                  color_discrete_sequence=["#1a1a2e","#495057","#868e96","#ced4da","#f8f9fa"])
    fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                       plot_bgcolor="white", paper_bgcolor="white",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig3, use_container_width=True)

# PAGE 3 - RISK INTELLIGENCE
elif page == "Risk Intelligence":
    st.subheader("Risk Intelligence")
    st.markdown("Predict delay risk .")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1: ri_origin  = st.selectbox("Origin",      city_list, index=0)
    with col2: ri_dest    = st.selectbox("Destination", city_list, index=1)
    with col3: ri_truck   = st.selectbox("Truck",       ["Light","Medium","Heavy","Refrigerated","Flatbed"])
    with col4: ri_cargo   = st.selectbox("Cargo",       list(cargo_truck_rules.keys()))
    with col5: ri_weather = st.selectbox("Weather",     ["Clear","Rain","Fog","Snow","Storm","Windy"])
    with col6: ri_weight  = st.number_input("Weight (kg)", 500, 20000, 8000, 500)

    if st.button("Predict delay risk"):
        key  = f"{ri_origin}->{ri_dest}"
        dist = route_graph[key]["best_route"]["distance_km"] if key in route_graph else 1000
        now  = datetime.datetime.now()
        risk = predict_risk(ri_truck, ri_weather, ri_cargo, ri_weight,
                            now.month, (now.month-1)//3+1, dist)
        level = "HIGH" if risk>60 else "MEDIUM" if risk>30 else "LOW"
        col_g, col_w = st.columns([1, 1.5])
        with col_g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk,
                number={"suffix":"%","font":{"size":36,"color":"#1a1a2e"}},
                domain={"x":[0,1],"y":[0,1]},
                gauge={"axis":{"range":[0,100],"tickcolor":"#dee2e6"},
                       "bar":{"color":"#1a1a2e"},
                       "bgcolor":"#f8f9fa",
                       "borderwidth":0,
                       "steps":[{"range":[0,30],"color":"#d4edda"},
                                 {"range":[30,60],"color":"#fff3cd"},
                                 {"range":[60,100],"color":"#f8d7da"}],
                       "threshold":{"line":{"color":"#1a1a2e","width":3},
                                    "thickness":0.75,"value":risk}}))
            fig.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20),
                              paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            color = "#dc3545" if risk>60 else "#fd7e14" if risk>30 else "#28a745"
            st.markdown(
                f"**{ri_origin} → {ri_dest}:** "
                f"<span style='color:{color};font-weight:600'>{level} ({risk}%)</span>",
                unsafe_allow_html=True)
        with col_w:
            st.markdown("**Risk across all weather conditions**")
            weather_list = ["Clear","Rain","Fog","Snow","Storm","Windy"]
            risks = [{"Weather":w, "Risk": predict_risk(ri_truck, w, ri_cargo, ri_weight,
                      now.month, (now.month-1)//3+1, dist)} for w in weather_list]
            risk_df = pd.DataFrame(risks)
            colors  = ["#d4edda" if r<30 else "#fff3cd" if r<60 else "#f8d7da" for r in risk_df["Risk"]]
            fig5 = go.Figure(go.Bar(x=risk_df["Weather"], y=risk_df["Risk"],
                                    marker_color=colors,
                                    marker_line_color="#dee2e6",
                                    marker_line_width=1))
            fig5.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                               plot_bgcolor="white", paper_bgcolor="white",
                               yaxis=dict(range=[0,100],gridcolor="#f0f0f0"),
                               xaxis=dict(showgrid=False))
            st.plotly_chart(fig5, use_container_width=True)

# PAGE 4 - FLEET ANALYTICS
elif page == "Fleet Analytics":
    st.subheader("Fleet Analytics — 2025")
    st.markdown("Truck utilization and cargo distribution.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Truck type distribution**")
        truck_dist = monthly_stats.groupby("truck_type")["trip_count"].sum().reset_index()
        fig6 = go.Figure(go.Pie(
            labels=truck_dist["truck_type"],
            values=truck_dist["trip_count"],
            marker_colors=["#1a1a2e","#495057","#868e96","#ced4da","#adb5bd"],
            hole=0.45, textinfo="label+percent", textfont_size=12))
        fig6.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    with col2:
        st.markdown("**Average weight by truck type**")
        weight_dist = monthly_stats.groupby("truck_type")["avg_weight"].mean().reset_index()
        fig7 = go.Figure(go.Bar(
            x=weight_dist["truck_type"], y=weight_dist["avg_weight"],
            marker_color="#1a1a2e",
            marker_line_color="#dee2e6", marker_line_width=1))
        fig7.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="white", paper_bgcolor="white",
                           yaxis=dict(gridcolor="#f0f0f0"),
                           xaxis=dict(showgrid=False))
        st.plotly_chart(fig7, use_container_width=True)
    st.markdown("**Top 10 cities by trip volume**")
    top_cities = city_stats.nlargest(10,"total_trips")
    fig8 = go.Figure(go.Bar(
        x=top_cities["origin_city"], y=top_cities["total_trips"],
        marker_color=["#1a1a2e" if i<3 else "#868e96" for i in range(len(top_cities))],
        marker_line_color="#dee2e6", marker_line_width=1))
    fig8.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(gridcolor="#f0f0f0"),
                       xaxis=dict(showgrid=False))
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown("**Monthly truck utilization trends**")
    fig9 = px.line(monthly_stats, x="month", y="trip_count", color="truck_type",
                   markers=True,
                   color_discrete_sequence=["#1a1a2e","#495057","#868e96","#ced4da","#adb5bd"],
                   labels={"trip_count":"Trips","month":"Month","truck_type":"Truck"})
    fig9.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                       plot_bgcolor="white", paper_bgcolor="white",
                       legend=dict(orientation="h", y=-0.25),
                       yaxis=dict(gridcolor="#f0f0f0"),
                       xaxis=dict(showgrid=False))
    st.plotly_chart(fig9, use_container_width=True)