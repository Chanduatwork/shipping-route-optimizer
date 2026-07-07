# Shipping Route Optimizer — A Truck Routing Engine

A full-stack logistics routing application that finds the optimal truck route between US freight cities using real road data, graph algorithms, machine learning, and natural language AI.

## Live App
[Shipping Route Optimizer on Streamlit](https://shipping-route-optimizer.streamlit.app/)

## Project Overview

Built a graph-based truck routing engine using NetworkX Dijkstra over a MongoDB route graph of 380 city-pair edges enriched with OpenRouteService HGV API road data and historical delay rates, deployed as a 4-dashboard Streamlit app supporting multi-hop route optimization by truck type, cargo weight, and weather risk.

## Dashboards

| Dashboard | Description |
|---|---|
| Route Planner | Natural language or manual input, Dijkstra routing, XGBoost risk score, AI explanation, Folium map |
| Route Analytics | Historical trip patterns from 500K trips — busiest routes, monthly trends, storm risk |
| Risk Intelligence | Delay risk prediction across all weather conditions with gauge chart |
| Fleet Analytics | Truck type distribution, average weights, city volumes, monthly trends |

## Architecture

```
User query (natural language or manual input)
        |
        v
Groq LLaMA 3.3 70B
parses query into structured JSON
        |
        v
NetworkX Dijkstra
finds optimal multi-hop route
weighted by truck type, weather, cargo weight
        |
        v
XGBoost
predicts delay risk (AUC 0.683)
        |
        v
Groq LLaMA 3.3 70B
generates route explanation
        |
        v
Folium map + metrics
displayed in Streamlit
```

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | PySpark, Pandas, NumPy |
| Graph Routing | NetworkX (Dijkstra) |
| ML Model | XGBoost, Scikit-learn, MLflow |
| AI / NLP | LangChain, Groq LLaMA 3.3 70B |
| Routing API | OpenRouteService HGV API |
| Geocoding | Nominatim (OpenStreetMap) |
| Storage | Google Drive (Parquet), MongoDB Atlas |
| Visualization | Folium, Plotly |
| App Framework | Streamlit |
| Deployment | Streamlit Community Cloud |

## Dataset

- 500,000 synthetic shipping trips for year 2025
- 20 major US freight hub cities: Los Angeles, Chicago, Dallas, Houston, Atlanta, Memphis, Louisville, New York, Philadelphia, Baltimore, Columbus, Indianapolis, Nashville, Kansas City, Denver, Phoenix, Seattle, San Francisco, Detroit, Miami
- 380 unique city pairs with real ORS HGV road data
- Cargo types: Electronics, Fresh Produce, Machinery, Dry Goods, Chemicals, Auto Parts
- Truck types: Light, Medium, Heavy, Refrigerated, Flatbed
- Cargo-truck validation rules based on real logistics constraints

## ML Model

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| ROC-AUC | 0.683 |
| Baseline (Random Forest) | 0.672 |
| Training size | 400,000 rows |
| Test size | 100,000 rows |
| Features | truck type, weather, cargo, weight, month, quarter, route distance |
| Leakage fix | Probabilistic labels with Gaussian noise |

## Routing Algorithm

- Graph: NetworkX DiGraph, 20 nodes (cities), 380 directed edges (ORS routes)
- Algorithm: Dijkstra weighted by adjusted duration
- Weight formula: base duration times truck multiplier times weather penalty times weight factor
- Multi-hop: Dijkstra finds the globally optimal path and may route through an intermediate city when that is faster than going direct

## Project Structure

```
shipping-route-optimizer/
    app.py                          Streamlit application
    requirements.txt                Python dependencies
    runtime.txt                     Python version (3.11)
    .streamlit/
        config.toml                 Theme configuration
    data/
        cities.json                 20 city coordinates
        route_graph_1.json          ORS routes (95 pairs)
        route_graph_2.json          ORS routes (95 pairs)
        route_graph_3.json          ORS routes (95 pairs)
        route_graph_4.json          ORS routes (95 pairs)
        route_stats.parquet         Route analytics
        monthly_stats.parquet       Monthly analytics
        city_stats.parquet          City analytics
        rf_model.pkl                Trained XGBoost model
        encoders.pkl                Label encoders
    notebook/
        shipping_optimization.ipynb Full Colab notebook
```

## How to Run Locally

```bash
git clone https://github.com/Chanduatwork/shipping-route-optimizer
cd shipping-route-optimizer
pip install -r requirements.txt
```

Add API keys to `.streamlit/secrets.toml`:

```toml
ORS_KEY = "your_ors_key"
GROQ_KEY = "your_groq_key"
```

Run the app:

```bash
streamlit run app.py
```

## API Keys Required

| API | Purpose | Cost |
|---|---|---|
| OpenRouteService | HGV truck routing | Free, 2000 requests/day |
| Groq | LLaMA 3.3 70B LLM | Free |
| Nominatim | City geocoding | Free, no key needed |
| MongoDB Atlas | Document storage | Free, M0 tier |

## Author

Chandu Vemmasani
GitHub: [Chanduatwork](https://github.com/Chanduatwork)
LinkedIn: [linkedin.com/in/chandu-301](https://linkedin.com/in/chandu-301)
Email: chanduvemmasani@gmail.com
