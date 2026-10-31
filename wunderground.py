import base64
from math import ceil
from pathlib import Path
import time
import pandas as pd
import plotly.graph_objects as go
import requests

API_KEY = "e5b2a88d11944d64b2a88d11946d6443"
STATIONS = {
    "ICIREN71": {"nickname": "Cirencester", "colour": "#1f77b4"},
    "IBUCKING20": {"nickname": "Great Missenden", "colour": "#d62728"},
    "IESSWEND2": {"nickname": "Saffron Walden", "colour": "#910367"},
    "IDEAL50": {"nickname": "Deal", "colour": "#ff7f0e"},
}
TIMEZONE = "Europe/London"
API_URL = "https://api.weather.com/v2/pws/observations/all/1day"
FONT_PATH = Path("/home/richd/.local/share/fonts/Inter_28pt-Black.ttf")
FONT_FAMILY = "Volte, Arial, sans-serif"
OUTPUT_DIR = Path(_file).resolve().parent if "file_" in globals() else Path.cwd()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
html_path = OUTPUT_DIR / "wunderground.html"

LOCATION_COLUMNS = 2
LOCATION_BOX_LEFT = 0.01
LOCATION_BOX_RIGHT = 0.35
LOCATION_BOX_TOP = 0.98
LOCATION_FIRST_ROW_Y = 0.955
LOCATION_ROW_SPACING = 0.033
LOCATION_COLUMN_X = [0.022, 0.18]
LOCATION_LABEL_OFFSET = 0.021
MAX_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 60

today = pd.Timestamp.now(tz=TIMEZONE).date()
file_date = today.isoformat()

def download_station_table(session, station_id):
    try:
        response = session.get(
            API_URL,
            params={
                "stationId": station_id,
                "units": "m",
                "format": "json",
                "numericPrecision": "decimal",
                "apiKey": API_KEY,
            },
            timeout=30,
        )
    except requests.RequestException:
        raise RuntimeError("Connection failed or timed out.") from None
    if response.status_code in (401, 403):
        message = response.text.replace(API_KEY, "[REDACTED]")[:500]
        raise PermissionError(
            f"API authentication/access error {response.status_code}: {message}"
        )
    if response.status_code == 204:
        raise ValueError("The station returned no observations.")
    if response.status_code != 200:
        message = response.text.replace(API_KEY, "[REDACTED]")[:500]
        raise RuntimeError(f"API error {response.status_code}: {message}")
    table = pd.json_normalize(response.json().get("observations", []))
    if table.empty:
        raise ValueError("The station returned no observations.")
    required = {"obsTimeUtc", "metric.tempHigh", "metric.dewptHigh"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing API fields: {', '.join(sorted(missing))}")
    table["Observation_UTC"] = pd.to_datetime(
        table["obsTimeUtc"], utc=True, errors="coerce"
    )
    local_times = table["Observation_UTC"].dt.tz_convert(TIMEZONE)
    table = table.loc[local_times.dt.date == today].copy()
    # Plot UK wall-clock times consistently, regardless of browser timezone.
    table["Datetime"] = (
        table["Observation_UTC"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    )
    # These match the high-value fields used by the website's table.
    table["Temperature_C"] = pd.to_numeric(table["metric.tempHigh"], errors="coerce")
    table["Dew_Point_C"] = pd.to_numeric(table["metric.dewptHigh"], errors="coerce")
    table["Station"] = station_id
    table["Location"] = STATIONS[station_id]["nickname"]
    table = (
        table.dropna(subset=["Datetime", "Temperature_C", "Dew_Point_C"])
        .sort_values("Observation_UTC")
        .drop_duplicates("Observation_UTC", keep="last")
    )
    if table.empty:
        raise ValueError("No valid temperature/dew-point observations for today.")
    return table

successful_stations = {}
with requests.Session() as session:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[Attempt {attempt}/{MAX_ATTEMPTS}] Fetching today's weather...")
        for station_id, settings in STATIONS.items():
            if station_id in successful_stations:
                continue
            try:
                table = download_station_table(session, station_id)
                successful_stations[station_id] = table
                print(
                    f"  OK: {settings['nickname']} ({station_id}) — "
                    f"{len(table):,} observations"
                )
            except PermissionError:
                raise
            except (RuntimeError, ValueError) as error:
                print(f"  Failed: {settings['nickname']} ({station_id}): {error}")
        if len(successful_stations) == len(STATIONS):
            break
        if attempt < MAX_ATTEMPTS:
            print(
                f"Obtained {len(successful_stations)}/{len(STATIONS)} stations. "
                f"Retrying missing stations in {RETRY_DELAY_SECONDS} seconds..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

if len(successful_stations) < len(STATIONS):
    missing = [
        f"{settings['nickname']} ({station_id})"
        for station_id, settings in STATIONS.items()
        if station_id not in successful_stations
    ]
    raise RuntimeError(
        f"Pipeline aborted after {MAX_ATTEMPTS} attempts. "
        f"Missing stations: {', '.join(missing)}"
    )

print("\nAll four locations retrieved. Building HTML chart...")
pull_time = pd.Timestamp.now(tz=TIMEZONE).strftime("%d%m%y | %H%M")
weather = pd.concat(
    [successful_stations[station_id] for station_id in STATIONS],
    ignore_index=True,
)
available_station_ids = list(STATIONS)
master_times = pd.DatetimeIndex(sorted(weather["Datetime"].dropna().unique()))
fig = go.Figure()
aligned_station_data = {}

for station_id in available_station_ids:
    settings = STATIONS[station_id]
    station_data = (
        weather.loc[weather["Station"] == station_id]
        .sort_values("Observation_UTC")
        .drop_duplicates("Datetime", keep="last")
        .sort_values("Datetime")
    )
    for column, label, dash in [
        ("Temperature_C", "temperature", "solid"),
        ("Dew_Point_C", "dew point", "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=station_data["Datetime"],
            y=station_data[column],
            mode="lines",
            line={"color": settings["colour"], "width": 2.5, "dash": dash},
            name=f"{settings['nickname']} {label}",
            showlegend=False,
            hoverinfo="skip",
        ))
    observations = station_data.set_index("Datetime")[
        ["Temperature_C", "Dew_Point_C"]
    ]
    aligned = observations.reindex(master_times).ffill()
    observation_times = pd.Series(
        observations.index.strftime("%H%M"), index=observations.index
    )
    aligned["Observation_Time"] = observation_times.reindex(master_times).ffill()
    aligned_station_data[station_id] = aligned

hover_text, hover_y = [], []
for timestamp in master_times:
    location_sections, temperatures = [], []
    for station_id in available_station_ids:
        settings = STATIONS[station_id]
        row = aligned_station_data[station_id].loc[timestamp]
        if row[["Observation_Time", "Temperature_C", "Dew_Point_C"]].isna().any():
            continue
        location_sections.append(
            f"<span style='color:{settings['colour']}'>&#9679;</span> "
            f"<b>{settings['nickname']}: {row['Observation_Time']}</b><br>"
            f"    Temperature: {row['Temperature_C']:.1f} °C<br>"
            f"    Dewpoint: {row['Dew_Point_C']:.1f} °C"
        )
        temperatures.append(row["Temperature_C"])
    hover_text.append("<br><br>".join(location_sections))
    hover_y.append(sum(temperatures) / len(temperatures) if temperatures else None)

fig.add_trace(go.Scatter(
    x=master_times,
    y=hover_y,
    text=hover_text,
    mode="markers",
    marker={"size": 18, "color": "rgba(0,0,0,0)"},
    showlegend=False,
    hovertemplate="%{text}<extra></extra>",
))
for label, dash in [("Temperature", "solid"), ("Dew point", "dash")]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line={"color": "#333333", "width": 2.5, "dash": dash},
        name=label, hoverinfo="skip",
    ))

location_rows = ceil(len(available_station_ids) / LOCATION_COLUMNS)
location_box_bottom = (
    LOCATION_FIRST_ROW_Y - (location_rows - 1) * LOCATION_ROW_SPACING - 0.022
)
fig.update_layout(
    width=1000,
    height=750,
    font={"family": FONT_FAMILY, "size": 13, "color": "#222222"},
    title={
        "text": f"Wunderground Temperature and Dewpoint — {file_date}",
        "x": 0.5, "xanchor": "center",
    },
    annotations=[{
        "xref": "paper", "yref": "paper",
        "x": 1, "y": 1.08,
        "text": pull_time,
        "showarrow": False,
        "xanchor": "right", "yanchor": "top",
        "font": {"family": FONT_FAMILY, "size": 10, "color": "#666666"},
    }],
    xaxis={
        "title": "Hour (UK time)",
        "tickmode": "linear",
        "dtick": 60 * 60 * 1000,
        "tickformat": "%H",
        "showgrid": True,
        "gridcolor": "#d9d9d9",
    },
    yaxis={
        "title": "Temperature (°C)",
        "showgrid": True,
        "gridcolor": "#d9d9d9",
    },
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x",
    hoverdistance=100,
    hoverlabel={
        "font": {"family": FONT_FAMILY, "size": 14},
        "bgcolor": "#eeeeee",
        "bordercolor": "#999999",
        "align": "left",
    },
    legend={
        "x": LOCATION_BOX_LEFT,
        "y": location_box_bottom - 0.008,
        "xanchor": "left",
        "yanchor": "top",
        "orientation": "h",
        "bgcolor": "#eeeeee",
        "bordercolor": "#aaaaaa",
        "borderwidth": 1,
        "font": {"family": FONT_FAMILY, "size": 12},
        "itemsizing": "constant",
    },
    margin={"l": 80, "r": 40, "t": 80, "b": 90},
)
fig.add_shape(
    type="rect", xref="paper", yref="paper",
    x0=LOCATION_BOX_LEFT, x1=LOCATION_BOX_RIGHT,
    y0=location_box_bottom, y1=LOCATION_BOX_TOP,
    fillcolor="#eeeeee",
    line={"color": "#aaaaaa", "width": 1},
    layer="above",
)
for index, station_id in enumerate(available_station_ids):
    row, column = divmod(index, LOCATION_COLUMNS)
    x_position = LOCATION_COLUMN_X[column]
    y_position = LOCATION_FIRST_ROW_Y - row * LOCATION_ROW_SPACING
    settings = STATIONS[station_id]
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=x_position, x1=x_position + 0.014,
        y0=y_position, y1=y_position,
        line={"color": settings["colour"], "width": 4},
        layer="above",
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=x_position + LOCATION_LABEL_OFFSET, y=y_position,
        text=settings["nickname"], showarrow=False,
        xanchor="left", yanchor="middle",
        font={"family": FONT_FAMILY, "size": 12, "color": "#222222"},
    )
fig.add_annotation(
    xref="paper", yref="paper", x=1, y=-0.12,
    text="Data provided by Weather Underground",
    showarrow=False, xanchor="right",
    font={"family": FONT_FAMILY, "size": 11, "color": "#666666"},
)
html = fig.to_html(include_plotlyjs=True, full_html=True)
if FONT_PATH.is_file():
    font_data = base64.b64encode(FONT_PATH.read_bytes()).decode("ascii")
    font_css = (
        "<style>@font-face{font-family:'Volte';"
        f"src:url(data:font/otf;base64,{font_data}) format('opentype');"
        "font-weight:400;font-style:normal;}</style>"
    )
    html = html.replace("</head>", font_css + "</head>", 1)
html_path.write_text(html, encoding="utf-8")
print(f"Chart saved with all four locations to: {html_path}")
