from datetime import datetime
from io import StringIO
from math import ceil
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.graph_objects as go
import requests

STATIONS = {
    "ICIREN19": {"nickname": "Cirencester", "colour": "#1f77b4"},
    "IGREAT245": {"nickname": "Great Missenden", "colour": "#d62728"},
    "IPEVEN35": {"nickname": "Pevensey Bay", "colour": "#2ca02c"},
    "ISAFFR63": {"nickname": "Saffron Walden", "colour": "#910367"},
}

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR # or just REPO_DIR if you want root

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT_DIR / f"wunderground.csv"
html_path = OUTPUT_DIR / f"wunderground.html"

LOCATION_COLUMNS = 2
LOCATION_BOX_LEFT = 0.01
LOCATION_BOX_RIGHT = 0.35
LOCATION_BOX_TOP = 0.98
LOCATION_FIRST_ROW_Y = 0.955
LOCATION_ROW_SPACING = 0.033
LOCATION_COLUMN_X = [0.022, 0.18]
LOCATION_LABEL_OFFSET = 0.021

today = datetime.now(ZoneInfo("Europe/London")).date()
url_date = f"{today.year}-{today.month}-{today.day}"
file_date = today.strftime("%Y-%m-%d")


def extract_number(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def download_station_table(station_id):
    url = (
        f"https://www.wunderground.com/dashboard/pws/{station_id}/table/"
        f"{url_date}/{url_date}/daily"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [
                " ".join(str(x) for x in column if str(x) != "nan").strip()
                for column in table.columns
            ]
        else:
            table.columns = [str(column).strip() for column in table.columns]

        required = {"Time", "Temperature", "Dew Point"}
        if required.issubset(table.columns):
            table = table.copy()
            table["Station"] = station_id
            table["Location"] = STATIONS[station_id]["nickname"]

            time_text = table["Time"].astype(str).str.strip()
            table["Datetime"] = pd.to_datetime(
                file_date + " " + time_text,
                errors="coerce",
            )

            temperature_f = extract_number(table["Temperature"])
            dewpoint_f = extract_number(table["Dew Point"])

            table["Temperature_C"] = (temperature_f - 32) * 5 / 9
            table["Dew_Point_C"] = (dewpoint_f - 32) * 5 / 9

            table = table.dropna(
                subset=["Datetime", "Temperature_C", "Dew_Point_C"]
            )
            table = table.sort_values("Datetime")
            table = table.drop_duplicates("Datetime", keep="last")
            return table

    raise ValueError(f"Weather table not found for {station_id}")


station_frames = []

for station_id, settings in STATIONS.items():
    try:
        station_table = download_station_table(station_id)
        station_frames.append(station_table)
        print(
            f"Downloaded {len(station_table)} observations for "
            f"{settings['nickname']} ({station_id})"
        )
    except Exception as error:
        print(f"Could not download {station_id}: {error}")

if not station_frames:
    raise RuntimeError("No station data could be downloaded.")

weather = pd.concat(station_frames, ignore_index=True)

csv_path = OUTPUT_DIR / f"wunderground_weather.csv"
weather.to_csv(csv_path, index=False)

available_station_ids = [
    station_id
    for station_id in STATIONS
    if station_id in weather["Station"].unique()
]

master_times = pd.DatetimeIndex(
    sorted(weather["Datetime"].dropna().unique())
)

fig = go.Figure()
aligned_station_data = {}

for station_id in available_station_ids:
    settings = STATIONS[station_id]
    nickname = settings["nickname"]
    colour = settings["colour"]

    station_data = (
        weather.loc[weather["Station"] == station_id]
        .sort_values("Datetime")
        .drop_duplicates("Datetime", keep="last")
    )

    # Visible lines do not create their own hover labels.
    fig.add_trace(
        go.Scatter(
            x=station_data["Datetime"],
            y=station_data["Temperature_C"],
            mode="lines",
            line={"color": colour, "width": 2.5},
            name=f"{nickname} temperature",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=station_data["Datetime"],
            y=station_data["Dew_Point_C"],
            mode="lines",
            line={"color": colour, "width": 2.5, "dash": "dash"},
            name=f"{nickname} dew point",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    observations = station_data.set_index("Datetime")[
        ["Temperature_C", "Dew_Point_C"]
    ]

    aligned = observations.reindex(master_times).ffill()

    observation_times = pd.Series(
        observations.index.strftime("%H%M"),
        index=observations.index,
    )

    aligned["Observation_Time"] = (
        observation_times.reindex(master_times).ffill()
    )

    aligned_station_data[station_id] = aligned

# Build one hover label containing all locations.
hover_text = []
hover_y = []

for timestamp in master_times:
    location_sections = []
    temperatures = []

    for station_id in available_station_ids:
        settings = STATIONS[station_id]
        row = aligned_station_data[station_id].loc[timestamp]

        if (
            pd.isna(row["Observation_Time"])
            or pd.isna(row["Temperature_C"])
            or pd.isna(row["Dew_Point_C"])
        ):
            continue

        location_sections.append(
            f"<span style='color:{settings['colour']}'>●</span> "
            f"<b>{settings['nickname']}: {row['Observation_Time']}</b><br>"
            f"    Temperature: {row['Temperature_C']:.1f} °C<br>"
            f"    Dewpoint: {row['Dew_Point_C']:.1f} °C"
        )
        temperatures.append(row["Temperature_C"])

    hover_text.append("<br><br>".join(location_sections))
    hover_y.append(
        sum(temperatures) / len(temperatures) if temperatures else None
    )

# This single transparent trace controls the complete hover window.
fig.add_trace(
    go.Scatter(
        x=master_times,
        y=hover_y,
        text=hover_text,
        mode="markers",
        marker={
            "size": 18,
            "color": "rgba(0,0,0,0)",
        },
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    )
)

# Separate line-style legend.
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line={"color": "#333333", "width": 2.5},
        name="Temperature",
        hoverinfo="skip",
    )
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line={"color": "#333333", "width": 2.5, "dash": "dash"},
        name="Dew point",
        hoverinfo="skip",
    )
)

location_rows = ceil(len(available_station_ids) / LOCATION_COLUMNS)
location_box_bottom = (
    LOCATION_FIRST_ROW_Y
    - (location_rows - 1) * LOCATION_ROW_SPACING
    - 0.022
)

fig.update_layout(
    width=1000,
    height=750,
    font={"family": "Arial", "size": 13, "color": "#222222"},
    title={
        "text": f"Temperature and Dew Point — {file_date}",
        "x": 0.5,
        "xanchor": "center",
    },
    xaxis={
        "title": "Hour",
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

    # Do not use "x unified": it creates the unwanted hour heading.
    hovermode="x",
    hoverdistance=100,
    hoverlabel={
        "font": {"family": "Arial", "size": 14},
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
        "font": {"family": "Arial", "size": 12},
        "itemsizing": "constant",
    },
    margin={"l": 80, "r": 40, "t": 80, "b": 70},
)

# Compact two-column location legend.
fig.add_shape(
    type="rect",
    xref="paper",
    yref="paper",
    x0=LOCATION_BOX_LEFT,
    x1=LOCATION_BOX_RIGHT,
    y0=location_box_bottom,
    y1=LOCATION_BOX_TOP,
    fillcolor="#eeeeee",
    line={"color": "#aaaaaa", "width": 1},
    layer="above",
)

for index, station_id in enumerate(available_station_ids):
    row = index // LOCATION_COLUMNS
    column = index % LOCATION_COLUMNS
    x_position = LOCATION_COLUMN_X[column]
    y_position = LOCATION_FIRST_ROW_Y - row * LOCATION_ROW_SPACING
    settings = STATIONS[station_id]

    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=x_position,
        x1=x_position + 0.014,
        y0=y_position,
        y1=y_position,
        line={"color": settings["colour"], "width": 4},
        layer="above",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x_position + LOCATION_LABEL_OFFSET,
        y=y_position,
        text=settings["nickname"],
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font={"family": "Arial", "size": 12, "color": "#222222"},
    )

fig.write_html(html_path, include_plotlyjs=True)

print(f"CSV saved to:  {csv_path}")
print(f"Chart saved to: {html_path}")
