import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable


# Automatically determine the download period in UTC.
end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
start_time = (end_time - timedelta(days=1)).replace(hour=0, minute=0)

# Stations grouped by country/network.
stations_by_network = {
    "GB__ASOS": """
        EGAA EGAC EGAE EGBB EGBE EGBJ EGCC EGCK EGCN EGDG EGDL EGDM EGDR EGDX EGDY
        EGEC EGEO EGFF EGGD EGGP EGGW EGHC EGHD EGHE EGHH EGHI EGHK EGHQ EGJA EGJB
        EGJJ EGKA EGKB EGKK EGLC EGLF EGLL EGMC EGMD EGMH EGMW EGNC EGNH EGNJ EGNM
        EGNO EGNR EGNS EGNT EGNV EGNX EGOM EGOP EGOS EGOV EGOW EGPA EGPB EGPC EGPD
        EGPE EGPF EGPH EGPI EGPK EGPL EGPM EGPN EGPO EGPU EGQA EGQK EGQL EGQM EGQS
        EGSC EGSD EGSH EGSS EGSY EGTC EGTE EGTG EGTK EGUB EGUL EGUN EGUO EGUW EGUY
        EGVA EGVN EGVO EGVP EGWC EGWU EGXC EGXD EGXE EGXG EGXP EGXS EGXT EGXU EGXV
        EGXW EGXZ EGYC EGYD EGYE EGYH EGYM
    """.split(),
    "FR__ASOS": """
        LFAC LFAQ LFAT LFBA LFBC LFBD LFBE LFBF LFBG LFBH LFBI LFBL LFBM LFBO LFBP
        LFBS LFBT LFBU LFBX LFBY LFBZ LFCG LFCK LFCR LFDB LFDH LFGA LFGJ LFHP LFJL
        LFJR LFKB LFKC LFKF LFKJ LFKS LFLB LFLC LFLD LFLH LFLL LFLM LFLN LFLP LFLQ
        LFLS LFLU LFLV LFLW LFLX LFLY LFMC LFMD LFMH LFMI LFMK LFML LFMN LFMO LFMP
        LFMT LFMU LFMV LFMY LFOA LFOB LFOC LFOE LFOF LFOH LFOI LFOJ LFOK LFOP LFOR
        LFOT LFOV LFOW LFOZ LFPB LFPC LFPG LFPM LFPN LFPO LFPT LFPV LFQA LFQB LFQG
        LFQI LFQQ LFRB LFRC LFRD LFRG LFRH LFRI LFRJ LFRK LFRL LFRM LFRN LFRO LFRQ
        LFRS LFRT LFRU LFRV LFRZ LFSB LFSD LFSF LFSG LFSI LFSL LFSM LFSN LFSO LFSR
        LFST LFSX LFTH LFTW LFVP LFXA LFYR
    """.split(),
    "NL__ASOS": """
        EHAK EHAM EHBK EHDL EHDV EHEH EHFD EHFS EHFZ EHGG EHGR EHHW EHJA EHJR EHKD
        EHKV EHLE EHLW EHMA EHMG EHPG EHQE EHQN EHQS EHQW EHRD EHSA EHSC EHSG EHVK
        EHVL EHWO
    """.split(),
    "DE__ASOS": """
        EDAC EDAH EDAR EDBC EDDB EDDC EDDE EDDF EDDG EDDH EDDK EDDL EDDM EDDN EDDP
        EDDR EDDS EDDT EDDV EDDW EDFE EDFH EDFM EDGS EDHI EDHK EDHL EDJA EDKA EDLN
        EDLP EDLV EDLW EDMA EDMO EDNY EDOP EDQM EDRZ EDSB EDTD EDTL EDTY EDVE EDVK
        EDWE EDWI EDXW ETAD ETAR ETEB ETEK ETGG ETGI ETGQ ETGY ETGZ ETHA ETHB ETHC
        ETHE ETHF ETHL ETHN ETHR ETHS ETIC ETIE ETIH ETIK ETML ETMN ETND ETNG ETNH
        ETNJ ETNL ETNN ETNS ETNT ETNU ETNW ETOR ETOU ETSA ETSB ETSE ETSH ETSI ETSL
        ETSN ETUO ETWM
    """.split(),
    "BE__ASOS": """
        EBAW EBBE EBBL EBBR EBCI EBCV EBDT EBFN EBFS EBLB EBLG EBOS
    """.split(),
    "IE__ASOS": """
        EICK EICM EIDL EIDW EIKN EIKY EIME EINN EISG EIWF
    """.split(),
    "DK__ASOS": """
        EKAH EKAV EKBI EKCH EKEB EKGC EKGF EKHA EKHD EKHN EKHR EKKA EKOD EKRK EKRN
        EKRS EKSB EKSN EKSP EKTE EKVD EKVG EKVJ EKYT
    """.split(),
    "SE__ASOS": """
        ESCF ESCM ESDF ESGG ESGJ ESGL ESGP ESGR ESGT ESIA ESIB ESKM ESKN ESKS ESMK
        ESMQ ESMS ESMT ESMV ESMX ESND ESNG ESNK ESNL ESNN ESNO ESNQ ESNS ESNU ESNV
        ESNX ESNY ESNZ ESOE ESOH ESOK ESOW ESPA ESPE ESSA ESSB ESSD ESSF ESSK ESSL
        ESSP ESST ESSV ESTA ESTL ESUD ESUP ESUT
    """.split(),
    "CH__ASOS": """
        LSGC LSGG LSGS LSMA LSMC LSMD LSME LSMK LSMM LSMP LSMS LSMV LSZA LSZB LSZC
        LSZG LSZH LSZL LSZR LSZS
    """.split(),
    "AT__ASOS": """
        LOAG LOAN LOAV LOIH LOWG LOWI LOWK LOWL LOWS LOWW LOWZ LOXA LOXN LOXT LOXZ
        TTPP
    """.split(),
    "CZ__ASOS": """
        LKCV LKKB LKKU LKKV LKLB LKLN LKMT LKNA LKPD LKPJ LKPO LKPR LKTB LKVO
    """.split(),
    "NO__ASOS": """
        ENAL ENAN ENAT ENBL ENBN ENBO ENBR ENBS ENBV ENCN ENDR ENDU ENEK ENEV ENFB
        ENFG ENFL ENGC ENGK ENGM ENHD ENHE ENHF ENHK ENHM ENHV ENJS ENKB ENKR ENLA
        ENLE ENLK ENMH ENML ENMS ENNA ENNE ENNK ENNM ENNO ENOA ENOL ENOV ENQA ENQC
        ENQR ENRA ENRM ENRO ENRS ENRY ENSB ENSD ENSE ENSF ENSG ENSH ENSK ENSL ENSN
        ENSO ENSR ENSS ENST ENTC ENTO ENUG ENUN ENVA ENVD ENVH ENVR ENWV ENZV
    """.split(),
}

# Combine the station lists, removing any duplicate station codes.
stations = list(dict.fromkeys(
    station
    for country_stations in stations_by_network.values()
    for station in country_stations
))

url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
params = {
    "network": list(stations_by_network),
    "station": stations,
    "data": "gust_mph",
    "year1": start_time.year,
    "month1": start_time.month,
    "day1": start_time.day,
    "hour1": start_time.hour,
    "minute1": start_time.minute,
    "year2": end_time.year,
    "month2": end_time.month,
    "day2": end_time.day,
    "hour2": end_time.hour,
    "minute2": end_time.minute,
    "tz": "Etc/UTC",
    "format": "onlycomma",
    "latlon": "yes",
    "elev": "no",
    "missing": "M",
    "trace": "T",
    "direct": "no",
    "report_type": [3, 4],
}

response = requests.get(url, params=params, timeout=300)
response.raise_for_status()

df = pd.read_csv(StringIO(response.text), na_values=["M"], comment="#")
required_columns = {"station", "valid", "lon", "lat", "gust_mph"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Unexpected response from IEM: {response.text[:300]}")

df["valid"] = pd.to_datetime(df["valid"], utc=True)
df["gust_mph"] = pd.to_numeric(df["gust_mph"], errors="coerce")
df = df.sort_values(["station", "valid"]).reset_index(drop=True)

print(f"Period: {start_time:%Y-%m-%d %H:%M} to {end_time:%Y-%m-%d %H:%M} UTC")
print(f"Requested {len(stations):,} stations across {len(stations_by_network)} networks.")
print(f"Downloaded {len(df):,} observations across {df['station'].nunique()} stations.")

# User settings.
FIGURE_SIZE = (10, 12)
TITLE_FONT_SIZE = 20
SUBTITLE_FONT_SIZE = 12
LEGEND_FONT_SIZE = 14
X_AXIS_TITLE_FONT_SIZE = 13
Y_AXIS_TITLE_FONT_SIZE = 13
X_AXIS_TICK_FONT_SIZE = 11
Y_AXIS_TICK_FONT_SIZE = 11
GUST_LABEL_FONT_SIZE = 13
LABEL_OUTLINE_WIDTH = 1.4
OUTLINE_BRIGHTNESS_THRESHOLD = 0.60  # Higher = white outlines used sooner.
MAP_EXTENT = [-12, 15, 42, 57]
GUST_THRESHOLD = 50
COLOUR_MIN = 50
COLOUR_MAX = 100
COLOUR_STEP = 5

FONT_REGULAR = "/home/richd/.local/share/fonts/Inter_28pt-Medium.ttf"
FONT_BOLD = "/home/richd/.local/share/fonts/Inter_28pt-Bold.ttf"
regular_font = fm.FontProperties(fname=FONT_REGULAR)
bold_font = fm.FontProperties(fname=FONT_BOLD)
oakwhite = (238/255, 250/255, 255/255)
oakdarkblue = (16/255, 21/255, 51/255)

def get_outline_colour(rgba):
    # Calculate weighted brightness directly from the displayed RGB colour.
    red, green, blue = rgba[:3]
    brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if brightness >= OUTLINE_BRIGHTNESS_THRESHOLD else "white"

# Determine the time range of the downloaded observations.
data_times = pd.to_datetime(df["valid"], utc=True, errors="coerce").dropna()
if data_times.empty:
    raise ValueError("No valid observation timestamps are available.")
subtitle = (
    f"{data_times.min():%d %b %Y %H:%M} to "
    f"{data_times.max():%d %b %Y %H:%M} UTC"
)

# Retain the observation containing the highest gust at each station.
plot_data = df.copy().reset_index(drop=True)
for column in ["lon", "lat", "gust_mph"]:
    plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
plot_data = plot_data.dropna(subset=["station", "lon", "lat", "gust_mph"])
if plot_data.empty:
    raise ValueError("No valid gust observations are available to plot.")

max_gusts = plot_data.loc[
    plot_data.groupby("station")["gust_mph"].idxmax()
].sort_values("gust_mph").reset_index(drop=True)
max_gusts = max_gusts.loc[max_gusts["gust_mph"] > GUST_THRESHOLD].copy()

# Fixed colour scale with discrete 5 mph bands.
colour_levels = list(range(COLOUR_MIN, COLOUR_MAX + COLOUR_STEP, COLOUR_STEP))
cmap = plt.get_cmap("viridis_r", len(colour_levels) - 1)
norm = BoundaryNorm(colour_levels, ncolors=cmap.N, clip=True)

# Draw the map and country boundaries.
coordinates = ccrs.PlateCarree()
fig, ax = plt.subplots(
    figsize=FIGURE_SIZE,
    subplot_kw={"projection": ccrs.Mercator()},
    layout="constrained",
)
fig.patch.set_facecolor("white")
ax.set_extent(MAP_EXTENT, crs=coordinates)
ax.set_facecolor(oakwhite)
ax.add_feature(
    cfeature.LAND.with_scale("10m"),
    facecolor="#D5D4D4C1",
    edgecolor="none",
    zorder=0,
)
ax.coastlines(resolution="10m", color=oakdarkblue, linewidth=0.7)
ax.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    edgecolor=oakdarkblue,
    linewidth=0.8,
    zorder=2,
)

# Use each label's colour to select its black or white outline.
for row in max_gusts.itertuples(index=False):
    label_colour = cmap(norm(row.gust_mph))
    ax.text(
        row.lon,
        row.lat,
        f"{row.gust_mph:.0f}",
        transform=coordinates,
        ha="center",
        va="center",
        fontproperties=bold_font,
        fontsize=GUST_LABEL_FONT_SIZE,
        color=label_colour,
        path_effects=[
            pe.Stroke(
                linewidth=LABEL_OUTLINE_WIDTH,
                foreground=get_outline_colour(label_colour),
            ),
            pe.Normal(),
        ],
        clip_on=True,
        zorder=3,
    )

# Longitude and latitude labels.
grid = ax.gridlines(
    draw_labels=True,
    linewidth=0.4,
    color="grey",
    alpha=0.4,
    linestyle="--",
    x_inline=False,
    y_inline=False,
)
grid.top_labels = False
grid.right_labels = False
grid.xlabel_style = {
    "fontproperties": regular_font,
    "fontsize": X_AXIS_TICK_FONT_SIZE,
    "color": oakdarkblue,
}
grid.ylabel_style = {
    "fontproperties": regular_font,
    "fontsize": Y_AXIS_TICK_FONT_SIZE,
    "color": oakdarkblue,
}
ax.set_xlabel(
    "Longitude",
    fontproperties=regular_font,
    fontsize=X_AXIS_TITLE_FONT_SIZE,
    labelpad=20,
    color=oakdarkblue,
)
ax.set_ylabel(
    "Latitude",
    fontproperties=regular_font,
    fontsize=Y_AXIS_TITLE_FONT_SIZE,
    labelpad=35,
    color=oakdarkblue,
)

# Main title and smaller date/time subheading.
ax.set_title(
    f"Maximum wind gust by station (> {GUST_THRESHOLD} mph)",
    fontproperties=bold_font,
    fontsize=TITLE_FONT_SIZE,
    color=oakdarkblue,
    pad=40,
)
ax.text(
    0.5,
    1.015,
    subtitle,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontproperties=regular_font,
    fontsize=SUBTITLE_FONT_SIZE,
    color=oakdarkblue,
    clip_on=False,
)

# Colour scale matching the label colours.
colourbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    boundaries=colour_levels,
    ticks=colour_levels,
    spacing="uniform",
    drawedges=True,
    shrink=0.7,
    pad=0.04,
)
colourbar.set_label(
    "Maximum gust (mph)",
    fontproperties=regular_font,
    fontsize=LEGEND_FONT_SIZE,
    color=oakdarkblue,
)
colourbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE, colors=oakdarkblue)
for label in colourbar.ax.get_yticklabels():
    label.set_fontproperties(regular_font)
    label.set_fontsize(LEGEND_FONT_SIZE)

plt.show()