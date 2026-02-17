import streamlit as st
import pandas as pd
import numpy as np
import heapq
import time
import requests
import folium
import random
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, PolyLineTextPath


# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Optimasi Rute Wisata",
    page_icon=r"C:\Users\Ashilah\OneDrive\Dokumen\Dokumen Ashilah\Per undip an\Universitas-Diponegoro-Semarang-Logo.png",
    layout="wide"
)

st.title(
    "PERBANDINGAN ALGORITMA NEAREST NEIGHBOR DAN ANT COLONY OPTIMIZATION "
    "DALAM OPTIMASI RUTE WISATA SEMARANG "
    "BERBASIS MULTI-ATTRIBUTE UTILITY THEORY"
)

st.markdown("""
**Nama:** Ashilah Tsuraya Izzati  
**NIM:** 24050122140129  
**Program Studi:** Statistika
""")


# ======================================================
# UPLOAD DATA
# ======================================================
uploaded_dest = st.file_uploader(
    "Upload Data Destinasi (.csv / .xlsx)",
    type=["csv", "xlsx"],
    key="dest"
)

uploaded_dist = st.file_uploader(
    "Upload Data Jarak Antardestinasi (.csv / .xlsx)",
    type=["csv", "xlsx"],
    key="dist"
)


def read_data(uploaded_file):
    """Membaca file CSV / XLSX"""
    if uploaded_file is None:
        return None

    name = uploaded_file.name
    try:
        if name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file, sep=";", engine="python")
            except Exception:
                return pd.read_csv(uploaded_file, engine="python")
        elif name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.warning(f"Gagal membaca file {name}: {e}")
        return None


if not (uploaded_dest and uploaded_dist):
    st.info("Silakan upload kedua file (Destinasi dan Jarak).")
    st.stop()


# ======================================================
# BACA FILE
# ======================================================
destinasi = read_data(uploaded_dest)
dist_matrix = read_data(uploaded_dist)

if destinasi is None or dist_matrix is None:
    st.error("Gagal membaca salah satu file. Periksa format dan kolom.")
    st.stop()


# ======================================================
# CLEANING & VALIDASI DATA
# ======================================================
required_cols = ["Name", "Lat", "Lon", "Rating", "Reviews", "Fee"]

for col in required_cols:
    if col not in destinasi.columns:
        st.error(
            f"Kolom '{col}' tidak ditemukan. "
            f"Pastikan kolom: {required_cols}"
        )
        st.stop()

destinasi = destinasi.copy()
destinasi.drop_duplicates(subset=["Name"], inplace=True)
destinasi.dropna(subset=["Rating", "Reviews", "Fee"], inplace=True)

destinasi["Rating"] = destinasi["Rating"].astype(float)
destinasi["Reviews"] = destinasi["Reviews"].astype(int)
destinasi["Fee"] = destinasi["Fee"].astype(float)

names_dist = (
    list(dist_matrix.columns[1:])
    if dist_matrix.shape[1] > 1
    else list(dist_matrix.columns)
)

full_cost = (
    dist_matrix.iloc[:, 1:].to_numpy(dtype=float)
    if dist_matrix.shape[1] > 1
    else dist_matrix.to_numpy(dtype=float)
)

np.fill_diagonal(full_cost, np.inf)

st.success("✅ Data berhasil diproses")
st.markdown("---")

# ======================================================
# PREVIEW DATA
# ======================================================
with st.expander("📄 Lihat Data Input"):
    st.subheader("Data Destinasi")
    st.dataframe(destinasi, use_container_width=True)

    st.subheader("Matriks Jarak Antardestinasi")
    st.dataframe(dist_matrix, use_container_width=True)

# ======================================================
# INPUT TITIK AWAL
# ======================================================
hotel_name = st.text_input("Titik Awal", value=names_dist[0] if len(names_dist)>0 else "Hotel")

if hotel_name not in names_dist:
    st.error(f"⚠️ Hotel '{hotel_name}' tidak ditemukan di matriks waktu/jarak. Pastikan nama persis sama seperti pada header matriks.")
    st.stop()
else:
    hotel_idx = names_dist.index(hotel_name)
    hotel_lat = destinasi.loc[destinasi["Name"] == hotel_name, "Lat"].values[0] if hotel_name in destinasi["Name"].values else destinasi["Lat"].mean()
    hotel_lon = destinasi.loc[destinasi["Name"] == hotel_name, "Lon"].values[0] if hotel_name in destinasi["Name"].values else destinasi["Lon"].mean()

# ======================================================
# FUNGSI VISUALISASI & TABEL RUTE
# ======================================================
st.sidebar.subheader("Pengaturan Visualisasi Rute")
route_type_choice = st.sidebar.radio(
    "Pilih tipe rute yang ingin ditampilkan di semua algoritma:",
    ["Jarak Aktual (Road Distance)", "Jarak Euclidean (Garis Lurus)"]
,      index=0
)

# ===============================
# Fungsi visualisasi rute 
# ===============================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImYwZDNhOGVjZTg5YmJkYTYyMDlhNzMwN2ZlMzg5YjI1OTAxYjg5YzQ1NzdmYzgxMjYzYmYzODBlIiwiaCI6Im11cm11cjY0In0="

def get_real_route(coords):

    if len(coords) < 2:
        return coords

    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    # ORS harus format [lon, lat]
    body = {
        "coordinates": [[lon, lat] for lat, lon in coords],
        "preference": "shortest",
        "instructions": False
    }

    try:
        res = requests.post(url, json=body, headers=headers, timeout=30)
        data = res.json()

        # Debug: tampilkan response jika rute tidak muncul
        if "features" not in data or len(data["features"]) == 0:
            st.warning(f"⚠️ ORS tidak mengembalikan rute. Response: {data}")
            return coords  # fallback garis lurus

        # Ambil koordinat rute
        route_coords = [(pt[1], pt[0]) for pt in data["features"][0]["geometry"]["coordinates"]]
        return route_coords

    except requests.exceptions.Timeout:
        st.warning("⚠️ ORS request timeout. Menampilkan garis lurus sementara.")
        return coords
    except Exception as e:
        st.warning(f"⚠️ ORS request gagal: {e}. Menampilkan garis lurus sementara.")
        return coords


# ======================================================
# Fungsi visualisasi rute
# ======================================================
def visualize_route(
    df_dest,
    order,
    hotel_name,
    route_type="Jarak Aktual (Road Distance)",
    color="green",
    height=500
):
    """Visualisasi rute wisata di Folium"""
    center_lat = df_dest["Lat"].mean()
    center_lon = df_dest["Lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
    MarkerCluster().add_to(m)

    # Ambil koordinat semua titik
    hotel_row = df_dest[df_dest["Name"] == hotel_name].iloc[0]
    coords = [(hotel_row["Lat"], hotel_row["Lon"])]
    for name in order:
        if name != hotel_name:
            row = df_dest[df_dest["Name"] == name].iloc[0]
            coords.append((row["Lat"], row["Lon"]))
    coords.append((hotel_row["Lat"], hotel_row["Lon"]))  # kembali ke hotel

    # Pilih tipe rute
    if route_type == "Jarak Aktual (Road Distance)":
        route_coords = get_real_route(coords)
    else:
        route_coords = coords  # garis lurus

    # Gambar rute
    route_line = folium.PolyLine(route_coords, color=color, weight=4, opacity=0.8)
    route_line.add_to(m)
    PolyLineTextPath(route_line, "➤", repeat=True, offset=10, spacing=100).add_to(m)

    # Tambah marker
    all_stops = [hotel_name] + [n for n in order if n != hotel_name]
    for idx, name in enumerate(all_stops):
        row = df_dest[df_dest["Name"] == name].iloc[0]
        lat, lon = row["Lat"], row["Lon"]
        label_html = f"""
        <div style="display:inline-block;
                    min-width:110px;
                    max-width:160px;
                    font-size:11px;
                    font-weight:600;
                    color:#000;
                    background-color:rgba(255,255,255,0.85);
                    border:1px solid #555;
                    border-radius:8px;
                    padding:6px 10px;
                    text-align:center;">
            <span style="color:#003366;">{idx+1}.</span> {name}
        </div>
        """
        folium.Marker([lat + 0.0006, lon + 0.0003], icon=folium.DivIcon(html=label_html)).add_to(m)

    # Marker hotel
    folium.Marker(
        location=(hotel_row["Lat"], hotel_row["Lon"]),
        popup="Titik Awal / Hotel",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    st_folium(m, width=950, height=height, returned_objects=[])
# ======================================================
# TABEL RUTE
# ======================================================
def route_table_from_path(order, path_idx_seq, cost_matrix):
    rows = []
    total_dist = 0
    for i in range(len(path_idx_seq)-1):
        a, b = path_idx_seq[i], path_idx_seq[i+1]
        dist = cost_matrix[a][b]
        total_dist += dist
        rows.append({
            "Urutan": i+1,
            "Dari": order[a],
            "Ke": order[b],
            "Jarak (km)": round(dist,2)
        })
    df = pd.DataFrame(rows)
    return df, total_dist

# ======================================================
# PERHITUNGAN MAUT
# ======================================================
def hitung_maut(df, w_rating, w_review, w_fee):
    maut = df.copy()

    def norm(col):
        if maut[col].max() == maut[col].min():
            return np.ones(len(maut)) * 0.5
        return (maut[col] - maut[col].min()) / (maut[col].max() - maut[col].min())

    maut["rating_n"] = norm("Rating")
    maut["review_n"] = norm("Reviews")

    maut["fee_n"] = 1 + (maut["Fee"].min() - maut["Fee"]) / (
        maut["Fee"].max() - maut["Fee"].min()
    )

    def utility_exp(x):
        return (np.exp(x**2) - 1) / 1.71

    maut["rating_u"] = utility_exp(maut["rating_n"])
    maut["review_u"] = utility_exp(maut["review_n"])
    maut["fee_u"] = utility_exp(maut["fee_n"])

    maut["utility"] = (
        w_rating * maut["rating_u"]
        + w_review * maut["review_u"]
        + w_fee * maut["fee_u"]
    )

    return maut
# ======================================================
# SUBMATRIX
# ======================================================
def submatrix(topN, full_matrix, all_names, titik_awal):
    selected_names = topN["Name"].tolist()
    selected_names = [n for n in selected_names if n != titik_awal]
    order = [titik_awal] + selected_names
    idxs = [all_names.index(n) for n in order]

    sub_matrix = full_matrix[np.ix_(idxs, idxs)].copy()
    np.fill_diagonal(sub_matrix, np.inf)

    return sub_matrix, order
# ======================================================
# ALGORITMA BRANCH AND BOUND
# ======================================================
def run_branch_and_bound(cost_matrix, order, start_city):
    jarak = np.array(cost_matrix)
    nama_destinasi = order
    num_destinasi = len(nama_destinasi)

    start_idx = nama_destinasi.index(start_city)

    # ===============================
    # LOWER BOUND
    # ===============================
    def bound_estimate(tail, current_path):
        unvisited = [i for i in range(num_destinasi) if i not in current_path]
        if not unvisited:
            return 0
        return min(jarak[tail][j] for j in unvisited)

    # ===============================
    # INISIALISASI
    # ===============================
    heap = []
    best_cost = float("inf")
    best_path = None
    iteration = 0
    log = []

    start_time = time.perf_counter()

    # ===============================
    # LEVEL PERTAMA
    # ===============================
    for j in range(num_destinasi):
        if j != start_idx and jarak[start_idx][j] > 0:
            cost = jarak[start_idx][j]
            estimasi = bound_estimate(j, [start_idx, j])
            bound = cost + estimasi
            heapq.heappush(heap, (bound, cost, [start_idx, j]))

    # ===============================
    # BRANCH AND BOUND
    # ===============================
    while heap:
        iteration += 1
        bound, current_cost, path = heapq.heappop(heap)
        tail = path[-1]

        if bound >= best_cost:
            continue

        # GOAL
        if len(path) == num_destinasi:
            total_cost = current_cost + jarak[tail][start_idx]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path + [start_idx]
            continue

        # BRANCH
        for k in range(num_destinasi):
            if k not in path and jarak[tail][k] > 0:
                new_cost = current_cost + jarak[tail][k]
                estimasi = bound_estimate(k, path + [k])
                new_bound = new_cost + estimasi

                if new_bound < best_cost:
                    heapq.heappush(
                        heap, (new_bound, new_cost, path + [k])
                    )

    runtime = time.perf_counter() - start_time

    return best_path, best_cost, runtime
# ======================================================
# ALGORITMA NEAREST NEIGHBOR
# ======================================================
def run_nearest_neighbor(sub_matrix, order, titik_awal):
    jarak = np.array(sub_matrix)
    nama_destinasi = order
    num_destinasi = len(nama_destinasi)

    start_idx = nama_destinasi.index(titik_awal)

    start_time = time.perf_counter()

    visited = [False] * num_destinasi
    current_city = start_idx
    visited[current_city] = True

    tour = [current_city]
    total_distance_nn = 0.0

    for _ in range(num_destinasi - 1):
        unvisited_cities = [i for i, v in enumerate(visited) if not v]

        nearest_city = min(
            unvisited_cities,
            key=lambda x: jarak[current_city][x]
        )

        nearest_distance = jarak[current_city][nearest_city]

        tour.append(nearest_city)
        visited[nearest_city] = True
        total_distance_nn += nearest_distance
        current_city = nearest_city

    # kembali ke hotel
    tour.append(start_idx)
    total_distance_nn += jarak[current_city][start_idx]

    time_nn = time.perf_counter() - start_time

    return tour, total_distance_nn, time_nn
# ======================================================
# ALGORITMA ANT COLONY OPTIMIZATION
# ======================================================
def run_aco(sub_matrix, order, titik_awal):
    random.seed(42)
    np.random.seed(42)

    jarak = np.array(sub_matrix)
    nama_destinasi = order
    num_destinasi = len(nama_destinasi)

    start_idx = nama_destinasi.index(titik_awal)

    # =========================
    # PARAMETER ACO
    # =========================
    num_ants = 10
    num_iterations = 5
    alpha = 1
    beta = 2
    rho = 0.5
    Q = 1
    initial_pheromone = 1

    pheromones = [[initial_pheromone]*num_destinasi for _ in range(num_destinasi)]
    visibility = [[0 if jarak[i][j] == 0 else 1/jarak[i][j]
                    for j in range(num_destinasi)]
                    for i in range(num_destinasi)]

    best_tour = None
    total_distance_aco = float('inf')
    best_iter = -1
    best_ant = -1

    start_time = time.perf_counter()

    for iteration in range(num_iterations):
        all_tours = []

        for ant in range(num_ants):
            current = start_idx
            visited = [False]*num_destinasi
            visited[current] = True
            tour = [current]

            for _ in range(num_destinasi - 1):
                unvisited = [i for i, v in enumerate(visited) if not v]

                denominator = sum(
                    (pheromones[current][j]**alpha) *
                    (visibility[current][j]**beta)
                    for j in unvisited
                )

                if denominator == 0:
                    chosen = random.choice(unvisited)
                else:
                    probs = [
                        (pheromones[current][j]**alpha *
                         visibility[current][j]**beta) / denominator
                        for j in unvisited
                    ]
                    r = random.random()
                    cum = 0
                    for idx, p in enumerate(probs):
                        cum += p
                        if r <= cum:
                            chosen = unvisited[idx]
                            break

                visited[chosen] = True
                tour.append(chosen)
                current = chosen

            tour.append(start_idx)
            length = sum(jarak[tour[i]][tour[i+1]] for i in range(len(tour)-1))
            all_tours.append((tour, length))

            if length < total_distance_aco:
                total_distance_aco = length
                best_tour = tour
                best_iter = iteration + 1
                best_ant = ant + 1

        # update pheromone
        for tour, length in all_tours:
            deposit = Q / length
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                pheromones[a][b] += deposit
                pheromones[b][a] += deposit

    time_aco = time.perf_counter() - start_time

    return best_tour, total_distance_aco, time_aco, best_iter, best_ant

# ===============================
# TAB
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Multi-Attribute Utility Theory",
    "Branch and Bound",
    "Nearest Neighbor",
    "Ant Colony Optimization",
    "Perbandingan"
])

# ===============================
# TAB 1 – MAUT
# ===============================
with tab1:
    st.subheader("Seleksi Destinasi (MAUT)")

    # ===============================
    # Metode Pembobotan
    # ===============================
    metode_bobot = st.selectbox(
        "Metode pembobotan:",
        ["Preferensi Pengguna", "Bobot ROC", "Bobot Manual"]
    )

    # Default bobot
    w_rating, w_review, w_fee = 0.3, 0.4, 0.3

    # ===============================
    # Preferensi Pengguna
    # ===============================
    if metode_bobot == "Preferensi Pengguna":
        preferensi = st.selectbox(
            "Pilih preferensi utama:",
            ["Trend", "Hemat Biaya", "Kualitas Destinasi"]
        )

        if preferensi == "Hemat Biaya":
            w_rating, w_review, w_fee = 0.3, 0.3, 0.4
        elif preferensi == "Kualitas Destinasi":
            w_rating, w_review, w_fee = 0.4, 0.3, 0.3

    # ===============================
    # Bobot ROC
    # ===============================
    elif metode_bobot == "Bobot ROC":
        r1 = st.selectbox("Peringkat 1", ["Rating", "Reviews", "Fee"])
        r2 = st.selectbox(
            "Peringkat 2",
            [x for x in ["Rating", "Reviews", "Fee"] if x != r1]
        )
        r3 = [x for x in ["Rating", "Reviews", "Fee"] if x not in [r1, r2]][0]

        ranks = [r1, r2, r3]
        n = 3

        roc = {}
        for i, k in enumerate(ranks):
            roc[k] = sum(1/j for j in range(i+1, n+1)) / n

        w_rating = roc["Rating"]
        w_review = roc["Reviews"]
        w_fee = roc["Fee"]

    # ===============================
    # Bobot Manual
    # ===============================
    else:
        with st.expander("🔧 Sesuaikan Bobot Manual"):
            w_rating = st.number_input(
                "Rating", 0.0, 1.0, w_rating, step=0.0001, format="%0.4f"
            )
            w_review = st.number_input(
                "Reviews", 0.0, 1.0, w_review, step=0.0001, format="%0.4f"
            )
            w_fee = st.number_input(
                "Fee", 0.0, 1.0, w_fee, step=0.0001, format="%0.4f"
            )

        total_w = w_rating + w_review + w_fee
        if total_w == 0:
            w_rating, w_review, w_fee = 0.33, 0.33, 0.34
        else:
            w_rating /= total_w
            w_review /= total_w
            w_fee /= total_w

    # ===============================
    # Tampilkan Bobot
    # ===============================
    col1, col2, col3 = st.columns(3)
    col1.metric("Bobot Rating", round(w_rating, 2))
    col2.metric("Bobot Reviews", round(w_review, 2))
    col3.metric("Bobot Fee", round(w_fee, 2))

    st.info(
        f"Bobot akhir → "
        f"Rating={w_rating:.2f}, "
        f"Reviews={w_review:.2f}, "
        f"Fee={w_fee:.2f}"
    )

    st.markdown("---")

    # ===============================
    # Jumlah Destinasi
    # ===============================
    N_DEST = st.number_input(
        "Jumlah destinasi terpilih",
        min_value=1,
        max_value=len(names_dist)-1,
        value=min(10, len(names_dist)-1)
    )

    # ===============================
    # Hitung MAUT
    # ===============================
    hasil_maut = hitung_maut(destinasi, w_rating, w_review, w_fee)

    dest_not_hotel = hasil_maut[hasil_maut["Name"] != hotel_name]
    topN = dest_not_hotel.sort_values("utility", ascending=False).head(N_DEST)

    st.success("Destinasi terpilih berhasil ditentukan")

    st.dataframe(
        topN[["Name", "Rating", "Reviews", "Fee", "utility"]],
        use_container_width=True
    )

    # ===============================
    # Submatrix Jarak
    # ===============================
    sub_matrix, order = submatrix(
        topN=topN,
        full_matrix=full_cost,
        all_names=names_dist,
        titik_awal=hotel_name
    )

    st.session_state["sub_matrix"] = sub_matrix
    st.session_state["order"] = order
    st.session_state["hotel_name"] = hotel_name

    st.subheader("📊 Submatrix Jarak Antar Destinasi Terpilih")
    df_sub_matrix = pd.DataFrame(sub_matrix, index=order, columns=order)
    st.dataframe(df_sub_matrix.round(2), use_container_width=True)

# ===============================
# ISI TAB 2 - BRANCH AND BOUND
# ===============================
with tab2:
    st.subheader("Algoritma Branch and Bound")

    if st.button("Run B&B"):
        sub_matrix = st.session_state["sub_matrix"]
        order = st.session_state["order"]
        hotel_name = st.session_state["hotel_name"]

        path_idx, total_dist, runtime_bb = run_branch_and_bound(sub_matrix, order, hotel_name)

        if path_idx is None:
            st.error("Tidak ditemukan solusi")
        else:
            st.session_state.bnb_path_idx = path_idx
            st.session_state.total_distance_bb = total_dist
            st.session_state.runtime_bb = runtime_bb
            st.metric("Total Jarak (km)", f"{total_dist:.2f}")
            st.metric("Waktu Komputasi (detik)", f"{runtime_bb:.6f}")

    if "bnb_path_idx" in st.session_state:
        st.subheader("🗺️ Visualisasi Rute B&B")
        valid_bnb_idx = [i for i in st.session_state["bnb_path_idx"] if i < len(st.session_state["order"])]
        route_order = [st.session_state["order"][i] for i in valid_bnb_idx]
        visualize_route(
            df_dest=destinasi,
            order=route_order,
            hotel_name=st.session_state["hotel_name"],
            route_type=route_type_choice,
            color="blue"
        )

        st.subheader("📋 Tabel Rute B&B")
        df_rute, _ = route_table_from_path(
            order=st.session_state["order"],
            path_idx_seq=valid_bnb_idx,
            cost_matrix=np.array(st.session_state["sub_matrix"])
        )
        st.dataframe(df_rute, use_container_width=True)

# ===============================
# ISI TAB 3 - NEAREST NEIGHBOR
# ===============================
with tab3:
    st.subheader("Algoritma Nearest Neighbor")

    if st.button("Run NN"):
        sub_matrix = st.session_state["sub_matrix"]
        order = st.session_state["order"]
        hotel_name = st.session_state["hotel_name"]

        nn_path_idx, total_nn, time_nn = run_nearest_neighbor(sub_matrix, order, hotel_name)

        st.session_state.nn_path_idx = nn_path_idx
        st.session_state.total_distance_nn = total_nn
        st.session_state.runtime_nn = time_nn
        st.metric("Total Jarak (km)", f"{total_nn:.2f}")
        st.metric("Waktu Komputasi (detik)", f"{time_nn:.6f}")

    if "nn_path_idx" in st.session_state:
        st.subheader("🗺️ Visualisasi Rute NN")
        valid_nn_idx = [i for i in st.session_state.nn_path_idx if i < len(st.session_state["order"])]
        route_order = [st.session_state["order"][i] for i in valid_nn_idx]
        visualize_route(
            df_dest=destinasi,
            order=route_order,
            hotel_name=st.session_state["hotel_name"],
            route_type=route_type_choice,
            color="green"
        )

        st.subheader("📋 Tabel Rute NN")
        df_rute, _ = route_table_from_path(
            order=st.session_state["order"],
            path_idx_seq=valid_nn_idx,
            cost_matrix=np.array(st.session_state["sub_matrix"])
        )
        st.dataframe(df_rute, use_container_width=True)

# ===============================
# ISI TAB 4 - ANT COLONY OPTIMIZATION
# ===============================
with tab4:
    st.subheader("Algoritma Ant Colony Optimization")

    if st.button("Run ACO"):
        sub_matrix = st.session_state["sub_matrix"]
        order = st.session_state["order"]
        hotel_name = st.session_state["hotel_name"]

        aco_path_idx, total_aco, time_aco, *_ = run_aco(sub_matrix, order, hotel_name)

        st.session_state.aco_path_idx = aco_path_idx
        st.session_state.total_distance_aco = total_aco
        st.session_state.runtime_aco = time_aco
        st.metric("Total Jarak (km)", f"{total_aco:.2f}")
        st.metric("Waktu Komputasi (detik)", f"{time_aco:.2f}")

    if "aco_path_idx" in st.session_state:
        st.subheader("🗺️ Visualisasi Rute ACO")
        valid_aco_idx = [i for i in st.session_state.aco_path_idx if i < len(st.session_state["order"])]
        route_order =  [st.session_state["order"][i] for i in valid_aco_idx]
        visualize_route(
            df_dest=destinasi,
            order=route_order,
            hotel_name=st.session_state["hotel_name"],
            route_type=route_type_choice,
            color="orange"
        )

        st.subheader("📋 Tabel Rute ACO")
        df_rute, _ = route_table_from_path(
            order=st.session_state["order"],
            path_idx_seq=valid_aco_idx,
            cost_matrix=np.array(st.session_state["sub_matrix"])
        )
        st.dataframe(df_rute, use_container_width=True)

# ===============================
# ISI TAB 5 - PERBANDINGAN
# ===============================
with tab5:
    st.subheader("📊 Perbandingan Hasil Metode Optimasi Rute")

    required_keys = ["total_distance_bb", "total_distance_nn", "total_distance_aco","runtime_bb", "runtime_nn", "runtime_aco"]
    if not all(k in st.session_state for k in required_keys):
        st.warning("⚠️ Jalankan NN, BNB, dan ACO terlebih dahulu.")
        st.stop()

    total_distance_bb = st.session_state.total_distance_bb
    total_distance_nn = st.session_state.total_distance_nn
    total_distance_aco = st.session_state.total_distance_aco
    runtime_bb = st.session_state.runtime_bb
    runtime_nn = st.session_state.runtime_nn
    runtime_aco = st.session_state.runtime_aco

    RE_bnb = 0.0
    RE_nn = abs(total_distance_nn - total_distance_bb) / total_distance_bb * 100
    RE_aco = abs(total_distance_aco - total_distance_bb) / total_distance_bb * 100

    df_perbandingan = pd.DataFrame({
        "Metode": [
            "Branch and Bound (BNB)",
            "Nearest Neighbor (NN)",
            "Ant Colony Optimization (ACO)"
        ],
        "Total Jarak Tempuh (km)": [
            total_distance_bb,
            total_distance_nn,
            total_distance_aco
        ],
        "Waktu Komputasi (detik)": [
            round(runtime_bb, 6),
            round(runtime_nn, 6),
            round(runtime_aco, 6)
        ],
        "Relative Error (%)": [
            f"{RE_bnb:.4f}",
            f"{RE_nn:.4f}",
            f"{RE_aco:.4f}"
        ]
    })

    st.dataframe(df_perbandingan, use_container_width=True)
    st.info("Relative Error dihitung terhadap hasil Branch and Bound sebagai baseline.")
st.markdown("---")
st.caption("Aplikasi dibuat untuk keperluan penelitian/skripsi.")

