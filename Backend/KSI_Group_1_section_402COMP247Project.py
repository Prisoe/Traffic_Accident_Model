# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 17:19:34 2025

@author: marco favaretto
@student ID: 301186334
"""
# import sys
# import time
# import logging
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")  # Adds timestamps

cluster_labels = ["Etobicoke", "Central-Eastern", "North York", "Scarborough", "Downtown"]

# Grouping columns
categorical_cols = ["TRAFFCTL", "RDSFCOND", "IMPACTYPE", "VISIBILITY", "INVTYPE", "LIGHT", "ACCLASS"]

contextual_cols = [
    "CYCCOND",
    "CYCACT",
    "CYCLISTYPE",
    "PEDACT",
    "PEDTYPE",
    "PEDCOND",
    "DRIVACT",
    "DRIVCOND",
    "MANOEUVER",
    "ACCLOC",
    "INITDIR",
    "VEHTYPE",
]

location_cols = [
    "HOOD_158",
    "NEIGHBOURHOOD_158",
    "HOOD_140",
    "NEIGHBOURHOOD_140",
    "DIVISION",
    "OBJECTID",
    "LATITUDE",
    "LONGITUDE",
    "STREET1",
    "STREET2",
    "FATAL_NO",
    "x",
    "y",
]

yes_nan_col = [
    "ALCOHOL",
    "AUTOMOBILE",
    "CYCLIST",
    "EMERG_VEH",
    "MOTORCYCLE",
    "TRUCK",
    "TRSN_CITY_VEH",
    "PASSENGER",
    "PEDESTRIAN",
    "DISABILITY",
    "REDLIGHT",
    "SPEEDING",
    "AG_DRIV",
]


# Transformation function to impute ACCNUM missing values based on location, date, and time
def impute_accnum(df, distance_threshold=None):
    # Create a copy of the DataFrame
    df_filled = df.copy()

    # Replace empty strings with NaN and coerce ACCNUM to numeric
    df_filled["ACCNUM"] = df_filled["ACCNUM"].replace("", pd.NA)
    df_filled["ACCNUM"] = pd.to_numeric(df_filled["ACCNUM"], errors="coerce")

    # Determine the starting point for new ACCNUM values
    max_accnum = df_filled["ACCNUM"].max() if df_filled["ACCNUM"].notna().any() else 0
    new_accnum_start = int(max_accnum) + 1 if max_accnum is not np.nan else 1
    new_accnum_counter = new_accnum_start

    # Ensure YEAR column exists (for proper grouping)
    if "YEAR" not in df_filled.columns:
        df_filled["YEAR"] = pd.to_datetime(df_filled["DATE"]).dt.year

    if distance_threshold is not None:
        # Group by DATE, TIME, and YEAR for within-year processing
        group_cols = ["DATE", "TIME", "YEAR"]
        df_filled["Group_ID"] = -1  # Initialize Group_ID
        current_group_id = 0

        for _, group in df_filled.groupby(group_cols):
            if group[["LATITUDE", "LONGITUDE"]].isna().any(axis=1).all():
                continue

            coords = np.deg2rad(group[["LATITUDE", "LONGITUDE"]].values)
            tree = BallTree(coords, metric="haversine")

            indices = tree.query_radius(coords, r=distance_threshold, return_distance=False)

            group_indices = group.index
            visited = set()
            for i, neighbors in enumerate(indices):
                if group_indices[i] in visited:
                    continue
                for neighbor_idx in neighbors:
                    df_filled.loc[group_indices[neighbor_idx], "Group_ID"] = current_group_id
                    visited.add(group_indices[neighbor_idx])
                current_group_id += 1

        # Impute missing ACCNUM for each group
        for group_id, group in df_filled.groupby("Group_ID"):
            if group_id == -1:  # For rows not assigned to a group
                for idx in group.index:
                    if pd.isna(df_filled.loc[idx, "ACCNUM"]):
                        df_filled.loc[idx, "ACCNUM"] = new_accnum_counter
                        new_accnum_counter += 1
                continue

            if group["ACCNUM"].notna().any():
                filled_accnum = group["ACCNUM"].dropna().iloc[0]
                df_filled.loc[group.index, "ACCNUM"] = group["ACCNUM"].fillna(filled_accnum)
            else:
                df_filled.loc[group.index, "ACCNUM"] = new_accnum_counter
                new_accnum_counter += 1

        # Drop temporary Group_ID column
        df_filled = df_filled.drop(columns=["Group_ID"])
    else:
        # Group by LATITUDE, LONGITUDE, DATE, TIME, and YEAR when no distance_threshold is provided
        group_cols = ["LATITUDE", "LONGITUDE", "DATE", "TIME", "YEAR"]
        for _, group in df_filled.groupby(group_cols):
            if group["ACCNUM"].notna().any():
                filled_accnum = group["ACCNUM"].dropna().iloc[0]
                df_filled.loc[group.index, "ACCNUM"] = group["ACCNUM"].fillna(filled_accnum)
            else:
                df_filled.loc[group.index, "ACCNUM"] = new_accnum_counter
                new_accnum_counter += 1

    # Convert ACCNUM to integer
    df_filled["ACCNUM"] = df_filled["ACCNUM"].astype(int)
    return df_filled


# Function to create a ACCIDENT_ID to deal with duplicated ACCNUM
def process_accnum_data(df):
    # Step 1: Create a copy of the original DataFrame
    df_copy = df.copy()

    # Step 2: Ensure 'YEAR' column exists (derived from 'DATE' if necessary)
    if "YEAR" not in df_copy.columns and "DATE" in df_copy.columns:
        df_copy["YEAR"] = pd.to_datetime(df_copy["DATE"]).dt.year

    # Step 3: Create a new column for combined 'ACCNUM' and 'YEAR'
    df_copy["ACCIDENT_ID"] = df_copy["ACCNUM"].astype(str) + "_" + df_copy["YEAR"].astype(str)

    # Step 4: Handle duplicates within the same year
    # Here, we'll add a counter to each duplicate ACCIDENT_ID
    df_copy["DUPLICATE_COUNT"] = df_copy.groupby("ACCIDENT_ID").cumcount()

    # Return the processed DataFrame
    return df_copy


# Transformation function to clean the DATE column and extract only the date part
def clean_date(df):
    df_cleaned = df.copy()
    df_cleaned["DATE"] = pd.to_datetime(df_cleaned["DATE"]).dt.date
    return df_cleaned


# Transformation function to extract the hour from the TIME dividing by 100 and taking teh integer part
def extract_hour(df):
    # Check if TIME column exists
    if "TIME" in df.columns:
        try:
            # Extract hour from TIME (e.g., 1850 -> 18)
            df["HOUR"] = (df["TIME"] // 100).astype(int)
            # Ensure HOUR is between 0 and 23
            df["HOUR"] = df["HOUR"].clip(0, 23)
        except Exception as e:
            print(f"Error extracting hour from TIME column: {e}")
            return df
    elif "DATE" in df.columns:
        # Fallback to DATE column if TIME is not available
        try:
            df["HOUR"] = pd.to_datetime(df["DATE"]).dt.hour
        except Exception as e:
            print(f"Error extracting hour from DATE column: {e}")
            return df
    else:
        print("Error: Neither TIME nor DATE column found in DataFrame.")
        return df

    # Handle any NaN values in HOUR (fill with 0)
    df["HOUR"] = df["HOUR"].fillna(0).astype(int)
    return df


# Transformation function to impute missing values in a target column using KNN based on latitude and longitude
def fill_missing_location_with_knn(df, target_col, k=5):
    # Create a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Separate rows with known and missing values in the target column
    known = df_filled[df_filled[target_col].notna()]
    missing = df_filled[df_filled[target_col].isna()]

    # Return original DataFrame if no missing or known values exist
    if missing.empty or known.empty:
        print(f"No missing values or no known values to infer {target_col}.")
        return df_filled

    # Use 'LATITUDE' and 'LONGITUDE' for KNN
    X_known = known[["LATITUDE", "LONGITUDE"]].values
    X_missing = missing[["LATITUDE", "LONGITUDE"]].values

    # Fit Nearest Neighbors model on known coordinates
    nbrs = NearestNeighbors(n_neighbors=min(k, len(known)), algorithm="auto").fit(X_known)
    distances, indices = nbrs.kneighbors(X_missing)

    # Helper function to find the most common value among neighbors
    def most_common_value(neighbor_indices, known_values):
        values = [known_values.iloc[idx] for idx in neighbor_indices]
        return Counter(values).most_common(1)[0][0]

    # Fill missing values using KNN predictions
    for i, row in enumerate(missing.index):
        neighbor_indices = indices[i]
        predicted_value = most_common_value(neighbor_indices, known[target_col])
        df_filled.loc[row, target_col] = predicted_value

    return df_filled


# Transformation function to impute missing values in a target column using KNN based on latitude, longitude, date and time
def fill_missing_contextual_with_knn(df, target_col, k=5):
    # Create a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Ensure DATE is in the correct format
    df_filled["DATE"] = pd.to_datetime(df_filled["DATE"]).dt.date  # Ensure it's a `datetime.date`
    df_filled["DATE_ORDINAL"] = pd.to_datetime(df_filled["DATE"]).map(lambda x: x.toordinal())  # Convert to ordinal

    # Separate rows with known and missing values in the target column
    known = df_filled[df_filled[target_col].notna()]
    missing = df_filled[df_filled[target_col].isna()]

    # Return original DataFrame if no missing or known values exist
    if missing.empty or known.empty:
        print(f"No missing values or no known values to infer {target_col}.")
        return df_filled

    # Use 'LATITUDE', 'LONGITUDE', 'DATE_ORDINAL', and 'TIME' for KNN
    X_known = known[["LATITUDE", "LONGITUDE", "DATE_ORDINAL", "TIME"]].values
    X_missing = missing[["LATITUDE", "LONGITUDE", "DATE_ORDINAL", "TIME"]].values

    # Fit Nearest Neighbors model on known coordinates
    nbrs = NearestNeighbors(n_neighbors=min(k, len(known)), algorithm="auto").fit(X_known)
    distances, indices = nbrs.kneighbors(X_missing)

    # Helper function to find the most common value among neighbors
    def most_common_value(neighbor_indices, known_values):
        values = [known_values.iloc[idx] for idx in neighbor_indices]
        return Counter(values).most_common(1)[0][0]

    # Fill missing values using KNN predictions
    for i, row in enumerate(missing.index):
        neighbor_indices = indices[i]
        predicted_value = most_common_value(neighbor_indices, known[target_col])
        df_filled.loc[row, target_col] = predicted_value

    # Drop the temporary DATE_ORDINAL column
    df_filled = df_filled.drop(columns=["DATE_ORDINAL"])

    return df_filled


# Transformation function to impute missing VEHTYPE values within groups defined by ACCNUM
def impute_vehtype(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Group by 'ACCIDENT_ID' (ensures unique accident identification)
    for accident_id, group in df_filled.groupby("ACCIDENT_ID"):
        if group["VEHTYPE"].notna().any():
            # If at least one non-missing VEHTYPE exists, use the first one to fill
            filled_vehtype = group["VEHTYPE"].dropna().iloc[0]
            df_filled.loc[group.index, "VEHTYPE"] = group["VEHTYPE"].fillna(filled_vehtype)
        else:
            # If all VEHTYPE values are missing, set as "Unknown"
            df_filled.loc[group.index, "VEHTYPE"] = "Unknown"

    return df_filled


# Transformation function to convert Yes/No or NaN columns to binary (1 for "yes", 0 for "no" or NaN)
def yes_nan_to_binary(X):
    return X.apply(lambda col: (col.str.lower() == "yes").astype(int))


# Transformation function for ordinal mappings
def ordinal_mapping(df):
    df_transformed = df.copy()

    traffctl_map = {cat: idx for idx, cat in enumerate(df["TRAFFCTL"].dropna().unique())}
    rdscond_map = {cat: idx for idx, cat in enumerate(df["RDSFCOND"].dropna().unique())}
    impactype_map = {cat: idx for idx, cat in enumerate(df["IMPACTYPE"].dropna().unique())}
    visibility_map = {cat: idx for idx, cat in enumerate(df["VISIBILITY"].dropna().unique())}
    light_map = {cat: idx for idx, cat in enumerate(df["LIGHT"].dropna().unique())}
    acclass_map = {cat: idx for idx, cat in enumerate(df["ACCLASS"].dropna().unique())}
    road_class_map = {cat: idx for idx, cat in enumerate(df["ROAD_CLASS"].dropna().unique())}
    invtype_map = {cat: idx for idx, cat in enumerate(df["INVTYPE"].dropna().unique())}

    fatal_key = next(k for k, v in acclass_map.items() if "Fatal" in k)
    df_transformed.loc[(df["ACCLASS"] == fatal_key) & (df["INJURY"].isna()), "INJURY"] = "None"
    df_transformed["INJURY"] = df_transformed["INJURY"].fillna("None")
    injury_map = {cat: idx for idx, cat in enumerate(df_transformed["INJURY"].unique())}
    if "None" in injury_map and injury_map["None"] != 0:
        injury_map = {"None": 0, **{k: v + 1 for k, v in injury_map.items() if k != "None"}}

    for col in contextual_cols:
        if col == "VEHTYPE":
            df_transformed[col] = df_transformed[col].fillna("Unknown")
        elif col == "DRIVCOND":
            df_transformed[col] = df_transformed[col].fillna("Unknown")
        else:
            df_transformed[col] = df_transformed[col].fillna("N/A")

    cyccond_map = {cat: idx for idx, cat in enumerate(df_transformed["CYCCOND"].unique())}
    if "N/A" in cyccond_map and cyccond_map["N/A"] != 0:
        cyccond_map = {"N/A": 0, **{k: v + 1 for k, v in cyccond_map.items() if k != "N/A"}}
    cycact_map = {cat: idx for idx, cat in enumerate(df_transformed["CYCACT"].unique())}
    if "N/A" in cycact_map and cycact_map["N/A"] != 0:
        cycact_map = {"N/A": 0, **{k: v + 1 for k, v in cycact_map.items() if k != "N/A"}}
    cyclistype_map = {cat: idx for idx, cat in enumerate(df_transformed["CYCLISTYPE"].unique())}
    if "N/A" in cyclistype_map and cyclistype_map["N/A"] != 0:
        cyclistype_map = {"N/A": 0, **{k: v + 1 for k, v in cyclistype_map.items() if k != "N/A"}}
    pedact_map = {cat: idx for idx, cat in enumerate(df_transformed["PEDACT"].unique())}
    if "N/A" in pedact_map and pedact_map["N/A"] != 0:
        pedact_map = {"N/A": 0, **{k: v + 1 for k, v in pedact_map.items() if k != "N/A"}}
    pedtype_map = {cat: idx for idx, cat in enumerate(df_transformed["PEDTYPE"].unique())}
    if "N/A" in pedtype_map and pedtype_map["N/A"] != 0:
        pedtype_map = {"N/A": 0, **{k: v + 1 for k, v in pedtype_map.items() if k != "N/A"}}
    pedcond_map = {cat: idx for idx, cat in enumerate(df_transformed["PEDCOND"].unique())}
    if "N/A" in pedcond_map and pedcond_map["N/A"] != 0:
        pedcond_map = {"N/A": 0, **{k: v + 1 for k, v in pedcond_map.items() if k != "N/A"}}
    drivact_map = {cat: idx for idx, cat in enumerate(df_transformed["DRIVACT"].unique())}
    if "N/A" in drivact_map and drivact_map["N/A"] != 0:
        drivact_map = {"N/A": 0, **{k: v + 1 for k, v in drivact_map.items() if k != "N/A"}}
    drivcond_map = {cat: idx for idx, cat in enumerate(df_transformed["DRIVCOND"].unique())}
    if "Unknown" in drivcond_map and drivcond_map["Unknown"] != 4:
        drivcond_map = {k: v if k != "Unknown" else 4 for k, v in drivcond_map.items()}
        used_indices = set(drivcond_map.values())
        next_idx = 0
        for k in drivcond_map.keys():
            if k != "Unknown" and drivcond_map[k] in used_indices:
                while next_idx in used_indices:
                    next_idx += 1
                drivcond_map[k] = next_idx
                used_indices.add(next_idx)
    manoeuver_map = {cat: idx for idx, cat in enumerate(df_transformed["MANOEUVER"].unique())}
    if "N/A" in manoeuver_map and manoeuver_map["N/A"] != 0:
        manoeuver_map = {"N/A": 0, **{k: v + 1 for k, v in manoeuver_map.items() if k != "N/A"}}
    accloc_map = {cat: idx for idx, cat in enumerate(df_transformed["ACCLOC"].unique())}
    if "N/A" in accloc_map and accloc_map["N/A"] != 0:
        accloc_map = {"N/A": 0, **{k: v + 1 for k, v in accloc_map.items() if k != "N/A"}}
    initdir_map = {cat: idx for idx, cat in enumerate(df_transformed["INITDIR"].unique())}
    if "N/A" in initdir_map and initdir_map["N/A"] != 0:
        initdir_map = {"N/A": 0, **{k: v + 1 for k, v in initdir_map.items() if k != "N/A"}}
    vehtype_map = {cat: idx for idx, cat in enumerate(df_transformed["VEHTYPE"].unique())}
    if "Unknown" in vehtype_map and vehtype_map["Unknown"] != 0:
        vehtype_map = {"Unknown": 0, **{k: v + 1 for k, v in vehtype_map.items() if k != "Unknown"}}

    df_transformed["TRAFFCTL"] = df_transformed["TRAFFCTL"].map(traffctl_map)
    df_transformed["RDSFCOND"] = df_transformed["RDSFCOND"].map(rdscond_map)
    df_transformed["IMPACTYPE"] = df_transformed["IMPACTYPE"].map(impactype_map)
    df_transformed["VISIBILITY"] = df_transformed["VISIBILITY"].map(visibility_map)
    df_transformed["LIGHT"] = df_transformed["LIGHT"].map(light_map)
    df_transformed["ACCLASS"] = df_transformed["ACCLASS"].map(acclass_map)
    df_transformed["ROAD_CLASS"] = df_transformed["ROAD_CLASS"].map(road_class_map)
    df_transformed["INVTYPE"] = df_transformed["INVTYPE"].map(invtype_map)
    df_transformed["INJURY"] = df_transformed["INJURY"].map(injury_map)
    df_transformed["CYCCOND"] = df_transformed["CYCCOND"].map(cyccond_map)
    df_transformed["CYCACT"] = df_transformed["CYCACT"].map(cycact_map)
    df_transformed["CYCLISTYPE"] = df_transformed["CYCLISTYPE"].map(cyclistype_map)
    df_transformed["PEDACT"] = df_transformed["PEDACT"].map(pedact_map)
    df_transformed["PEDTYPE"] = df_transformed["PEDTYPE"].map(pedtype_map)
    df_transformed["PEDCOND"] = df_transformed["PEDCOND"].map(pedcond_map)
    df_transformed["DRIVACT"] = df_transformed["DRIVACT"].map(drivact_map)
    df_transformed["DRIVCOND"] = df_transformed["DRIVCOND"].map(drivcond_map)
    df_transformed["MANOEUVER"] = df_transformed["MANOEUVER"].map(manoeuver_map)
    df_transformed["ACCLOC"] = df_transformed["ACCLOC"].map(accloc_map)
    df_transformed["INITDIR"] = df_transformed["INITDIR"].map(initdir_map)
    df_transformed["VEHTYPE"] = df_transformed["VEHTYPE"].map(vehtype_map)

    mappings = {
        "TRAFFCTL": traffctl_map,
        "RDSFCOND": rdscond_map,
        "IMPACTYPE": impactype_map,
        "VISIBILITY": visibility_map,
        "LIGHT": light_map,
        "ACCLASS": acclass_map,
        "ROAD_CLASS": road_class_map,
        "INVTYPE": invtype_map,
        "INJURY": injury_map,
        "CYCCOND": cyccond_map,
        "CYCACT": cycact_map,
        "CYCLISTYPE": cyclistype_map,
        "PEDACT": pedact_map,
        "PEDTYPE": pedtype_map,
        "PEDCOND": pedcond_map,
        "DRIVACT": drivact_map,
        "DRIVCOND": drivcond_map,
        "MANOEUVER": manoeuver_map,
        "ACCLOC": accloc_map,
        "INITDIR": initdir_map,
        "VEHTYPE": vehtype_map,
    }
    for col, mapping in mappings.items():
        print(f"\nDynamic {col} Mapping:")
        print("Number : Category")
        print("----------------")
        for category, number in mapping.items():
            print(f"{number} : {category}")

    return df_transformed, mappings


# Helper function to visualize total missing values and its percentage
def missing_data_summary(df, sort_by_percentage=True):
    missing_data = df.isnull().sum()
    missing_percentage = np.round(((missing_data / len(df)) * 100), 2)
    column_types = df.dtypes
    missing_info = pd.DataFrame(
        {"Missing Values": missing_data, "Missing Percentage": missing_percentage, "Column Type": column_types}
    )
    if sort_by_percentage:
        missing_info = missing_info.sort_values(by="Missing Percentage", ascending=False)
    return missing_info


# Helper function to summarize the total of accidents by class
def summarize_accident(df_unique):
    accclass_counts = df_unique.groupby("ACCNUM")["ACCLASS"].first().value_counts().sort_index()
    accclass_labels = {0: "Property Damage Only", 1: "Non-Fatal Injury", 2: "Fatal"}
    print("\nTotal Accidents by Type:")
    for acc_class, count in accclass_counts.items():
        label = accclass_labels[acc_class]
        print(f"Total {label}: {count}")
    total_by_type = accclass_counts.sum()
    total_unique_accidents = df_unique["ACCNUM"].nunique()
    print(f"\nSum of Totals by Type: {total_by_type}")
    print(f"Total Unique Accidents: {total_unique_accidents}")


# region PLOTS
def plot_cluster_locations(df):
    n_clusters = 5
    # Ensure location data exists
    location_data = df[["LATITUDE", "LONGITUDE"]].dropna()
    if len(location_data) == 0:
        print("No valid location data for clustering.")
        return

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[location_data.index, "Cluster"] = kmeans.fit_predict(location_data)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        cluster_data = df[df["Cluster"] == cluster]
        label = (
            f"Cluster {cluster} ({cluster_labels[cluster]})"
            if cluster_labels and len(cluster_labels) > cluster
            else f"Cluster {cluster}"
        )
        plt.scatter(cluster_data["LONGITUDE"], cluster_data["LATITUDE"], label=label, alpha=0.1, s=10)
    plt.title("Accident Locations by Cluster")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Clusters")
    plt.grid(True)
    plt.show()


def plot_accident_categories(df):
    # Ensure required columns exist
    if "CYCLIST" not in df.columns or "PEDESTRIAN" not in df.columns:
        print("Required columns ('CYCLIST' and 'PEDESTRIAN') are missing.")
        return

    # Create the "Category" column based on CYCLIST and PEDESTRIAN flags
    df["Category"] = "Vehicle-Only"
    df.loc[df["CYCLIST"] == 1, "Category"] = "Cyclist"
    df.loc[df["PEDESTRIAN"] == 1, "Category"] = "Pedestrian"

    # Set up the plot
    plt.figure(figsize=(10, 6))
    categories = ["Cyclist", "Pedestrian", "Vehicle-Only"]
    colors = {"Cyclist": "green", "Pedestrian": "red", "Vehicle-Only": "blue"}

    for category in categories:
        category_data = df[df["Category"] == category]
        plt.scatter(
            category_data["LONGITUDE"],
            category_data["LATITUDE"],
            label=category,
            color=colors[category],
            alpha=0.1,  # Increased alpha for better visibility
            s=15,  # Slightly larger point size for clarity
        )

    # Add titles, labels, and legend
    plt.title("Accident Concentrations by Category (Cyclist, Pedestrian, Vehicle-Only)", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(title="Accident Category", fontsize=10)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.show()


def plot_combined_distribution_by_cluster(df, acclass_map, cluster_labels):
    # Map accident severity labels (ACCLASS) using the provided mapping
    acclass_labels = {v: k for k, v in acclass_map.items()}
    df["ACCLASS_LABEL"] = df["ACCLASS"].map(acclass_labels)

    # Create an accident category column (Cyclist, Pedestrian, Vehicle-Only)
    df["Category"] = "Vehicle-Only"
    df.loc[df["CYCLIST"] == 1, "Category"] = "Cyclist"
    df.loc[df["PEDESTRIAN"] == 1, "Category"] = "Pedestrian"

    # Map cluster labels (for meaningful names instead of numbers)
    df["Cluster_Label"] = df["Cluster"].map(lambda x: cluster_labels[int(x)] if x != -1 else "Unknown")

    # Compute percentage distribution and counts for ACCLASS by cluster
    pivot_acclass = pd.crosstab(df["Cluster_Label"], df["ACCLASS_LABEL"], normalize="index") * 100
    counts_acclass = pd.crosstab(df["Cluster_Label"], df["ACCLASS_LABEL"])
    annot_acclass = pivot_acclass.round(1).astype(str) + "\n(" + counts_acclass.astype(str) + ")"

    # Compute percentage distribution and counts for categories by cluster
    pivot_category = pd.crosstab(df["Cluster_Label"], df["Category"], normalize="index") * 100
    counts_category = pd.crosstab(df["Cluster_Label"], df["Category"])
    annot_category = pivot_category.round(1).astype(str) + "\n(" + counts_category.astype(str) + ")"

    # Create the subplots for the two heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)  # Slightly larger width for clarity

    # Heatmap for Accident Severity (ACCLASS)
    sns.heatmap(
        pivot_acclass,
        annot=annot_acclass,
        fmt="s",
        cmap="YlOrRd",
        cbar_kws={"label": "Percentage (%)"},
        vmin=0,
        vmax=100,
        ax=axes[0],
    )
    axes[0].set_title("Accident Severity (ACCLASS)", fontsize=14)
    axes[0].set_xlabel("Severity", fontsize=12)
    axes[0].set_ylabel("Cluster", fontsize=12)
    axes[0].tick_params(axis="y", rotation=0)  # Horizontal y-axis labels

    # Heatmap for Accident Category
    sns.heatmap(
        pivot_category,
        annot=annot_category,
        fmt="s",
        cmap="YlOrRd",
        cbar_kws={"label": "Percentage (%)"},
        vmin=0,
        vmax=100,
        ax=axes[1],
    )
    axes[1].set_title("Accident Category", fontsize=14)
    axes[1].set_xlabel("Category", fontsize=12)
    axes[1].set_ylabel("", fontsize=12)  # Avoid duplicate y-axis label
    axes[1].tick_params(axis="y", rotation=0)  # Horizontal y-axis labels

    # Add a main title and layout adjustments
    plt.suptitle("Percentage Distribution of Accident Characteristics by Cluster", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_acclass_by_time_and_injury(df, acclass_map, injury_map):
    # Function to bin time into time-of-day categories
    def bin_time(time):
        if 6 <= time < 12:
            return "Morning"
        elif 12 <= time < 18:
            return "Afternoon"
        elif 18 <= time < 22:
            return "Evening"
        else:
            return "Night"

    # Ensure the TIME column exists and bin it
    if "TIME" not in df.columns:
        print("The column 'TIME' is missing in the dataset.")
        return
    df["TIME_BIN"] = df["TIME"].apply(bin_time)

    # Map ACCLASS and INJURY labels using the provided mappings
    acclass_labels = {v: k for k, v in acclass_map.items()}  # Reverse mapping
    injury_labels = {v: k for k, v in injury_map.items()}  # Reverse mapping
    df["ACCLASS_LABEL"] = df["ACCLASS"].map(acclass_labels)
    df["INJURY_LABEL"] = df["INJURY"].map(injury_labels)

    # Group data by time of day, ACCLASS, and INJURY and compute counts
    grouped = df.groupby(["TIME_BIN", "ACCLASS_LABEL", "INJURY_LABEL"]).size().reset_index(name="Count")
    grouped["Count"] = grouped["Count"].astype(int)

    # Pivot the grouped data for a heatmap-friendly format
    pivot = grouped.pivot_table(
        index=["TIME_BIN", "ACCLASS_LABEL"], columns="INJURY_LABEL", values="Count", fill_value=0
    ).astype(int)

    # Calculate percentages for each row
    totals = pivot.sum(axis=1)  # Sum of all injuries per time bin and ACCLASS
    pivot_percentage = pivot.div(totals, axis=0) * 100  # Convert to percentages

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_percentage,
        annot=True,
        fmt=".1f",
        cmap="rocket_r",
        cbar_kws={"label": "Percentage of Accidents (%)"},
        linewidths=0.5,  # Add gridlines for clarity
    )
    plt.title("Accident Severity (ACCLASS) by Time of Day and Injury Severity (Percentages)", fontsize=14)
    plt.xlabel("Injury Severity", fontsize=12)
    plt.ylabel("Time of Day and Accident Severity (ACCLASS)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)  # Horizontal y-axis labels for readability
    plt.tight_layout()
    plt.show()


def plot_acclass_by_conditions(df, acclass_map, light_map, rdsfcond_map, visibility_map):
    # Reverse mappings for labels
    acclass_labels = {v: k for k, v in acclass_map.items()}
    light_labels = {v: k for k, v in light_map.items()}
    rdsfcond_labels = {v: k for k, v in rdsfcond_map.items()}
    visibility_labels = {v: k for k, v in visibility_map.items()}

    # Map labels to the DataFrame
    df["ACCLASS_LABEL"] = df["ACCLASS"].map(acclass_labels)
    df["LIGHT_LABEL"] = df["LIGHT"].map(light_labels)
    df["RDSFCOND_LABEL"] = df["RDSFCOND"].map(rdsfcond_labels)
    df["VISIBILITY_LABEL"] = df["VISIBILITY"].map(visibility_labels)

    # Combine rare conditions into broader categories
    df["LIGHT_LABEL"] = df["LIGHT_LABEL"].replace(
        {"Dawn": "Twilight", "Dawn, artificial": "Twilight", "Dusk": "Twilight", "Dusk, artificial": "Twilight"}
    )
    rare_rdsfcond = ["Ice", "Loose Sand or Gravel", "Loose Snow", "Packed Snow", "Slush", "Spilled liquid"]
    df["RDSFCOND_LABEL"] = df["RDSFCOND_LABEL"].apply(lambda x: "Other" if x in rare_rdsfcond else x)
    rare_visibility = ["Drifting Snow", "Fog, Mist, Smoke, Dust", "Freezing Rain", "Snow", "Strong wind"]
    df["VISIBILITY_LABEL"] = df["VISIBILITY_LABEL"].apply(lambda x: "Other" if x in rare_visibility else x)

    # Compute contingency tables and perform chi-squared tests
    light_counts = pd.crosstab(df["LIGHT_LABEL"], df["ACCLASS_LABEL"])
    rdsfcond_counts = pd.crosstab(df["RDSFCOND_LABEL"], df["ACCLASS_LABEL"])
    visibility_counts = pd.crosstab(df["VISIBILITY_LABEL"], df["ACCLASS_LABEL"])

    for condition, counts in [
        ("LIGHT", light_counts),
        ("RDSFCOND", rdsfcond_counts),
        ("VISIBILITY", visibility_counts),
    ]:
        chi2, p, _, _ = chi2_contingency(counts)
        print(f"Chi-squared test for {condition} vs ACCLASS: p-value = {p:.4f}")

    all_acclass_categories = sorted(df["ACCLASS_LABEL"].unique())

    # Generate percentage and counts pivot tables
    light_pivot = pd.crosstab(df["LIGHT_LABEL"], df["ACCLASS_LABEL"], normalize="columns") * 100
    light_pivot = light_pivot.reindex(columns=all_acclass_categories, fill_value=0.0)
    light_counts_pivot = pd.crosstab(df["LIGHT_LABEL"], df["ACCLASS_LABEL"])
    light_counts_pivot = light_counts_pivot.reindex(columns=all_acclass_categories, fill_value=0)
    light_pivot.index = [f"LIGHT - {idx}" for idx in light_pivot.index]
    light_counts_pivot.index = [f"LIGHT - {idx}" for idx in light_counts_pivot.index]

    rdsfcond_pivot = pd.crosstab(df["RDSFCOND_LABEL"], df["ACCLASS_LABEL"], normalize="columns") * 100
    rdsfcond_pivot = rdsfcond_pivot.reindex(columns=all_acclass_categories, fill_value=0.0)
    rdsfcond_counts_pivot = pd.crosstab(df["RDSFCOND_LABEL"], df["ACCLASS_LABEL"])
    rdsfcond_counts_pivot = rdsfcond_counts_pivot.reindex(columns=all_acclass_categories, fill_value=0)
    rdsfcond_pivot.index = [f"RDSFCOND - {idx}" for idx in rdsfcond_pivot.index]
    rdsfcond_counts_pivot.index = [f"RDSFCOND - {idx}" for idx in rdsfcond_counts_pivot.index]

    visibility_pivot = pd.crosstab(df["VISIBILITY_LABEL"], df["ACCLASS_LABEL"], normalize="columns") * 100
    visibility_pivot = visibility_pivot.reindex(columns=all_acclass_categories, fill_value=0.0)
    visibility_counts_pivot = pd.crosstab(df["VISIBILITY_LABEL"], df["ACCLASS_LABEL"])
    visibility_counts_pivot = visibility_counts_pivot.reindex(columns=all_acclass_categories, fill_value=0)
    visibility_pivot.index = [f"VISIBILITY - {idx}" for idx in visibility_pivot.index]
    visibility_counts_pivot.index = [f"VISIBILITY - {idx}" for idx in visibility_counts_pivot.index]

    # Combine all conditions for the heatmap
    combined_pivot = pd.concat([light_pivot, rdsfcond_pivot, visibility_pivot], axis=0)
    combined_counts = pd.concat([light_counts_pivot, rdsfcond_counts_pivot, visibility_counts_pivot], axis=0)

    # Annotate the heatmap with both percentages and counts
    annot = combined_pivot.round(1).astype(str) + "\n(" + combined_counts.astype(str) + ")"

    # Plot the combined heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        combined_pivot,
        annot=annot,
        fmt="s",
        cmap="rocket_r",
        cbar_kws={"label": "Percentage of Accidents (%)"},
        vmin=0,
        vmax=100,
    )
    plt.title("Percentage Distribution of Accident Severity (ACCLASS) by Conditions", fontsize=14)
    plt.xlabel("Accident Severity (ACCLASS)", fontsize=12)
    plt.ylabel("Condition and Category", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)  # Horizontal y-axis labels for better readability
    plt.tight_layout()
    plt.show()


def plot_distribution_of_accidents_by_hour_bar_with_trend(df):
    """
    Plot a bar chart showing the hourly distribution of accidents with a trend line.

    Parameters:
    df (pd.DataFrame): Unified DataFrame (e.g., working_df) containing all relevant columns.
    """
    # Verify the existence and validity of the HOUR column
    if "HOUR" not in df.columns or df["HOUR"].isna().all():
        print("HOUR column not found or all values are NaN. Please ensure HOUR is computed correctly.")
        return

    # Debug: Check the unique values in the HOUR column
    print("Unique values in HOUR column before adjustment:", sorted(df["HOUR"].dropna().unique()))

    # Convert 24:00 to 00:00 if present
    df = df.copy()
    df["HOUR"] = df["HOUR"].replace(24, 0)

    # Debug: Check the unique values in the HOUR column after adjustment
    print("Unique values in HOUR column after adjustment:", sorted(df["HOUR"].dropna().unique()))

    # Calculate the total number of accidents
    total_accidents = df["ACCIDENT_ID"].nunique()  # Updated to use ACCIDENT_ID
    if total_accidents == 0:
        print("No accidents found in the dataset.")
        return

    # Count accidents per hour
    hourly_counts = df.groupby("HOUR")["ACCIDENT_ID"].nunique()  # Updated to use ACCIDENT_ID

    # Convert counts to percentages
    hourly_percentages = (hourly_counts / total_accidents * 100).reindex(range(24), fill_value=0)

    # Debug: Print the counts and percentages
    print("Hourly Counts:")
    print(hourly_counts)
    print("Hourly Percentages:")
    print(hourly_percentages)

    # Plot the bar plot
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x=range(24), y=hourly_percentages, color="lightblue", edgecolor="black", alpha=0.7)

    # Add a moving average trend line
    window_size = 3
    moving_avg = hourly_percentages.rolling(window=window_size, center=True).mean()
    ax.plot(range(24), moving_avg, color="navy", linewidth=2.5, label=f"{window_size}-Hour Moving Average")

    # Add percentage labels on top of each bar
    for i, percentage in enumerate(hourly_percentages):
        if percentage > 0:
            ax.text(
                i,
                percentage + 0.2,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                fontweight="bold",
            )

    # Highlight rush hours (7:00-9:00 for morning rush, 16:00-18:00 for evening rush)
    ax.axvspan(7, 9, color="red", alpha=0.15, label="Morning Rush (7:00-9:00)")
    ax.axvspan(16, 18, color="orange", alpha=0.15, label="Evening Rush (16:00-18:00)")

    # Add an annotation for the peak hour
    peak_hour = hourly_percentages.idxmax()
    peak_value = hourly_percentages[peak_hour]
    ax.annotate(
        f"Peak at {peak_hour}:00\n{peak_value:.1f}%",
        xy=(peak_hour, peak_value),
        xytext=(peak_hour + 2, peak_value + 1),
        arrowprops=dict(facecolor="black", shrink=0.05),
        fontsize=10,
        color="black",
        fontweight="bold",
    )

    # Add grid lines for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Customize the plot
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=10)
    ax.set_ylim(0, max(hourly_percentages) + 2)
    ax.set_xlabel("Hour of the Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Accidents (%)", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Accidents by Hour of the Day (Percentage)", fontsize=14, fontweight="bold", pad=20)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # Print total accidents for debugging purposes
    print("Total Accidents by Hour:")
    print(hourly_counts)
    print(f"Total Number of Accidents: {total_accidents}")


def plot_injury_severity_distribution(df, injury_map):
    """
    Create a pie chart showing the distribution of injury severity categories with detailed annotations.

    Parameters:
    df (pd.DataFrame): DataFrame containing the INJURY column (numerical values after mapping).
    injury_map (dict): Mapping dictionary to convert numerical INJURY values back to categorical labels.
    """
    # Verify required column
    if "INJURY" not in df.columns:
        print("INJURY column not found in the dataset.")
        return

    # Debug: Check the unique values in INJURY
    print("Unique values in INJURY column before plotting:")
    print(sorted(df["INJURY"].dropna().unique()))

    # Reverse the mapping to get the original categorical labels
    reverse_injury_map = {v: k for k, v in injury_map.items()}
    df = df.copy()
    df["INJURY_LABEL"] = df["INJURY"].map(reverse_injury_map)

    # Debug: Check the unique labels after mapping
    print("Unique INJURY labels after reverse mapping:")
    print(sorted(df["INJURY_LABEL"].dropna().unique()))

    # Calculate the counts and percentages for each INJURY category
    injury_counts = df["INJURY_LABEL"].value_counts()
    total_accidents = injury_counts.sum()
    injury_percentages = injury_counts / total_accidents * 100

    # Debug: Print the counts and percentages
    print("Injury Severity Distribution:")
    print(injury_counts)
    print("\nInjury Severity Percentages:")
    print(injury_percentages.round(1))

    # Check if injury_counts is empty
    if injury_counts.empty:
        print("No data to plot for INJURY distribution.")
        return

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("Set2", len(injury_counts))
    explode = [0.05] * len(injury_counts)  # Slightly explode all slices for emphasis

    # Plot the pie chart
    wedges, texts, autotexts = plt.pie(
        injury_counts,
        labels=injury_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )

    # Customize the pie chart
    plt.title("Distribution of Injury Severity", fontsize=14, fontweight="bold", pad=20)

    # Add a legend with counts and percentages
    legend_labels = [
        f"{category}: {count} ({percent:.1f}%)"
        for category, count, percent in zip(injury_counts.index, injury_counts, injury_percentages)
    ]
    plt.legend(
        wedges,
        legend_labels,
        title="Injury Severity (Count & Percentage)",
        loc="upper left",
        bbox_to_anchor=(-0.4, 1),  # Move the legend to the left of the pie chart
        fontsize=9,  # Reduce font size for compactness
        title_fontsize=10,
    )

    # Add an annotation for the most common injury severity category
    most_common_category = injury_counts.idxmax()
    most_common_percentage = injury_percentages[most_common_category]
    plt.annotate(
        f"Most Common: {most_common_category}\n({most_common_percentage:.1f}%)",
        xy=(0, 0),
        xytext=(-0.1, -0.5),
        ha="center",
        va="center",
        fontsize=12,
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"),
    )

    # Ensure the pie chart is circular
    plt.axis("equal")
    # Adjust the layout to create more space around the pie chart
    plt.tight_layout()
    plt.show()


def plot_acclass_distribution(df, acclass_map):
    """
    Create a simple bar plot showing the distribution of ACCLASS categories.

    Parameters:
    df (pd.DataFrame): DataFrame containing the ACCLASS column (numerical values after mapping).
    acclass_map (dict): Mapping dictionary to convert numerical ACCLASS values back to categorical labels.
    """
    # Verify required column
    if "ACCLASS" not in df.columns:
        print("ACCLASS column not found in the dataset.")
        return

    # Debug: Check the unique values in ACCLASS
    print("Unique values in ACCLASS column before plotting:")
    print(sorted(df["ACCLASS"].dropna().unique()))

    # Reverse the mapping to get the original categorical labels
    reverse_acclass_map = {v: k for k, v in acclass_map.items()}
    df = df.copy()
    df["ACCLASS_LABEL"] = df["ACCLASS"].map(reverse_acclass_map)

    # Debug: Check the unique labels after mapping
    print("Unique ACCLASS labels after reverse mapping:")
    print(sorted(df["ACCLASS_LABEL"].dropna().unique()))

    # Calculate the counts and percentages for each ACCLASS category
    acclass_counts = df["ACCLASS_LABEL"].value_counts()
    total_accidents = acclass_counts.sum()
    acclass_percentages = (acclass_counts / total_accidents * 100).round(1)

    # Debug: Print the counts and percentages
    print("ACCLASS Distribution:")
    print(acclass_counts)
    print("\nACCLASS Percentages:")
    print(acclass_percentages)

    # Check if acclass_counts is empty
    if acclass_counts.empty:
        print("No data to plot for ACCLASS distribution.")
        return

    # Create the bar plot
    plt.figure(figsize=(12, 6))  # Increased figure size for better readability
    sns.barplot(
        x=acclass_counts.values,
        y=acclass_counts.index,
        palette="Set2",
        edgecolor="black",
        alpha=0.8,  # Slight transparency for visual appeal
    )

    # Customize the plot
    plt.title("Distribution of Accident Classifications (ACCLASS)", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Number of Accidents", fontsize=14, fontweight="bold")
    plt.ylabel("Accident Classifications (ACCLASS)", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add count and percentage labels on the bars
    for i, (count, percentage) in enumerate(zip(acclass_counts, acclass_percentages)):
        plt.text(
            count,
            i,
            f"{count} ({percentage:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # Adjust layout manually to ensure readability
    plt.tight_layout()
    plt.show()  # Explicitly show the plot


def plot_cooccurrence_matrix(df):
    """
    Plot a co-occurrence matrix of conditions and factors in traffic accidents.

    Parameters:
    df (pd.DataFrame): Unified DataFrame (e.g., working_df) containing all relevant columns.
    """
    # Define vehicle columns (excluding CYCLIST and PEDESTRIAN)
    vehicle_columns = ["AUTOMOBILE", "MOTORCYCLE", "TRUCK", "TRSN_CITY_VEH", "EMERG_VEH"]

    # Define variables, replacing AUTOMOBILE with VEHICLE
    variables = [
        "RDSFCOND",
        "VISIBILITY",
        "LIGHT",
        "SPEEDING",
        "AG_DRIV",
        "ALCOHOL",
        "CYCLIST",
        "PEDESTRIAN",
        "VEHICLE",
    ]

    categories = {}
    for var in variables:
        if var in ["SPEEDING", "AG_DRIV", "ALCOHOL", "CYCLIST", "PEDESTRIAN", "VEHICLE"]:
            categories[var] = [var]  # Simplified label (e.g., "ALCOHOL" instead of "ALCOHOL - ALCOHOL")
        else:
            if var == "RDSFCOND":
                categories[var] = ["Dry", "Wet", "Other"]
            elif var == "VISIBILITY":
                categories[var] = ["Clear", "Rain"]
            elif var == "LIGHT":
                categories[var] = ["Daylight", "Dark", "Twilight"]

    all_labels = []
    for var, cats in categories.items():
        all_labels.extend(
            [
                (
                    f"{var} - {cat}"
                    if var not in ["SPEEDING", "AG_DRIV", "ALCOHOL", "CYCLIST", "PEDESTRIAN", "VEHICLE"]
                    else cat
                )
                for cat in cats
            ]
        )

    n = len(all_labels)
    cooccurrence_matrix = np.zeros((n, n))
    total_accidents = df["ACCIDENT_ID"].nunique()  # Updated to use ACCIDENT_ID

    for i, label_i in enumerate(all_labels):
        # Handle labels for binary factors (no " - " in label)
        if label_i in ["SPEEDING", "AG_DRIV", "ALCOHOL", "CYCLIST", "PEDESTRIAN", "VEHICLE"]:
            var_i = label_i
            cat_i = label_i
        else:
            var_i, cat_i = label_i.split(" - ", 1)

        for j, label_j in enumerate(all_labels):
            if label_j in ["SPEEDING", "AG_DRIV", "ALCOHOL", "CYCLIST", "PEDESTRIAN", "VEHICLE"]:
                var_j = label_j
                cat_j = label_j
            else:
                var_j, cat_j = label_j.split(" - ", 1)

            if i == j:
                cooccurrence_matrix[i, j] = 100.0
                continue

            # Build conditions
            if var_i in ["SPEEDING", "AG_DRIV", "ALCOHOL", "CYCLIST", "PEDESTRIAN"]:
                condition_i = df[var_i] == 1
            elif var_i == "VEHICLE":
                condition_i = df[vehicle_columns].eq(1).any(axis=1)  # Any vehicle type involved
            else:
                if var_i == "LIGHT" and cat_i == "Dark":
                    condition_i = df[f"{var_i}_LABEL"].isin(["Dark", "Dark, artificial"])
                elif var_i == "LIGHT" and cat_i == "Twilight":
                    condition_i = df[f"{var_i}_LABEL"].isin(
                        ["Twilight", "Dawn", "Dawn, artificial", "Dusk", "Dusk, artificial"]
                    )
                else:
                    condition_i = df[f"{var_i}_LABEL"] == cat_i

            if var_j in ["SPEEDING", "AG_DRIV", "ALCOHOL", "CYCLIST", "PEDESTRIAN"]:
                condition_j = df[var_j] == 1
            elif var_j == "VEHICLE":
                condition_j = df[vehicle_columns].eq(1).any(axis=1)  # Any vehicle type involved
            else:
                if var_j == "LIGHT" and cat_j == "Dark":
                    condition_j = df[f"{var_j}_LABEL"].isin(["Dark", "Dark, artificial"])
                elif var_j == "LIGHT" and cat_j == "Twilight":
                    condition_j = df[f"{var_j}_LABEL"].isin(
                        ["Twilight", "Dawn", "Dawn, artificial", "Dusk", "Dusk, artificial"]
                    )
                else:
                    condition_j = df[f"{var_j}_LABEL"] == cat_j

            # Calculate co-occurrence count
            accnums_i = set(df[condition_i]["ACCIDENT_ID"])
            accnums_j = set(df[condition_j]["ACCIDENT_ID"])
            cooccurrence_count = len(accnums_i.intersection(accnums_j))
            cooccurrence_percentage = (cooccurrence_count / total_accidents) * 100
            cooccurrence_matrix[i, j] = cooccurrence_percentage

    cooccurrence_df = pd.DataFrame(cooccurrence_matrix, index=all_labels, columns=all_labels)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cooccurrence_df,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Percentage of Accidents (%)"},
        vmin=0,
        vmax=100,
    )
    plt.title("Co-occurrence Matrix of Conditions and Factors (Percentage of Accidents)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_accident_trend_by_year(df):
    """
    Plot a trend of accidents by year, showing yearly counts and a trend line.

    Parameters:
    df (pd.DataFrame): Unified DataFrame (e.g., working_df) containing all relevant columns.
    """
    # Verify DATE column
    if "DATE" not in df.columns:
        print("DATE column not found. Please ensure the dataset includes a DATE column.")
        return

    # Verify ACCIDENT_ID column
    if "ACCIDENT_ID" not in df.columns:
        print("ACCIDENT_ID column not found. Please ensure the dataset includes an ACCIDENT_ID column.")
        return

    # Extract the year from the DATE column
    df = df.copy()
    df["YEAR"] = pd.to_datetime(df["DATE"]).dt.year

    # Count the number of accidents per year
    yearly_counts = df.groupby("YEAR")["ACCIDENT_ID"].nunique().reset_index(name="Count")  # Updated to ACCIDENT_ID

    # Ensure all years from 2006 to 2023 are included
    all_years = pd.DataFrame({"YEAR": range(2006, 2024)})
    yearly_counts = all_years.merge(yearly_counts, on="YEAR", how="left").fillna(0)
    yearly_counts["Count"] = yearly_counts["Count"].astype(int)

    # Debug: Print the counts for each year
    print("Yearly Accident Counts:")
    print(yearly_counts)

    # Calculate year-over-year percentage change to identify unusual spikes
    yearly_counts["Percent Change"] = yearly_counts["Count"].pct_change() * 100
    print("\nYear-over-Year Percentage Change:")
    print(yearly_counts[["YEAR", "Count", "Percent Change"]])

    # Fit a linear trend line
    z = np.polyfit(yearly_counts["YEAR"], yearly_counts["Count"], 1)
    p = np.poly1d(z)
    trend_line = p(yearly_counts["YEAR"])

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_counts["YEAR"], yearly_counts["Count"], marker="o", color="green", label="Accidents per Year")
    plt.plot(yearly_counts["YEAR"], trend_line, linestyle="--", color="blue", label="Trend Line")

    # Add labels for each data point
    for i, count in enumerate(yearly_counts["Count"]):
        if count > 0:  # Only label years with accidents
            plt.text(yearly_counts["YEAR"].iloc[i], count + 10, str(int(count)), ha="center", va="bottom", fontsize=9)

    # Add a vertical line for COVID-19 lockdowns
    plt.axvline(x=2020, color="red", linestyle="--", label="COVID-19 Lockdowns")

    # Customize the plot
    plt.title("Accident Trend by Year (2006–2023)", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Year", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Accidents", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(yearly_counts["YEAR"], rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_yearly_factors(df, mappings):
    """
    Analyze yearly factors contributing to accidents and visualize their normalized trends over the years.

    Parameters:
    df (pd.DataFrame): Unified DataFrame (e.g., working_df) containing all relevant columns.
    mappings (dict): Dictionary of mappings for categorical variables (if needed).
    """
    # Extract the YEAR from the DATE column
    df = df.copy()
    df["YEAR"] = pd.to_datetime(df["DATE"]).dt.year

    # Filter data to include only relevant years (2006–2023)
    all_years = range(2006, 2024)
    df = df[df["YEAR"].isin(all_years)]

    # Define vehicle columns and check for involvement
    vehicle_columns = ["AUTOMOBILE", "MOTORCYCLE", "TRUCK", "TRSN_CITY_VEH", "EMERG_VEH"]
    vehicle_involvement = df.groupby("ACCIDENT_ID")[vehicle_columns].max().eq(1).any(axis=1).astype(int)
    no_vehicle_accidents = vehicle_involvement[vehicle_involvement == 0].index
    print(f"\nNumber of accidents with no vehicle involved: {len(no_vehicle_accidents)}")

    # Subset vehicle-related accidents if there are accidents without vehicles
    use_vehicle_subset = len(no_vehicle_accidents) > 0
    if use_vehicle_subset:
        print("Some accidents do not involve a vehicle. Subsetting to vehicle-related accidents.")
        vehicle_accidents = df[df["ACCIDENT_ID"].isin(vehicle_involvement[vehicle_involvement == 1].index)]
    else:
        print("All accidents involve a vehicle. No subsetting needed.")
        vehicle_accidents = df

    # Analyze major entities involved
    major_entities = ["PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK", "TRSN_CITY_VEH", "EMERG_VEH"]
    major_entities_to_plot = []
    total_unique_accidents = df["ACCIDENT_ID"].nunique()

    for factor in major_entities:
        accident_ids_with_factor = set(df[df[factor] == 1]["ACCIDENT_ID"])
        factor_accidents = len(accident_ids_with_factor)
        factor_percentage = (factor_accidents / total_unique_accidents) * 100
        print(f"\nAnalysis of {factor}: {factor_accidents} accidents ({factor_percentage:.2f}%)")
        if factor_percentage < 1:
            print(f"{factor} is too rare. Skipping.")
        else:
            major_entities_to_plot.append(factor)

    # Define colors for the entities
    colors = {
        "AUTOMOBILE": "blue",
        "PEDESTRIAN": "red",
        "CYCLIST": "green",
        "MOTORCYCLE": "yellow",
        "TRUCK": "orange",
        "TRSN_CITY_VEH": "purple",
        "EMERG_VEH": "pink",
    }

    # Prepare the plot
    plt.figure(figsize=(14, 8))
    years = list(all_years)
    bottom = np.zeros(len(years))
    entity_data = {}

    for factor in major_entities_to_plot:
        # Use the subset of vehicle-related accidents for pedestrian and cyclist involvement if applicable
        df_to_use = vehicle_accidents if factor in ["PEDESTRIAN", "CYCLIST"] and use_vehicle_subset else df
        accident_ids_with_factor = set(df_to_use[df_to_use[factor] == 1]["ACCIDENT_ID"])
        yearly_counts = (
            df[df["ACCIDENT_ID"].isin(accident_ids_with_factor)].groupby("YEAR").size().reindex(all_years, fill_value=0)
        )
        counts_array = yearly_counts.values
        if counts_array.max() != counts_array.min():
            normalized_counts = (counts_array - counts_array.min()) / (counts_array.max() - counts_array.min()) * 100
        else:
            normalized_counts = counts_array * 0  # Avoid division by zero for constant values
        entity_data[factor] = normalized_counts

    # Plot the entities' trends
    for factor in major_entities_to_plot:
        bars = plt.bar(
            years, entity_data[factor], bottom=bottom, label=factor, color=colors.get(factor, "gray"), edgecolor="white"
        )
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 5:  # Label bars with significant heights
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[i] + height / 2,
                    f"{entity_data[factor][i]:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )
        bottom += entity_data[factor]

    # Customize the plot
    plt.title("Normalized Trend of Major Entities Involved in Accidents (2006–2023)", fontsize=14, fontweight="bold")
    plt.xlabel("Year", fontsize=12, fontweight="bold")
    plt.ylabel("Normalized Involvement (%)", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.xticks(years, rotation=45, fontsize=10)
    plt.ylim(0, max(np.sum([entity_data[factor] for factor in major_entities_to_plot], axis=0)) * 1.1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_accident_reasons_heatmap(df, mappings):
    """
    Plot a heatmap showing the distribution of accident reasons across years.

    Parameters:
    df (pd.DataFrame): Unified DataFrame (e.g., working_df) containing all relevant columns.
    mappings (dict): Dictionary containing mappings for categorical variables.
    """
    # Extract the YEAR from the DATE column
    df = df.copy()
    df["YEAR"] = pd.to_datetime(df["DATE"]).dt.year

    # Filter data to include only relevant years (2006–2023)
    all_years = range(2006, 2024)
    df = df[df["YEAR"].isin(all_years)]

    # Define driver and environmental factors
    driver_factors = ["AG_DRIV", "ALCOHOL", "DISABILITY", "REDLIGHT", "SPEEDING"]
    env_factors = ["LIGHT", "RDSFCOND", "VISIBILITY"]

    # Map environmental labels
    light_labels = {v: k for k, v in mappings["LIGHT"].items()}
    rdsfcond_labels = {v: k for k, v in mappings["RDSFCOND"].items()}
    visibility_labels = {v: k for k, v in mappings["VISIBILITY"].items()}

    # Map categorical labels for LIGHT, RDSFCOND, and VISIBILITY
    df["LIGHT_LABEL"] = df["LIGHT"].map(light_labels)
    df["RDSFCOND_LABEL"] = df["RDSFCOND"].map(rdsfcond_labels)
    df["VISIBILITY_LABEL"] = df["VISIBILITY"].map(visibility_labels)

    # Combine rare conditions into broader categories for simplicity
    df["LIGHT_LABEL"] = df["LIGHT_LABEL"].replace(
        {"Dawn": "Twilight", "Dawn, artificial": "Twilight", "Dusk": "Twilight", "Dusk, artificial": "Twilight"}
    )
    rare_rdsfcond = ["Ice", "Loose Sand or Gravel", "Loose Snow", "Packed Snow", "Slush", "Spilled liquid"]
    df["RDSFCOND_LABEL"] = df["RDSFCOND_LABEL"].apply(lambda x: "Other" if x in rare_rdsfcond else x)
    rare_visibility = ["Drifting Snow", "Fog, Mist, Smoke, Dust", "Freezing Rain", "Snow", "Strong wind"]
    df["VISIBILITY_LABEL"] = df["VISIBILITY_LABEL"].apply(lambda x: "Other" if x in rare_visibility else x)

    # Define categories and labels
    categories = {
        "AG_DRIV": ["AG_DRIV"],
        "ALCOHOL": ["ALCOHOL"],
        "DISABILITY": ["DISABILITY"],
        "REDLIGHT": ["REDLIGHT"],
        "SPEEDING": ["SPEEDING"],
        "LIGHT": ["Dark", "Twilight"],
        "RDSFCOND": ["Wet", "Other"],
        "VISIBILITY": ["Rain", "Other"],
    }

    # Generate labels for the heatmap
    all_labels = []
    for var, cats in categories.items():
        all_labels.extend([f"{var} - {cat}" for cat in cats])

    # Modify labels to remove redundancy
    modified_labels = [
        var if var == cat else f"{var} - {cat}" for var, cat in [label.split(" - ", 1) for label in all_labels]
    ]

    # Prepare heatmap data
    heatmap_data = pd.DataFrame(index=all_years, columns=modified_labels)

    # Calculate totals per year
    total_drivers_per_year = df.groupby("YEAR").size().reindex(all_years, fill_value=0)
    total_accidents_per_year = df.groupby("YEAR")["ACCIDENT_ID"].nunique().reindex(all_years, fill_value=0)

    # Populate heatmap data
    for label in all_labels:
        var, cat = label.split(" - ", 1)
        modified_label = var if var == cat else label
        if var in driver_factors:
            yearly_counts = df[df[var] == 1].groupby("YEAR").size()
            pivot = (yearly_counts / total_drivers_per_year * 100).reindex(all_years, fill_value=0)
        else:
            yearly_counts = df[df[f"{var}_LABEL"] == cat].groupby("YEAR").size()
            pivot = (yearly_counts / total_accidents_per_year * 100).reindex(all_years, fill_value=0)
        heatmap_data[modified_label] = pivot

    # Normalize the data for each year
    for year in all_years:
        year_total = heatmap_data.loc[year].sum()
        if year_total > 0:
            heatmap_data.loc[year] = (heatmap_data.loc[year] / year_total * 100).round(1)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data.T,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Normalized Percentage (%)"},
    )
    plt.title("Heatmap of Accident Reasons Over Years (2006–2023)", fontsize=14, fontweight="bold")
    plt.xlabel("Year", fontsize=12, fontweight="bold")
    plt.ylabel("Reason", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Plots the ROC curve for the given model and test data.

    Parameters:
        model (object): Trained model with a `predict_proba` method.
        X_test (DataFrame): Test features.
        y_test (array-like): True labels.
        model_name (str): Name of the model being evaluated.

    Returns:
        None: Displays the ROC curve plot.
    """
    # Get the unique classes in y_test
    unique_classes = sorted(list(set(y_test)))

    # Define class labels dynamically based on unique classes
    class_labels = []
    if 0 in unique_classes:
        class_labels.append("Non-Fatal Injury (0)")
    if 1 in unique_classes:
        class_labels.append("Fatal (1)")
    if 2 in unique_classes:
        class_labels.append("Property Damage O (2)")

    plt.figure(figsize=(10, 6))

    # Plot ROC curves for the classes present in the data
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, i], pos_label=unique_classes[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Plots the confusion matrix as percentages for the given true and predicted values.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Name of the model being evaluated.

    Returns:
        None: Displays the confusion matrix plot.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert values to percentages
    row_sums = cm.sum(axis=1, keepdims=True)  # Sum of each row
    cm_percentage = cm / row_sums * 100  # Convert to percentages

    # Get unique class labels from y_test
    unique_classes = sorted(list(set(y_test)))

    # Define display labels dynamically based on unique classes
    display_labels = []
    if 0 in unique_classes:
        display_labels.append("Non-Fatal Injury (0)")
    if 1 in unique_classes:
        display_labels.append("Fatal (1)")
    if 2 in unique_classes:
        display_labels.append("Property Damage O (2)")

    # Create a larger figure to avoid layout issues
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjusted size to make it larger
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=display_labels)

    # Plot the confusion matrix
    disp.plot(cmap="viridis", ax=ax, xticks_rotation=45, values_format=".1f")  # Show percentages with 1 decimal

    # Adjust layout dynamically
    plt.title(f"Confusion Matrix (Percentages) for {model_name}", fontsize=14)
    plt.tight_layout(pad=2.0)  # Add padding to avoid clipping elements
    plt.grid(False)
    plt.show()


# endregion


def check_correlation(data, target_column="ACCLASS"):
    # Map TIME_BIN to numerical values if TIME_BIN exists
    if "TIME_BIN" in data.columns:
        time_bin_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
        data["TIME_BIN_NUM"] = data["TIME_BIN"].map(time_bin_mapping)

    # Extract MONTH and DAY_OF_WEEK from DATE
    if "DATE" in data.columns:
        data["MONTH"] = pd.to_datetime(data["DATE"]).dt.month
        data["DAY_OF_WEEK"] = pd.to_datetime(data["DATE"]).dt.weekday

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=["number"])

    # Compute correlation matrix
    correlation_matrix = numeric_data.corr()

    # Extract correlations with the target column
    target_corr = correlation_matrix[target_column].sort_values(ascending=False)

    # Display the correlations in descending order
    print("\nCorrelations with Target Variable (ACCLASS):")
    print(target_corr)

    return target_corr


def scale_lat_long(df_or_values, scaler=None):
    """
    Scales LATITUDE and LONGITUDE using MinMaxScaler.

    Parameters:
        df_or_values (DataFrame or ndarray): The DataFrame or array containing LATITUDE and LONGITUDE.
        scaler (MinMaxScaler, optional): The pre-fitted scaler. If None, fit a new one.

    Returns:
        Scaled DataFrame or ndarray, and the fitted scaler.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df_or_values)  # Fit scaler on the input data

    # Transform the input
    scaled_values = scaler.transform(df_or_values)

    return scaled_values, scaler


def prepare_data(df, oversampler="SMOTE"):
    """
    Prepares the dataset by performing feature scaling,
    splitting data into train/test, and applying oversampling techniques.

    Parameters:
        df (DataFrame): The input dataset.
        oversampler (str): The oversampling technique to use ("SMOTE" or "ADASYN").

    Returns:
        X_train_resampled, X_test, y_train_resampled, y_test
    """
    # Define columns to exclude
    columns_to_exclude = [
        "ACCIDENT_ID",
        "ACCLASS_LABEL",
        "ACCNUM",
        "AG_DRIV",
        "AUTOMOBILE",
        "Category",
        "Cluster",
        "Cluster_Label",
        "CYCACT",
        "CYCCOND",
        "CYCLIST",
        "CYCLISTYPE",
        "DATE",
        "DISABILITY",
        "DISTRICT",
        "DIVISION",
        "DRIVACT",
        "DUPLICATE_COUNT",
        "EMERG_VEH",
        "FATAL_NO",
        "HOOD_140",
        "HOOD_158",
        "HOUR",
        "IMPACTYPE",
        "INDEX",
        "INITDIR",
        "INJURY_LABEL",
        "INVAGE",
        "LIGHT",
        "LIGHT_LABEL",
        "MANOEUVER",
        "NEIGHBOURHOOD_140",
        "NEIGHBOURHOOD_158",
        "OBJECTID",
        "OFFSET",
        "RDSFCOND_LABEL",
        "REDLIGHT",
        "STREET1",
        "STREET2",
        "TIME",
        "TIME_BIN",
        "TRAFFCTL",
        "VISIBILITY_LABEL",
        "x",
        "y",
    ]

    # Normalize location-based features
    # scaler = MinMaxScaler()
    # df[["LATITUDE", "LONGITUDE"]] = scaler.fit_transform(df[["LATITUDE", "LONGITUDE"]])
    # Scale LATITUDE and LONGITUDE
    df[["LATITUDE", "LONGITUDE"]], scaler = scale_lat_long(df[["LATITUDE", "LONGITUDE"]])

    # Save the scaler for use in the API
    joblib.dump(scaler, "lat_long_scaler.pkl")

    # Define target variable and features
    y = df["ACCLASS"]
    X = df.drop(columns=["ACCLASS"] + columns_to_exclude)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the chosen oversampling technique
    if oversampler == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    elif oversampler == "ADASYN":
        adasyn = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    elif oversampler == "SMOTEENN":
        smote_enn = SMOTEENN(random_state=42)
        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    else:
        raise ValueError("Invalid oversampler. Choose 'SMOTE', 'ADASYN', or 'SMOTEENN'.")

    return X_train_resampled, X_test, y_train_resampled, y_test


# region TRAINING MODELS
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier using RandomizedSearchCV for hyperparameter tuning.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (array-like): Training labels.
        X_test (DataFrame): Test features.
        y_test (array-like): Test labels.

    Returns:
        best_model: The best Random Forest model found by RandomizedSearchCV.
        y_pred: Predicted labels for the test set.
    """

    # Randomized parameter grid
    param_distributions = {
        "n_estimators": np.linspace(100, 500, 10, dtype=int).tolist(),  # Sample a smaller range
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    # Perform RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(n_jobs=-1, random_state=42),
        param_distributions=param_distributions,
        n_iter=50,  # Number of random combinations to try
        scoring="f1_weighted",  # Weighted F1 score for imbalanced datasets
        cv=3,  # Reduced cross-validation folds for speed
        verbose=3,
        random_state=42,
        n_jobs=1,
    )
    print("Starting RandomizedSearchCV...")

    randomized_search.fit(X_train, y_train)
    print("RandomizedSearchCV completed!")

    # Best model and parameters
    best_model = randomized_search.best_estimator_
    print("\nBest Hyperparameters:", randomized_search.best_params_)
    print(f"\nBest F1-Weighted Score: {randomized_search.best_score_:.4f}")

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_model, y_pred


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier using RandomizedSearchCV for hyperparameter tuning.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (array-like): Training labels.
        X_test (DataFrame): Test features.
        y_test (array-like): Test labels.

    Returns:
        best_model: The best XGBoost model found by RandomizedSearchCV.
        y_pred: Predicted labels for the test set.
    """

    # Define the hyperparameter search space
    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5, 1],
    }

    # Perform RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric="logloss"),
        param_distributions=param_distributions,
        n_iter=50,  # Number of random combinations to try
        scoring="f1_weighted",  # Weighted F1 score for imbalanced datasets
        cv=3,  # Reduced cross-validation folds for speed
        verbose=2,
        random_state=42,
        n_jobs=1,
    )

    print("Starting RandomizedSearchCV for XGBoost...")

    randomized_search.fit(X_train, y_train)
    print("RandomizedSearchCV completed!")

    # Best model and parameters
    best_model = randomized_search.best_estimator_
    print("\nBest Hyperparameters:", randomized_search.best_params_)
    print(f"\nBest F1-Weighted Score: {randomized_search.best_score_:.4f}")

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_model, y_pred


def train_lightgbm(X_train, y_train, X_test, y_test, base_class_weights={0: 1, 1: 5, 2: 20}):
    """
    Train and evaluate the LightGBM model.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (array-like): Training labels.
        X_test (DataFrame): Test features.
        y_test (array-like): Test labels.
        base_class_weights (dict): Base class weights for the labels.

    Returns:
        model: Trained LightGBM model.
        y_pred: Predicted labels for the test set.
    """
    try:
        # Dynamically adjust class weights based on unique classes in y_train
        unique_classes = sorted(list(set(y_train)))
        adjusted_class_weight = {cls: base_class_weights[cls] for cls in unique_classes}

        # Train LightGBM model
        model = LGBMClassifier(class_weight=adjusted_class_weight, random_state=42, device="gpu")
        model.fit(X_train, y_train)

        # Predict probabilities for AUC-ROC
        try:
            y_prob = model.predict_proba(X_test)
        except AttributeError:
            y_prob = None
            print("LightGBM does not support predict_proba. Probabilities skipped.")

        # Predict class labels for evaluation
        y_pred = model.predict(X_test)

        # Classification Report
        print("\nLightGBM Classification Report:")
        print(classification_report(y_test, y_pred))

        # Return model, predictions, and probabilities (if available)
        return model, y_pred, y_prob

    except Exception as e:
        print(f"Error in training LightGBM: {e}")
        return None, None, None


# endregion


# region EVAULATE AUC-ROC
def evaluate_auc(model, X_test, y_test):
    """
    Evaluate the AUC-ROC for a given model, handling LightGBM and other edge cases.

    Parameters:
        model: Trained machine learning model.
        X_test (DataFrame): Test feature data.
        y_test (array-like): Test labels.

    Returns:
        None: Prints the AUC-ROC score or relevant error messages.
    """
    try:
        # Ensure y_test is a 1D array
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)  # Convert one-hot to label encoding

        # Check if model has predict_proba
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "predict"):  # Handle LightGBM when predict_proba is unavailable
            print("Warning: predict_proba is unavailable. Using predict for AUC-ROC (may be inaccurate).")
            y_score = model.predict(X_test)  # Fall back to predictions
            y_score = np.expand_dims(y_score, axis=1)  # Convert to 2D array for compatibility
        else:
            raise AttributeError("Model does not support predict or predict_proba.")

        # Handle binary and multi-class scenarios
        if y_score.shape[1] == 1:  # Binary classification case
            y_score = y_score.flatten()

        roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr")
        print(f"AUC-ROC Score: {roc_auc:.4f}")

    except AttributeError as e:
        print(f"Error: {e}. Skipping AUC-ROC evaluation.")
    except ValueError as e:
        print(f"Value Error in AUC-ROC evaluation: {e}")
    except Exception as e:
        print(f"Unexpected error during AUC-ROC evaluation: {e}")


# endregion

# region MAIN
if __name__ == "__main__":
    # Step 1: Load the dataset
    df = pd.read_csv("TOTAL_KSI_6386614326836635957.csv")

    # Step 2: Display initial missing data summary
    print("Initial Data Missing Summary:")
    print(missing_data_summary(df))

    # Step 3: Clean the DATE column
    print("Cleaning DATE column...")
    working_df = clean_date(df)

    # Optional Step: Extract hour from time if needed
    working_df = extract_hour(working_df)

    # Step 4: Impute missing ACCNUM values
    print("Imputing ACCNUM...")
    working_df = impute_accnum(working_df, distance_threshold=None)
    print("ACCNUM Imputation Completed.")
    print(missing_data_summary(working_df))

    # Step 5: Process ACCNUM to create ACCIDENT_ID and handle duplicates
    print("Processing ACCNUM to create ACCIDENT_ID...")
    working_df = process_accnum_data(working_df)
    print("ACCNUM Processing Completed.")

    # Step 6: KNN Imputation for location-based columns (ROAD_CLASS and DISTRICT)
    print("Applying KNN Imputation for location-based columns...")
    working_df = fill_missing_location_with_knn(working_df, target_col="ROAD_CLASS", k=5)
    working_df = fill_missing_location_with_knn(working_df, target_col="DISTRICT", k=5)
    print("KNN Imputation for location-based columns Completed.")
    print(missing_data_summary(working_df))

    # Step 7: KNN Imputation for contextual columns
    print("Applying KNN Imputation for contextual columns...")
    for col in categorical_cols:
        working_df = fill_missing_contextual_with_knn(working_df, target_col=col, k=5)
    print("KNN Imputation for contextual columns Completed.")
    print(missing_data_summary(working_df))

    # Step 8: Impute VEHTYPE values
    print("Imputing VEHTYPE...")
    working_df = impute_vehtype(working_df)
    print("VEHTYPE Imputation Completed.")
    print(missing_data_summary(working_df))

    # Step 9: Binary transformation for Yes/NaN columns
    print("Transforming Yes/No columns to binary...")
    working_df[yes_nan_col] = yes_nan_to_binary(working_df[yes_nan_col])
    print("Binary Transformation Completed.")

    # Step 10: Mode imputation for categorical columns
    print("Performing mode imputation for categorical columns...")
    imputer = SimpleImputer(strategy="most_frequent")
    working_df[categorical_cols] = imputer.fit_transform(working_df[categorical_cols])
    print("Mode Imputation Completed.")

    # Step 11: Ordinal mapping for contextual columns
    print("Applying ordinal mapping...")
    working_df, mappings = ordinal_mapping(working_df)
    print("Ordinal Mapping Completed.")

    # Step 12: Final missing data summary
    print("Final Data Missing Summary:")
    print(missing_data_summary(working_df))

    # Step 13: Validate if any contextual columns are missing
    missing_columns = [col for col in categorical_cols if col not in working_df.columns]
    print(f"Missing columns: {missing_columns if missing_columns else 'None'}")

    print("Execution completed successfully!")

    plot_cluster_locations(working_df)
    plot_accident_categories(working_df)
    plot_combined_distribution_by_cluster(working_df, mappings["ACCLASS"], cluster_labels)
    plot_acclass_by_time_and_injury(working_df, mappings["ACCLASS"], mappings["INJURY"])
    plot_acclass_by_conditions(
        working_df, mappings["ACCLASS"], mappings["LIGHT"], mappings["RDSFCOND"], mappings["VISIBILITY"]
    )
    plot_distribution_of_accidents_by_hour_bar_with_trend(working_df)
    plot_injury_severity_distribution(working_df, mappings["INJURY"])
    plot_acclass_distribution(working_df, mappings["ACCLASS"])
    plot_cooccurrence_matrix(working_df)
    plot_accident_trend_by_year(working_df)
    plot_yearly_factors(working_df, mappings)
    plot_accident_reasons_heatmap(working_df, mappings)

    correlation_results_with = check_correlation(working_df, target_column="ACCLASS")

    X_train_resampled, X_test, y_train_resampled, y_test = prepare_data(working_df, oversampler="SMOTE")

    # Train and evaluate models
    print("\nTraining models...")

    # Train Random Forest
    rf_model, rf_pred = train_random_forest(X_train_resampled, y_train_resampled, X_test, y_test)
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")
    plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
    # Evaluate AUC-ROC for Random Forest
    evaluate_auc(rf_model, X_test, y_test)

    # print("Final model parameters used in training:")
    # print(rf_model.get_params())

    # Train XGBoost
    xgb_model, xgb_pred = train_xgboost(X_train_resampled, y_train_resampled, X_test, y_test)
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
    plot_roc_curve(xgb_model, X_test, y_test, "XGBoost")
    # Evaluate AUC-ROC for XGBoost
    evaluate_auc(xgb_model, X_test, y_test)

    # print("Final model parameters used in training:")
    # print(xgb_model.get_params())

    # Serialize and Deserialize the Model
    with open("model.pkl", "wb") as file:
        pickle.dump(xgb_model, file)
    print("✅ Model saved successfully as 'xgb_model.pkl'")
# endregion
