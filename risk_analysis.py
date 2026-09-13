import pandas as pd

INPUT_FILE = "tumor_characteristics.csv"
OUTPUT_FILE = "tumor_risk_analysis.csv"


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

# Remove No Tumor because there is no meaningful
# tumor region to evaluate
df = df[df["tumor_class"] != "No Tumor"].copy()


# ==========================================
# AREA CATEGORY
# ==========================================

def classify_area(area):

    if area < 1.5:
        return "Small"

    elif area <= 4.5:
        return "Moderate"

    else:
        return "Large"


df["area_category"] = (
    df["relative_area_percent"]
    .apply(classify_area)
)


# ==========================================
# LOCATION CATEGORY
# ==========================================

def classify_location(row):

    if row["horizontal_location"] == "Center" and \
       row["vertical_location"] == "Middle":

        return "Central"

    return "Peripheral"


df["location_category"] = (
    df.apply(classify_location, axis=1)
)


# ==========================================
# RESEARCH RISK SCORE
# ==========================================

def calculate_risk(row):

    score = 0

    # Area contribution
    if row["area_category"] == "Moderate":
        score += 1

    elif row["area_category"] == "Large":
        score += 2

    # Location contribution
    if row["location_category"] == "Peripheral":
        score += 1

    # Convert score into research category
    if score <= 1:
        return "Lower"

    elif score == 2:
        return "Moderate"

    else:
        return "Higher"


df["research_risk_score"] = (
    df.apply(calculate_risk, axis=1)
)


# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print("\n===================================")
print("NEUROVISION RISK ANALYSIS")
print("===================================")

print("\nTotal tumor regions:", len(df))

print("\nArea categories:")
print(
    df["area_category"]
    .value_counts()
)

print("\nLocation categories:")
print(
    df["location_category"]
    .value_counts()
)

print("\nResearch risk categories:")
print(
    df["research_risk_score"]
    .value_counts()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print(
    "Risk categories are research-oriented "
    "and are NOT clinical diagnoses."
)