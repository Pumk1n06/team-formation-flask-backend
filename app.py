from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import random

app = Flask(__name__)
CORS(app)


def cluster_employees(data, algo, cluster_count, params):
    df = pd.DataFrame(data)

    # Flatten nested fields
    if "experience" in params:
        df["experience"] = df["experience"].apply(
            lambda exp: ", ".join([e["role"] for e in exp]) if isinstance(exp, list) else "")
    if "skills" in params:
        df["skills"] = df["skills"].apply(
            lambda skills: ", ".join([s["name"] for s in skills]) if isinstance(skills, list) else "")

    # Encode features
    features = df[params].astype(str)
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(features).toarray()

    if cluster_count > len(data):
        raise ValueError("Number of clusters cannot exceed number of candidates.")

    if algo == "KMeans":
        model = KMeans(n_clusters=cluster_count, random_state=42)
        labels = model.fit_predict(encoded_features)
    elif algo == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=cluster_count)
        labels = model.fit_predict(encoded_features)
    elif algo == "GaussianMixture":
        model = GaussianMixture(n_components=cluster_count, random_state=42)
        labels = model.fit_predict(encoded_features)
    else:
        raise ValueError("Unsupported clustering algorithm.")

    # Group candidates by cluster
    clustered = {}
    for idx, label in enumerate(labels):
        clustered.setdefault(label, []).append(data[idx])

    return list(clustered.values())


def split_clusters_into_teams(clustered_groups, team_size):
    final_teams = []

    for group in clustered_groups:
        random.shuffle(group)  # To distribute candidates more fairly
        teams = [group[i:i + team_size] for i in range(0, len(group), team_size)]
        final_teams.extend(teams)

    return final_teams


@app.route("/group", methods=["POST"])
def group():
    try:
        req = request.get_json()
        candidates = req.get("candidates", [])
        team_size = int(req.get("team_no", 4))  # This is team size
        params = req.get("params", [])
        algo = req.get("algo", "KMeans")
        cluster_count = int(req.get("clusters", 3))  # You can pass how many clusters you want

        if not candidates or not params:
            return jsonify({"success": False, "error": "Candidates and params are required"}), 400

        clustered = cluster_employees(candidates, algo, cluster_count, params)
        teams = split_clusters_into_teams(clustered, team_size)

        return jsonify({"success": True, "data": teams}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
