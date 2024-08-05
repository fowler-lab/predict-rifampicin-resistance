import pandas as pd
import MDAnalysis as mda
from scipy.stats import norm
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sbmlcore
import json
from scipy.stats import sem
from matplotlib.patches import Rectangle


def map_mut2pdb(df, pdb, outfile, phenotypes=False):
    """
    Maps mutation positions onto a PDB structure and writes the result
    to temperature factors in a new PDB file.

    Parameters:
    - df (DataFrame): DataFrame with columns ['resid', 'segid'] indicating mutation sites.
    - pdb (str): Path to the input PDB file.
    - outfile (str): Path to the output PDB file where the results will be saved.
    """

    # Load the PDB and select only the protein atoms
    universe = mda.Universe(pdb)
    protein = universe.select_atoms("protein")
    temp_pdb = "./data/temp/proteinOnly.pdb"
    protein.write(temp_pdb)

    # Re-load the protein-only pdb
    u = mda.Universe(temp_pdb)

    # Extract chain identifiers and residue IDs from the PDB
    chains = [str(atom.segment)[9] for atom in u.atoms]
    resids = [atom.resid for atom in u.atoms]
    pdb_df = pd.DataFrame({"segid": chains, "resid": resids})

    if phenotypes:
        df_unique = (
            df.groupby("resid")
            .apply(
                lambda g: g.loc[g["phenotype"] == 1] if (g["phenotype"] == 1).any() else g.iloc[[0]]
            )
            .reset_index(drop=True)
        )
    else:
        # Aggregate the mutation DataFrame to ensure unique (SEGID, RESID) pairs
        df_unique = df.drop_duplicates(subset=["segid", "resid"])

    # Ensure df_unique has unique rows for (segid, resid)
    df_unique = df_unique.drop_duplicates(subset=["segid", "resid"])

    # Map mutations onto atoms
    merged_df = pdb_df.merge(df_unique, on=["segid", "resid"], how="left", indicator=True)
    merged_df["mutated"] = (merged_df["_merge"] == "both").astype(int)

    # Write mutations to temperature factors
    if phenotypes:
        merged_df["phenotype"] = merged_df["phenotype"].fillna(0)
        tempfactors = merged_df["phenotype"].tolist()
    else:
        tempfactors = merged_df["mutated"].tolist()

    # Ensure the lengths match before adding tempfactors
    if len(tempfactors) != len(u.atoms):
        raise ValueError(f"Length of tempfactors ({len(tempfactors)}) does not match number of atoms ({len(u.atoms)}).")

    u.add_TopologyAttr("tempfactors", values=tempfactors)
    u.atoms.write(outfile)

    os.remove(temp_pdb)


def filter_multiple_phenos(group):
    """
    If a sample contains more than one phenotype,
    keep the resistant phenotype (preferably with MIC) if there is one.

    Parameters:
    group (pd.DataFrame): A dataframe containing sample data with phenotypes.

    Returns:
    pd.DataFrame: A filtered dataframe prioritizing resistant phenotypes.
    """
    if len(group) == 1:
        return group

    # Prioritize rows with 'R' phenotype
    prioritized_group = (
        group[group["PHENOTYPE"] == "R"] if "R" in group["PHENOTYPE"].values else group
    )

    # Check for rows with METHOD_MIC values
    with_mic = prioritized_group.dropna(subset=["METHOD_MIC"])
    return with_mic if not with_mic.empty else prioritized_group.iloc[0:1]


def wilson(count, n, confidence=0.95):
    """Calculates wilson confidence intervals for supplied counts"""
    if n == 0:
        return (0, 0)

    proportion = count / n
    z = norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z**2 / n)
    centre_adjusted_prob = proportion + (z**2 / (2 * n))
    adjusted_sd = z * np.sqrt((proportion * (1 - proportion) / n) + (z**2 / (4 * n**2)))

    lower = (centre_adjusted_prob - adjusted_sd) / denom
    upper = (centre_adjusted_prob + adjusted_sd) / denom

    return pd.Series(
        [proportion, lower, upper], index=["proportion", "lower_bound", "upper_bound"]
    )


def manual_bootstrap(model, X, y, threshold=0.5, n_iterations=100):
    """
    Perform bootstrapping to evaluate model performance with various metrics.

    Parameters:
    ----------
    model : sklearn estimator
        The trained model with a `predict_proba` method.
    X : array-like
        Feature data for predictions.
    y : array-like
        True labels corresponding to X.
    threshold : float, optional, default=0.5
        Threshold for converting predicted probabilities to binary predictions.
    n_iterations : int, optional, default=100
        Number of bootstrap iterations.
    """

    metrics = {
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1_score": [],
    }

    for i in range(n_iterations):
        boot_x, boot_y = resample(X, y)
        y_proba = model.predict_proba(boot_x)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        roc_auc = roc_auc_score(boot_y, y_proba)
        precision = precision_score(boot_y, y_pred)
        recall = recall_score(boot_y, y_pred)
        tn, fp, fn, tp = confusion_matrix(boot_y, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(boot_y, y_pred)

        metrics["roc_auc"].append(roc_auc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["specificity"].append(specificity)
        metrics["f1_score"].append(f1)

    return metrics


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def repeated_kfold_evaluation(
    X,
    y,
    model_class,
    model_params,
    threshold=0.5,
    n_splits=5,
    n_repeats=10,
    upsample_R=0,
):
    """
    Perform repeated k-fold cross-validation to evaluate model performance.

    Parameters:
    ----------
    X : DataFrame
        Feature data.
    y : Series or array-like
        True labels.
    model_class : class
        The model class to be instantiated and evaluated.
    model_params : dict
        Parameters to initialize the model class.
    threshold : float, optional, default=0.5
        Threshold for converting predicted probabilities to binary predictions.
    n_splits : int, optional, default=5
        Number of folds for k-fold cross-validation.
    n_repeats : int, optional, default=10
        Number of times to repeat the cross-validation process.
    upsample_R : int, optional, default=0
        Factor by which to upsample the positive class in the training set.

    Returns:
    -------
    dict
        A dictionary containing lists of performance metrics for each fold and repeat.
    """

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    metrics = {"precision": [], "recall": [], "roc_auc": [], "specificity": []}

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if upsample_R != 0:
            X_train_positive = X_train[y_train == 1]
            y_train_positive = y_train[y_train == 1]
            X_train_negative = X_train[y_train == 0]
            y_train_negative = y_train[y_train == 0]

            X_train_positive_upsampled = pd.concat([X_train_positive] * upsample_R)
            y_train_positive_upsampled = pd.concat([y_train_positive] * upsample_R)

            X_train = pd.concat([X_train_negative, X_train_positive_upsampled])
            y_train = pd.concat([y_train_negative, y_train_positive_upsampled])

        # Initialize and train the model using the Models class
        model = model_class(
            {"all": pd.concat([X_train, y_train], axis=1)}, **model_params
        )
        output = model.returning_output(output_plots=False)

        # Extract the necessary metrics from the model output
        metrics["precision"].append(output["Precision"])
        metrics["recall"].append(output["Sensitivity"])
        metrics["roc_auc"].append(output["ROC_AUC"])
        metrics["specificity"].append(output["Specificity"])

    # Calculate mean and confidence intervals for each metric
    summary = {}
    for metric in metrics:
        mean_metric = np.mean(metrics[metric])
        ci_lower = mean_metric - sem(metrics[metric]) * 1.96
        ci_upper = mean_metric + sem(metrics[metric]) * 1.96
        summary[metric] = {
            "mean": mean_metric,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return summary


def calculate_mean_sem(outputs, metric):
    mean_vals = []
    sem_vals = []
    for n in sorted(outputs.keys(), reverse=True):
        metric_values = [run[metric] for run in outputs[n]]
        mean_vals.append(np.mean(metric_values))
        sem_vals.append(sem(metric_values))
    return mean_vals, sem_vals


def calculate_mean_ci(outputs, metric, confidence=0.95):
    mean_vals = []
    ci_vals = []
    z_value = 1.96
    for n in sorted(outputs.keys(), reverse=True):
        metric_values = [run[metric] for run in outputs[n]]
        mean = np.mean(metric_values)
        std_err = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))
        ci = std_err * z_value
        mean_vals.append(mean)
        ci_vals.append(ci)
    return mean_vals, ci_vals


def plot_backwards_elim(outputs, vline=None, figsize=(10, 5)):
    """
    Plots performance metrics vs. number of features removed with lines of best fit.

    Parameters:
    outputs (dict): Dictionary where keys are the number of features and values are
                    dictionaries containing performance metrics.
    vline (int, optional): Vertical line to indicate a specific point on the plot.
    figsize (tuple, optional): Figure size.
    """
    num_features = sorted(outputs.keys(), reverse=True)
    num_features_removed = [max(num_features) - n for n in num_features]

    sensitivity_means, sensitivity_cis = calculate_mean_ci(
        outputs, "Sensitivity_shifted"
    )
    specificity_means, specificity_cis = calculate_mean_ci(
        outputs, "Specificity_shifted"
    )
    roc_auc_means, roc_auc_cis = calculate_mean_ci(outputs, "ROC_AUC")

    plt.figure(figsize=figsize)
    plt.errorbar(
        num_features_removed,
        sensitivity_means,
        yerr=sensitivity_cis,
        label="Sensitivity",
        color="blue",
        fmt="x",
    )
    plt.errorbar(
        num_features_removed,
        specificity_means,
        yerr=specificity_cis,
        label="Specificity",
        color="green",
        fmt="o",
    )
    plt.errorbar(
        num_features_removed,
        roc_auc_means,
        yerr=roc_auc_cis,
        label="ROC AUC",
        color="red",
        fmt="*",
    )
    plt.xlabel("Number of Features Removed")
    plt.ylabel("Metric Score")
    plt.title("Performance Metrics vs. Number of Features Removed")
    plt.legend(frameon=False)
    plt.ylim(0, 1.1)
    if vline is not None:
        plt.axvline(x=vline, color="gray", linestyle="--")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def plot_backwards_elim_2_plots(outputs1, outputs2, figsize=(15, 5)):
    """
    Plots performance metrics vs. number of features removed with lines of best fit.

    Parameters:
    outputs1 (dict): Dictionary where keys are the number of features and values are
                     dictionaries containing performance metrics for the first group.
    outputs2 (dict): Dictionary where keys are the number of features and values are
                     dictionaries containing performance metrics for the second group.
    """

    def extract_and_plot(ax, outputs, color, label_suffix):
        # Extracting the data for plotting
        num_features = sorted(outputs.keys(), reverse=True)
        sensitivity = [outputs[n]["Sensitivity_shifted"] for n in num_features]
        specificity = [outputs[n]["Specificity_shifted"] for n in num_features]
        roc_auc = [outputs[n]["ROC_AUC"] for n in num_features]

        # Calculate number of features removed
        num_features_removed = [max(num_features) - n for n in num_features]

        # Fit linear regression for each metric
        sensitivity_fit = np.polyfit(num_features_removed, sensitivity, 1)
        specificity_fit = np.polyfit(num_features_removed, specificity, 1)
        roc_auc_fit = np.polyfit(num_features_removed, roc_auc, 1)

        # Create polynomial functions from the fits
        sensitivity_poly = np.poly1d(sensitivity_fit)
        specificity_poly = np.poly1d(specificity_fit)
        roc_auc_poly = np.poly1d(roc_auc_fit)

        # Generate x values for the lines of best fit
        x = np.linspace(min(num_features_removed), max(num_features_removed), 100)

        # Plot scatter plots with regression lines
        ax.scatter(
            num_features_removed,
            sensitivity,
            label=f"Sensitivity {label_suffix}",
            color=color[0],
            marker="x",
        )
        ax.plot(x, sensitivity_poly(x), color=color[0], linestyle="--")

        ax.scatter(
            num_features_removed,
            specificity,
            label=f"Specificity {label_suffix}",
            color=color[1],
            marker="o",
        )
        ax.plot(x, specificity_poly(x), color=color[1], linestyle="--")

        ax.scatter(
            num_features_removed,
            roc_auc,
            label=f"ROC AUC {label_suffix}",
            color=color[2],
            marker="*",
        )
        ax.plot(x, roc_auc_poly(x), color=color[2], linestyle="--")

        ax.set_xlabel("Number of Features Removed")
        ax.set_ylabel("Metric Score")
        ax.set_ylim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Plot for the first outputs dictionary
    extract_and_plot(axes[0], outputs1, ["blue", "green", "red"], "Group 1")

    # Plot for the second outputs dictionary
    extract_and_plot(axes[1], outputs2, ["blue", "green", "red"], "Group 2")

    # Set titles
    axes[0].set_title("Performance Metrics for Distance Features")
    axes[1].set_title("Performance Metrics for Non-Distance Features")

    # Add legends
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    plt.show()


def plot_catalogue_proportions(
    catalogue, background=None, figsize=None, order=True, title=None
):
    """
    Plots the proportions with confidence intervals for mutations in the given catalogue.

    Parameters:
    catalogue (dict): A dictionary where keys are mutation identifiers and values are dictionaries
                      containing mutation details, including 'proportion' and 'confidence' intervals.
    background (float): A value on which to draw the vertical background line. Defaults to None.
    figsize (tuple): A tuple representing the figure size. Default is (10, 20).
    order (bool): Whether to order by proportion. Default: True
    title (str): Title of the plot. Defaults to None.
    """
    rows = []
    for mutation, details in catalogue.items():
        evid = details["evid"][0]
        rows.append(
            {
                "Mutation": mutation,
                "Proportion": evid["proportion"],
                "Lower_bound": evid["confidence"][0],
                "Upper_bound": evid["confidence"][1],
                "Interval": evid["confidence"][1] - evid["confidence"][0],
                "Background": background,
            }
        )
    df = pd.DataFrame(rows)

    # Sort DataFrame by Proportion
    if order:
        df = df.sort_values(by=["Proportion", "Interval"], ascending=[False, False])

    dataframes = []
    length = 106
    start = 0
    end = length
    while True:
        if len(df) > end:
            dataframes.append(df[start:end])
            start += length
            end += length
        else:
            dataframes.append(df[start:])
            break

    figures = []
    axes = []

    for df2 in dataframes:
        # Plotting
        if figsize is None:
            height = len(df2) / 9.85 + 0.9
            fig, ax = plt.subplots(figsize=(4, height))
        else:
            fig, ax = plt.subplots(figsize=figsize)
        xerr = [
            abs(df2["Proportion"] - df2["Lower_bound"]),
            abs(df2["Upper_bound"] - df2["Proportion"]),
        ]

        for i in range(len(df2)):
            ax.plot(
                [df2["Lower_bound"].iloc[i], df2["Upper_bound"].iloc[i]],
                [i, i],
                color="#377eb8",
                lw=1,
            )
            ax.plot(
                df2["Proportion"].iloc[i], i, marker="|", color="#377eb8", markersize=10
            )
            if background is not None:
                ax.axvline(
                    x=df2["Background"].iloc[i], color="red", linestyle="--", lw=1
                )

        ax.set_yticks(np.arange(len(df2)))
        ax.set_yticklabels([i if len(i) < 20 else i[:20] for i in df2["Mutation"]])
        ax.set_title(title)
        for item in ax.get_yticklabels():
            if figsize is None:
                item.set_fontsize(7)
            else:
                item.set_fontsize(9)

        for item in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels():
            item.set_fontsize(9)

        plt.xlabel("proportion resistant")
        # plt.ylabel("mutation")
        plt.tight_layout()
        plt.xlim(-0.05, 1.05)
        ax.set_ylim(-0.5, len(df2) - 0.5)  # Adjust y-axis limits to fit the data
        sns.despine()
        figures.append(fig)
        axes.append(ax)

    return figures, axes


def str_to_dict(s):
    """Convert strings to dictionary - helpful for evidence column"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def plot_distance_vs_proportion(all_solos, data, feature):
    """
    Plot distance to specified feature vs proportion resistance, with confidence measures.

    This function takes in two dataframes, `all_solos` and `data`, and generates a scatter plot showing the relationship
    between the distance to specificed feature and the proportion of resistance for each unique mutation. The size of the markers
    represents the confidence interval width of the proportion resistance.

    Parameters:
    all_solos (pd.DataFrame): DataFrame containing mutation and phenotype information for all observed solos.
    data (pd.DataFrame): DataFrame containing mutation and feature distance information - ie the ML dataset that contains features.
    feautre (str): Name of feature column in the data df to which distances will be calculated.

    Returns:
    None: Displays a plot.
    """

    # Cross tabulation to get phenotype counts for each unique mutation
    ct = pd.crosstab([all_solos.MUTATION], all_solos.PHENOTYPE)
    ct["total"] = ct.sum(axis=1)
    ct.sort_values(by="total", inplace=True, ascending=False)
    ct[["proportion_R", "ci_lower", "ci_upper"]] = ct.apply(
        lambda row: wilson(row["R"], row["total"]), axis=1, result_type="expand"
    )
    ct = ct.reset_index().rename(columns={"MUTATION": "mutation"})

    # Merging ct with ML dataframe to pull out distances to rifampicin for each mutation
    ct_features = pd.merge(ct, data, on=["mutation"], how="inner")
    ct_features["ci_width"] = ct_features["ci_upper"] - ct_features["ci_lower"]
    ct_features["marker_size"] = (
        1 / ct_features["ci_width"]
    ) * 30  # Scale for visibility

    # Plot distance to featurevs proportion resistance, with confidence measure
    fig, ax = plt.subplots(figsize=(12, 5))
    sc = ax.scatter(
        ct_features[feature],
        ct_features["proportion_R"],
        s=ct_features["marker_size"],
        c="purple",
        alpha=0.6,
        edgecolors="w",
        label="Data points",
    )
    sns.regplot(
        x=feature,
        y="proportion_R",
        data=ct_features,
        scatter=False,
        color="purple",
        ax=ax,
    )
    ax.set_xlabel(f"Distance")
    ax.set_ylabel("Proportion Resistance")
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    sns.despine(ax=ax, top=True, right=True)
    norm = plt.Normalize(ct_features["ci_width"].min(), ct_features["ci_width"].max())
    sm = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
    sm.set_array([])
    plt.show()


def plot_metrics_with_ci(
    results, feature_list, metrics, sort_by="Recall", figsize=(12, 10)
):
    """
    Plot performance metrics with confidence intervals for each feature.

    Parameters:
    results (dict): Dictionary containing results for each feature.
    feature_list (list): List of feature names.
    metrics (list): List of metrics to plot (e.g., ['Recall', 'Specificity']).
    sort_by (str): The metric to sort the features by.
    figsize (tuple): Dimensions of the plot (width, height).

    Returns:
    None
    """

    def calculate_mean_ci(results, feature_list, metric):
        mean_metric = []
        ci_lower = []
        ci_upper = []

        for feat in feature_list:
            metric_values = [result[metric] for result in results[feat]]
            mean_metric.append(np.mean(metric_values))
            ci = sem(metric_values) * 1.96  # 95% confidence interval
            ci_lower.append(np.mean(metric_values) - ci)
            ci_upper.append(np.mean(metric_values) + ci)

        return mean_metric, ci_lower, ci_upper

    # Calculate mean and CI for each metric
    all_metrics = {}
    for metric in metrics:
        mean_values, ci_lower, ci_upper = calculate_mean_ci(
            results, feature_list, metric
        )
        mean_values = np.array(mean_values)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)
        all_metrics[metric] = {
            "mean": mean_values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    # Sorting features by the specified metric
    sorted_features = sorted(
        zip(
            feature_list,
            all_metrics[sort_by]["mean"],
            all_metrics[sort_by]["ci_lower"],
            all_metrics[sort_by]["ci_upper"],
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_feature_list, _, _, _ = zip(*sorted_features)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["purple", "blue", "green", "red", "orange"]  # Add more colors if needed

    for idx, metric in enumerate(metrics):
        sorted_metric = sorted(
            zip(
                feature_list,
                all_metrics[metric]["mean"],
                all_metrics[metric]["ci_lower"],
                all_metrics[metric]["ci_upper"],
            ),
            key=lambda x: sorted_feature_list.index(x[0]),
        )
        (
            sorted_feature_list_metric,
            sorted_mean_values,
            sorted_ci_lower,
            sorted_ci_upper,
        ) = zip(*sorted_metric)

        ax.errorbar(
            sorted_mean_values,
            sorted_feature_list_metric,
            xerr=[
                sorted_mean_values - np.array(sorted_ci_lower),
                np.array(sorted_ci_upper) - sorted_mean_values,
            ],
            fmt="|",
            ecolor=colors[idx],
            capsize=5,
            label=f"{metric} with 95% CI",
            color=colors[idx],
        )

    ax.set_ylabel("Single Features")
    ax.set_xlabel("Performance Metric")
    ax.legend(frameon=False)
    plt.xlim(0, 1.05)
    ax.set_title("Univariate Logistic Regression with Confidence Intervals")
    sns.despine(ax=ax, top=True, right=True)
    plt.show()


def calculate_mean_ci(results, metric):
    means = []
    ci_95 = []
    for model, metrics in results.items():
        data = [m[metric] for m in metrics]
        mean = np.mean(data)
        ci = 1.96 * np.std(data) / np.sqrt(len(data))
        means.append(mean)
        ci_95.append(ci)
        print (metric)
        print (ci_95)
    return means, ci_95

def plot_recall_specificity_with_ci(results, figsize=(12, 8)):
    # Calculate mean and 95% CI for recall, specificity, and ROC AUC
    mean_metrics = {}
    ci_metrics = {}

    for model_type, metrics_list in results.items():
        mean_metrics[model_type] = {}
        ci_metrics[model_type] = {}
        for metric in ["Sensitivity_shifted", "Specificity_shifted", "ROC_AUC"]:
            mean, ci = calculate_mean_ci({model_type: metrics_list}, metric)
            mean_metrics[model_type][metric] = mean[0]
            ci_metrics[model_type][metric] = ci[0]

    # Metrics to plot
    metric_mapping = {
        "Sensitivity_shifted": "Sensitivity",
        "Specificity_shifted": "Specificity",
        "ROC_AUC": "ROC AUC",
    }
    metric_names = ["Sensitivity", "Specificity", "ROC AUC"]
    model_type_names = list(mean_metrics.keys())
    n_model_types = len(model_type_names)
    n_metrics = len(metric_names)

    # Positioning the bars
    bar_width = 0.15
    space_within_metric = 0.03  # space between bars within the same metric
    space_between_metrics = 0.15  # larger space between different metrics
    index = np.arange(n_metrics) * (n_model_types * (bar_width + space_within_metric) + space_between_metrics)

    # Set color palette
    colors = sns.color_palette("colorblind", n_model_types)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, model_type in enumerate(model_type_names):
        means = [
            mean_metrics[model_type][shifted_metric] for shifted_metric in metric_mapping.keys()
        ]
        cis = [
            ci_metrics[model_type][shifted_metric] for shifted_metric in metric_mapping.keys()
        ]

        bars = ax.bar(
            index + i * (bar_width + space_within_metric),
            means,
            bar_width,
            yerr=cis,
            capsize=5,
            edgecolor=colors[i],
            facecolor='none',
            label=model_type,
        )

        # Add mean values above each bar
        for bar, mean, ci in zip(bars, means, cis):
            height = bar.get_height() + ci
            ax.annotate(
                f'{mean:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
            )

    # Add labels, title, and legend
    ax.set_xticks(index + (bar_width + space_within_metric) * (n_model_types - 1) / 2)
    ax.set_xticklabels(['']*n_metrics)  # Remove xtick labels

    # Add metric labels at the top
    for i, metric in enumerate(metric_names):
        ax.text(index[i] + (bar_width + space_within_metric) * (n_model_types - 1) / 2, 1.07, metric,
                ha='center', va='bottom', fontsize=12)

    # Remove y-axis
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add model labels under each bar
    for i, model_type in enumerate(model_type_names):
        for j in range(n_metrics):
            ax.text(index[j] + i * (bar_width + space_within_metric), -0.05, model_type, ha='center', va='top')

    # Style the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(('outward', 10))

    # Show the plot
    plt.show()

def calculate_sensitivity(tp, fn):
    if tp + fn == 0:
        return float("NaN")
    return tp / (tp + fn)


def calculate_specificity(tn, fp):
    if tn + fp == 0:
        return float("NaN")
    return tn / (tn + fp)


def metrics_vs_distance(stats_df, figsize=(12, 5)):
    """
    Plots sensitivity and specificity against Rif distance for resistant and susceptible mutations.

    Args:
        stats_df (pd.DataFrame): DataFrame containing mutation statistics, including TP, FN, TN, FP,
                                 phenotype, and Rif_distance.
        figsize (tuple): Size of the plot (width, height). Default is (12, 5).

    Returns:
        None
    """

    sensitivities = []
    specificities = []

    for index, row in stats_df.iterrows():
        if row["phenotype"] == 1:  # Resistant
            sensitivity = calculate_sensitivity(row["TP"], row["FN"])
            sensitivities.append(sensitivity)
            specificities.append(float("NaN"))
        elif row["phenotype"] == 0:  # Susceptible
            specificity = calculate_specificity(row["TN"], row["FP"])
            specificities.append(specificity)
            sensitivities.append(float("NaN"))

    stats_df["Sensitivity"] = sensitivities
    stats_df["Specificity"] = specificities

    resistant_mutations = stats_df[stats_df["phenotype"] == 1]
    susceptible_mutations = stats_df[stats_df["phenotype"] == 0]

    plt.figure(figsize=figsize)
    plt.scatter(
        resistant_mutations["Rif_distance"],
        resistant_mutations["Sensitivity"],
        marker="x",
        color="brown",
        label="Sensitivity (Resistant)",
        alpha=0.9,
    )
    plt.scatter(
        susceptible_mutations["Rif_distance"],
        susceptible_mutations["Specificity"],
        marker="x",
        color="darkOrange",
        label="Specificity (Susceptible)",
        alpha=0.9,
    )

    plt.xlabel("Rif Distance")
    plt.ylabel("Value")

    legend = plt.legend()
    legend.get_frame().set_linewidth(0)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.grid(False)
    plt.show()


def generate_feature_set(df):
    """Generates feature set as described in methods.ipynb - should match exactly"""

    features = sbmlcore.FeatureDataset(
        df, protein="RNAP", species="M. tuberculosis", gene="rpoB"
    )

    volume = sbmlcore.AminoAcidVolumeChange()
    hydropathy = sbmlcore.AminoAcidHydropathyChangeKyteDoolittle()
    mw = sbmlcore.AminoAcidMWChange()
    pi = sbmlcore.AminoAcidPiChange()
    rogov = sbmlcore.AminoAcidRogovChange()
    rif_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "resname RFP",
        "Rif_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    mg_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "resname MG",
        "Mg_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    zn1_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "index 26082 and resname ZN",
        "Zn1_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    zn2_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "index 26083 and resname ZN",
        "Zn2_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    antisense_p_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "segid G and name P",
        "antisense_P_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    sense_p_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "segid H and name P",
        "sense_P_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    rna_distance = sbmlcore.StructuralDistances(
        "./data/pdb/5uh6.pdb",
        "segid I",
        "RNA_distance",
        infer_masses=True,
        offsets={"C": -6},
    )
    stride = sbmlcore.Stride(
        "./data/pdb/5uh6-peptide-only.pdb",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
    )
    freesasa = sbmlcore.FreeSASA("./data/pdb/5uh6.pdb", offsets={"C": -6})
    snap2 = sbmlcore.SNAP2(
        "./data/stride/5uh6-complete.csv",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
    )
    deepddg = sbmlcore.DeepDDG(
        "./data/ddg/5uh6.ddg", offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0}
    )
    rasp = sbmlcore.RaSP("./data/rasp/cavity_pred_5uh6_C.csv", offsets={"C": -6})
    temp = sbmlcore.TempFactors(
        "./data/pdb/5uh6.pdb", offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0}
    )
    rif_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "resname RFP",
        "Rif_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    mg_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "resname MG",
        "Mg_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    zn1_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "resid 1283 and resname ZN",
        "Zn1_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    zn2_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "resid 1284 and resname ZN",
        "Zn2_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    antisense_P_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "index 50737:51209 and name P",
        "antisense_P_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    sense_P_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "index 51210:51946 and name P",
        "sense_P_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    rna_min_distance = sbmlcore.TrajectoryDistances(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "resname G or resname A",
        "RNA_min_distance",
        distance_type="min",
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    phi_mean = sbmlcore.TrajectoryDihedrals(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "phi",
        "mean_phi",
        angle_type="mean",
        add_bonds=True,
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )
    psi_mean = sbmlcore.TrajectoryDihedrals(
        "./data/md_files/rpob-5uh6-3-warm.gro",
        [
            "./data/md_files/rpob-5uh6-3-md-1-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-2-50ns-dt1ns-nojump.xtc",
            "./data/md_files/rpob-5uh6-3-md-3-50ns-dt1ns-nojump.xtc",
        ],
        "./data/pdb/5uh6.pdb",
        "psi",
        "mean_psi",
        angle_type="mean",
        add_bonds=True,
        offsets={"A": 0, "B": 0, "C": -6, "D": 0, "E": 0, "F": 0},
        percentile_exclusion=True,
    )

    features.add_feature([volume, hydropathy, mw, pi, rogov])
    features.add_feature(
        [
            rif_distance,
            mg_distance,
            zn1_distance,
            zn2_distance,
            antisense_p_distance,
            sense_p_distance,
            rna_distance,
        ]
    )
    features.add_feature([stride, freesasa, snap2, deepddg, rasp, temp])
    features.add_feature(
        [
            rif_min_distance,
            mg_min_distance,
            zn1_min_distance,
            zn2_min_distance,
            antisense_P_min_distance,
            rna_min_distance,
        ]
    )
    features.add_feature([phi_mean, psi_mean])

    features.df = features.df.dropna(subset=["T"])

    features.df["secondary_structure_codes"] = pd.Categorical(
        features.df.secondary_structure,
        categories=features.df.secondary_structure.unique(),
    ).codes
    features.df = features.df.drop(
        columns=[
            "secondary_structure",
            "secondary_structure_long",
            "B",
            "C",
            "E",
            "G",
            "H",
            "T",
        ]
    )
    features.df = features.df.drop(
        columns=[
            "residue_sasa",
            "snap2_accuracy",
            "rasp_wt_nlf",
            "rasp_mt_nlf",
            "rasp_score_ml",
        ]
    )
    features.df = features.df.rename(
        columns={
            "d_Pi": "d_pi",
            "rasp_score_ml_fermi": "rasp_score",
            "secondary_structure_codes": "secondary_structure",
        }
    )

    features.df.to_csv("./data/tables/generated/features_dataset.csv")

    df = features.df.copy()

    return df



def confusion_matrix(labels, predictions, classes):
    """
    Creates a confusion matrix for given labels and predictions with specified classes.

    Parameters:
    labels (list): Actual labels.
    predictions (list): Predicted labels.
    classes (list): List of all classes.

    Returns:
    np.ndarray: Confusion matrix.
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for label, prediction in zip(labels, predictions):
        if label in class_to_index and prediction in class_to_index:
            cm[class_to_index[label], class_to_index[prediction]] += 1

    return cm

def plot_truthtables(truth_table, figsize=(2.5, 1.5), fontsize=12):
    """
    Plots a truth table as a confusion matrix to denote each cell.

    Parameters:
    truth_table (pd.DataFrame): DataFrame containing the truth table values.
                                The DataFrame should have the following structure:
                                - Rows: True labels ("R" and "S")
                                - Columns: Predicted labels ("R" and "S")
    figsize (tuple): Figure size for the plot.
    fontsize (int): Font size for the text in the plot.

    Returns:
    None
    """
    fig = plt.figure(figsize=figsize)
    axes = plt.gca()

    axes.set_xlim([0, 2])
    axes.set_xticks([0.5, 1.5])
    axes.set_xticklabels(["S", "R"], fontsize=fontsize)

    axes.add_patch(Rectangle((0, 0), 1, 1, fc="#e41a1c", alpha=0.7))
    axes.add_patch(Rectangle((1, 0), 1, 1, fc="#4daf4a", alpha=0.7))
    axes.add_patch(Rectangle((1, 1), 1, 1, fc="#fc9272", alpha=0.7))
    axes.add_patch(Rectangle((0, 1), 1, 1, fc="#4daf4a", alpha=0.7))

    axes.set_ylim([0, 2])
    axes.set_yticks([0.5, 1.5])
    axes.set_yticklabels(["R", "S"], fontsize=fontsize)

    axes.text(
        1.5,
        0.5,
        int(truth_table["R"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        1.5,
        1.5,
        int(truth_table["R"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        1.5,
        int(truth_table["S"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        0.5,
        int(truth_table["S"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    plt.show()