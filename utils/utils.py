import pandas as pd
import MDAnalysis as mda
from scipy.stats import norm
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os


def map_mut2pdb(df, pdb, outfile):
    """
    Maps mutation positions onto a PDB structure and writes the result
    to temperature factors in a new PDB file.

    Parameters:
    - df (DataFrame): DataFrame with columns ['RESID', 'SEGID'] indicating mutation sites.
    - pdb (str): Path to the input PDB file.
    - outfile (str): Path to the output PDB file where the results will be saved.
    """

    # Load the PDB and select only the protein atoms
    universe = mda.Universe(pdb)
    protein = universe.select_atoms('protein')
    temp_pdb = './data/temp/proteinOnly.pdb'
    protein.write(temp_pdb)

    # Re-load the protein-only pdb
    u = mda.Universe(temp_pdb)

    # Extract chain identifiers and residue IDs from the PDB
    chains = [str(atom.segment)[9] for atom in u.atoms]
    resids = [atom.resid for atom in u.atoms]
    pdb_df = pd.DataFrame({"segid": chains, "resid": resids})

    # Aggregate the mutation DataFrame to ensure unique (SEGID, RESID) pairs
    df_unique = df.drop_duplicates(subset=['segid', 'resid'])

    # Map mutations onto atoms
    merged_df = pdb_df.merge(df_unique, on=["segid", "resid"], how="left", indicator=True)
    merged_df['mutated'] = (merged_df['_merge'] == 'both').astype(int)

    # Write mutations to temperature factors
    u.add_TopologyAttr("tempfactors", values=merged_df['mutated'].tolist())
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
        group[group['PHENOTYPE'] == "R"] if "R" in group['PHENOTYPE'].values else group
    )

    # Check for rows with METHOD_MIC values
    with_mic = prioritized_group.dropna(subset=['METHOD_MIC'])
    return with_mic if not with_mic.empty else prioritized_group.iloc[0:1]

def wilson(count, n, confidence=0.95):
    """Calculates wilson confidence intervals for supplied counts"""
    if n == 0:
        return (0, 0)

    proportion = count / n
    z = norm.ppf(1 - (1-confidence) / 2)
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
        'roc_auc': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f1_score': []
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

        metrics['roc_auc'].append(roc_auc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['specificity'].append(specificity)
        metrics['f1_score'].append(f1)

    return metrics

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def repeated_kfold_evaluation(model, X, y, threshold=0.5, n_splits=5, n_repeats=10):
    """
    Perform repeated k-fold cross-validation to evaluate model performance.

    Parameters:
    ----------
    model : sklearn estimator
        The model to be trained and evaluated.
    X : DataFrame
        Feature data.
    y : Series or array-like
        True labels.
    threshold : float, optional, default=0.5
        Threshold for converting predicted probabilities to binary predictions.
    n_splits : int, optional, default=5
        Number of folds for k-fold cross-validation.
    n_repeats : int, optional, default=10
        Number of times to repeat the cross-validation process.

    Returns:
    -------
    dict
        A dictionary containing lists of performance metrics for each iteration.
    """
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    metrics = {
        'precision': [],
        'recall': [],
        'specificity': [],
        'roc_auc': []
    }

    for i in range(n_repeats):
        accuracies = []
        precisions = []
        recalls = []
        specificities = []
        roc_aucs = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred))
            specificities.append(specificity_score(y_test, y_pred))
            roc_aucs.append(roc_auc_score(y_test, y_proba))

        metrics['precision'].append(precisions)
        metrics['recall'].append(recalls)
        metrics['specificity'].append(specificities)
        metrics['roc_auc'].append(roc_aucs)

    return {metric: np.array(values) for metric, values in metrics.items()}

    
def plot_backwards_elim(outputs, vline=None, figsize=(10, 5)):
    """
    Plots performance metrics vs. number of features removed with lines of best fit.

    Parameters:
    outputs (dict): Dictionary where keys are the number of features and values are 
                    dictionaries containing performance metrics.
    """
    # Extracting the data for plotting
    num_features = sorted(outputs.keys(), reverse=True)
    sensitivity = [outputs[n]['Sensitivity_shifted'] for n in num_features]
    specificity = [outputs[n]['Specificity_shifted'] for n in num_features]
    roc_auc = [outputs[n]['ROC_AUC'] for n in num_features]

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

    plt.figure(figsize=figsize)
    plt.scatter(num_features_removed, sensitivity, label='Sensitivity', color='blue', marker='x')
    plt.plot(x, sensitivity_poly(x), color='blue', linestyle='--')
    plt.scatter(num_features_removed, specificity, label='Specificity', color='green', marker='o')
    plt.plot(x, specificity_poly(x), color='green', linestyle='--')
    plt.scatter(num_features_removed, roc_auc, label='ROC AUC', color='red', marker='*')
    plt.plot(x, roc_auc_poly(x), color='red', linestyle='--')
    plt.xlabel('Number of Features Removed')
    plt.ylabel('Metric Score')
    plt.title('Performance Metrics vs. Number of Features Removed')
    plt.legend(frameon=False)
    plt.ylim(0, 1.1)
    if vline != None:
        plt.vlines(x=vline, ymin=0, ymax=1, )
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
        sensitivity = [outputs[n]['Sensitivity_shifted'] for n in num_features]
        specificity = [outputs[n]['Specificity_shifted'] for n in num_features]
        roc_auc = [outputs[n]['ROC_AUC'] for n in num_features]

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
        ax.scatter(num_features_removed, sensitivity, label=f'Sensitivity {label_suffix}', color=color[0], marker='x')
        ax.plot(x, sensitivity_poly(x), color=color[0], linestyle='--')

        ax.scatter(num_features_removed, specificity, label=f'Specificity {label_suffix}', color=color[1], marker='o')
        ax.plot(x, specificity_poly(x), color=color[1], linestyle='--')

        ax.scatter(num_features_removed, roc_auc, label=f'ROC AUC {label_suffix}', color=color[2], marker='*')
        ax.plot(x, roc_auc_poly(x), color=color[2], linestyle='--')

        ax.set_xlabel('Number of Features Removed')
        ax.set_ylabel('Metric Score')
        ax.set_ylim(0, 1.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Plot for the first outputs dictionary
    extract_and_plot(axes[0], outputs1, ['blue', 'green', 'red'], 'Group 1')

    # Plot for the second outputs dictionary
    extract_and_plot(axes[1], outputs2, ['blue', 'green', 'red'], 'Group 2')

    # Set titles
    axes[0].set_title('Performance Metrics for Distance Features')
    axes[1].set_title('Performance Metrics for Non-Distance Features')

    # Add legends
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    plt.show()