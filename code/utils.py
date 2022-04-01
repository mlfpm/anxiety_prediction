import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import TensorDataset, DataLoader

sns.set_style("white")
sns.set_palette("colorblind")


# ---------------------------------------------- Data Pre-processing ------------------------------------------------- #

def remove_groups_without_data(df, group, columns):
    """
    If all the values in the group are NaNs, remove the entries from the dataframe.

    :param pd.DataFrame df: the pandas dataframe
    :param str group: column label to be grouped by
    :param list columns: the list of column labels of interest
    :return: the processed dataframe
    """
    """
        If all the values in the group are NaNs, remove the entries from
        the dataframe.
        Args:
            df (DataFrame) - the pandas dataframe
            group (str) - column label to be grouped by
            columns (list) - the list of column labels
        Returns:
            df (DataFrame) - the pandas dataframe without groups that only have
                NaNs in the specified columns
    """
    df_copy = pd.DataFrame(columns=list(df.columns))

    # for each unique value in the group column (patient)
    for unq in df[group].unique():
        all_nan_counter = 0
        # for each column that has to be filled in
        for column in columns:
            # if there are only NaNs in the column
            if df[df[group] == unq][column].isnull().all():
                all_nan_counter += 1

        # if all the columns are nans ignore the patient
        if all_nan_counter != len(columns):
            df_copy = pd.concat([df_copy, df[df[group] == unq]])

    df_copy.reset_index(drop=True, inplace=True)

    return df_copy


def resample_dates(df, pid_col):
    """
    Resample the dataframe such that no days of the year are missed.

    :param pd.DataFrame df: the dataframe containing temporal data
    :param str pid_col: column to group by
    :return: the resampled dataframe
    """
    resampled_dfs = []
    for pid in df[pid_col].unique():
        df_temp = df[df[pid_col] == pid].set_index("date")
        df_temp = df_temp.resample("D").asfreq().reset_index()
        df_temp[pid_col] = pid
        resampled_dfs.append(df_temp)
    res = pd.concat(resampled_dfs, axis=0).sort_values(
        by=[pid_col, "date"], ascending=[True, True]
    )
    return res


def clinical_data_encoder(df_nt):
    """
    Encode original clinical data values to categorical.

    :param pd.DataFrame df_nt: the dataframe containing the clinical data
    :return: the processed dataframe
    """
    # ["Anxiety_Group", "Sex", "Lives_Alone", "CurrentActivity", "Covid25", "Covid34", "Covid37", "Covid43"]
    df_nt["Anxiety_Group"] = df_nt["Anxiety_Group"].fillna("NaN")
    df_nt["Anxiety_Group"] = df_nt["Anxiety_Group"].map({"Non-Clinical Anxiety": 0, "Clinical Anxiety": 1})

    df_nt["Sex"] = df_nt["Sex"].map({"Mujer": 0, "Varon": 1})

    df_nt["Lives_Alone"] = df_nt["Lives_Alone"].map({"No": 0, "Si": 1})

    df_nt["CurrentActivity"] = df_nt["CurrentActivity"].fillna("NaN")
    df_nt["CurrentActivity"] = df_nt["CurrentActivity"].map(
        {
            "NaN": 0,
            "Activo / ama de casa / estudiante": 1,
            "Desempleado(a) con beneficio": 2,
            "Desempleado(a) y sin beneficio": 3,
            "Incapacidad temporal": 4,
            "Incapacidad permanente": 5,
            "Jubilado(a)": 6,
        }
    )

    df_nt["Covid25"] = df_nt["Covid25"].fillna("NaN")
    df_nt["Covid25"] = df_nt["Covid25"].map(
        {"NaN": 0, "Mala": 1, "Regular": 2, "Buena": 3, "Muy buena": 3, "Excelente ": 3}
    )  # negative = 1, regular = 2, positive = 3;

    df_nt["Covid34"] = df_nt["Covid34"].fillna("NaN")
    df_nt["Covid34"] = df_nt["Covid34"].map({"NaN": 0, "No": 1, "Si": 2})

    df_nt["Covid37"] = df_nt["Covid37"].fillna("NaN")
    df_nt["Covid37"] = df_nt["Covid37"].map(
        {
            "NaN": 0,
            "Nada en absoluto ": 1,
            "Ligeramente ": 2,
            "Moderadamente ": 3,
            "Mucho ": 4,
            "Extremadamente ": 5,
        }
    )

    df_nt["Covid43"] = df_nt["Covid43"].fillna("NaN")
    df_nt["Covid43"] = df_nt["Covid43"].map(
        {
            "NaN": 0,
            "Mucho menos ": 1,
            "Un poco menos ": 1,
            "Más o menos lo mismo ": 2,
            "Un poco más ": 3,
            "Mucho más": 3,
        }
    )  # 1 - much to little less, 2 - more or less the same, 3 - little to much more

    return df_nt


def clinical_data_decoder(df_nt):
    """
    Decode categorical clinical data values to text.

    :param pd.DataFrame df_nt: the dataframe containing the clinical data
    :return: the processed dataframe
    """
    df_nt["Anxiety_Group"] = df_nt["Anxiety_Group"].map({0: "Non-Clinical Anxiety", 1: "Clinical Anxiety"})
    df_nt["Sex"] = df_nt["Sex"].map({0: "Female", 1: "Male"})
    df_nt["Lives_Alone"] = df_nt["Lives_Alone"].map({0: "No", 1: "Yes"})
    df_nt["CurrentActivity"] = df_nt["CurrentActivity"].map(
        {
            0: "NA",
            1: "Employed, student or homemaker",
            2: "Unemployed with subsidy",
            3: "Unemployed without subsidy",
            4: "Long-term disability",
            5: "Temporarily incapacitated",
            6: "Retired",
        }
    )
    df_nt["Covid25"] = df_nt["Covid25"].map(
        # {"Buena": "Good", "Excelente": "Excellent", "Mala": "Bad", "Muy buena": "Very good", "Regular": "Regular"}
        {0: "NA", 1: "Negative", 2: "Regular", 3: "Positive"}
    )
    df_nt["Covid34"] = df_nt["Covid34"].map({0: "NA", 1: "No", 2: "Yes"})
    df_nt["Covid37"] = df_nt["Covid37"].map(
        {
            0: "NA",
            1: "Not at all",
            2: "Slightly",
            3: "Moderately",
            4: "A lot",
            5: "Extremely",
        }
    )
    df_nt["Covid43"] = df_nt["Covid43"].map(
        {
            0: "NA",
            1: "Much to little less",
            2: "More or less the same",
            3: "Little to much more",
        }
    )

    df_nt.rename(
        columns={
            "Age": "Age",
            "Lives_Alone": "Lives alone",
            "CurrentActivity": "Employment status",
            "Covid25": "Self-ratings of Physical Health",
            "Covid34": "Essential workers in household",
            "Covid37": "Worries about Life Stability",
            "Covid43": "Changes in Frequency of Social Interactions",
            "Anxiety_Group": "Anxiety group",
        },
        inplace=True,
    )

    return df_nt


#  --------------------------------------------- Data visualisation -------------------------------------------------- #
def f(max_chars):
    """
    Lambda expression to crop labels to max_chars length.

    :param int max_chars: maximum number of characters in a row
    """
    return lambda x: textwrap.fill(x.get_text(), max_chars)


def plot_time_series_per_groups(df, col, ax):
    """
    Create line plot of a temporal variable.

    :param pd.DataFrame df: dataframe containing the temporal data
    :param str col: column label to plot
    :param ax: figure axes to plot on
    """
    sns.lineplot(
        x="date", y=col, hue="Anxiety_Group", data=df, ax=ax
    )
    sns.despine(offset=10, trim=True, ax=ax)


def plot_lockdown_start(ax):
    """
    Plot vertical line at the date when the lockdown in Madrid started (2020-03-14).

    :param ax: figure axes to plot onto
    :return: the axes
    """
    ax.axvline(pd.Timestamp("2020-03-14"), color="k", linestyle="--")
    ax.text(
        x=pd.Timestamp("2020-03-14"),
        y=0.7,
        s="Lockdown start",
        alpha=0.7,
        color="k",
    )
    return ax


def plot_all_data_per_groups(df_imp, df_nt, nt_cols, fpath=None, figsize=(15, 15)):
    """
    Create plot of temporal and non-temporal variables in the patient population, grouped by anxiety type.

    :param pd.DataFrame df_imp: dataframe containing the temporal data
    :param pd.DataFrame df_nt: dataframe containing the clinical data
    :param list nt_cols: list of clinical data labels
    :param str fpath: full path to save the figure to; optional
    :param tuple figsize: figure size; optional
    """
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(3, 4)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax1 = fig.add_subplot(gs[0, :2])
    plot_time_series_per_groups(df_imp, "comm_med", ax1)
    ax1 = plot_lockdown_start(ax1)

    # format the ticks
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    ax1.set(xlabel="Date", ylabel="Scaled communication app usage")
    ax1.get_legend().remove()

    ax2 = fig.add_subplot(gs[0, 2:])
    plot_time_series_per_groups(df_imp, "social_med", ax2)
    ax2 = plot_lockdown_start(ax2)

    # format the ticks
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    ax2.set(xlabel="Date", ylabel="Scaled social app usage")

    col_idx = -1
    for i in range(1, 3):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            if col_idx == -1:
                sns.histplot(x="Age", hue="Anxiety group", data=df_nt, ax=ax)
                sns.despine(offset=10, trim=True, ax=ax)
            else:
                df_nt[nt_cols[col_idx]] = df_nt[nt_cols[col_idx]].fillna("NaN")
                sns.countplot(x=nt_cols[col_idx], hue="Anxiety group", data=df_nt, ax=ax)

                ax.set_xticklabels(map(f(7), ax.get_xticklabels()))
                sns.despine(offset=10, trim=True, ax=ax)
            col_idx += 1
            ax.get_legend().remove()

    if fpath is not None:
        fig.savefig(fpath, format='pdf', dpi=300)


def plot_kfold_train_test_data(
        comm_tr,
        comm_tst,
        ema_tr,
        ema_tst,
        ema_cols,
        acc_tr,
        acc_tst,
        split,
        figsize=(15, 20),
        max_chars=10,
):
    """
    Create plot of temporal and non-temporal variables in the patient population, grouped by anxiety type in each fold
    of the k-fold cross-validation.

    :param pd.DataFrame comm_tr: training set - dataframe containing the temporal data
    :param pd.DataFrame comm_tst: test set - dataframe containing the temporal data
    :param pd.DataFrame ema_tr: training set - dataframe containing the clinical data
    :param pd.DataFrame ema_tst: test set - dataframe containing the clinical data
    :param list ema_cols: list of clinical data labels
    :param float acc_tr: training accuracy
    :param float acc_tst: test accuracy
    :param int split: split index
    :param tuple figsize: figure size; optional
    :param int max_chars: maximum number of characters in a row when writing the labels; optional
    :return: the figure
    """
    n_features = len(ema_cols) + 2
    fig, axs = plt.subplots(n_features, 2, figsize=figsize)
    fig.suptitle("Split " + str(split))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for i in range(n_features):
        if i == 0:
            plot_time_series_per_groups(comm_tr, "comm_med", axs[i, 0])
            plot_time_series_per_groups(comm_tst, "comm_med", axs[i, 1])

            for j, ax in enumerate(axs[i].ravel()):
                # plot vertical line at lockdown start date
                ax = plot_lockdown_start(ax)

                # format the ticks
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                if j == 0:
                    ax.set(
                        xlabel="",
                        ylabel="Comm. app usage",
                        title="Training set (accuracy = {:.2f})".format(acc_tr),
                    )
                    ax.get_legend().remove()
                else:
                    ax.set(
                        xlabel="",
                        ylabel="",
                        title="Test set (accuracy = {:.2f})".format(acc_tst),
                    )
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.8))
        elif i == 1:
            plot_time_series_per_groups(comm_tr, "social_med", axs[i, 0])
            plot_time_series_per_groups(comm_tst, "social_med", axs[i, 1])

            for j, ax in enumerate(axs[i].ravel()):
                # plot vertical line at lockdown start date
                ax = plot_lockdown_start(ax)

                # format the ticks
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                ax.get_legend().remove()
        else:
            if ema_cols[i - 2] == "Age":
                sns.histplot(
                    x=ema_cols[i - 2], hue="Anxiety_Group", data=ema_tr, ax=axs[i, 0]
                )
                sns.despine(offset=10, trim=True, ax=axs[i, 0])
                sns.histplot(
                    x=ema_cols[i - 2], hue="Anxiety_Group", data=ema_tst, ax=axs[i, 1]
                )
                sns.despine(offset=10, trim=True, ax=axs[i, 1])
            else:
                sns.countplot(
                    x=ema_cols[i - 2], hue="Anxiety_Group", data=ema_tr, ax=axs[i, 0]
                )
                sns.despine(offset=10, trim=True, ax=axs[i, 0])
                sns.countplot(
                    x=ema_cols[i - 2], hue="Anxiety_Group", data=ema_tst, ax=axs[i, 1]
                )
                sns.despine(offset=10, trim=True, ax=axs[i, 1])

            for j, ax in enumerate(axs[i].ravel()):
                if j == 0:
                    ax.set(xlabel="", ylabel=ema_cols[i - 2])
                else:
                    ax.set(xlabel="", ylabel="")
                if ema_cols[i - 2] in ["CurrentActivity", "Covid37", "Covid43"]:
                    ax.set_xticklabels(map(f(max_chars), ax.get_xticklabels()))
                ax.get_legend().remove()

    plt.close()
    return fig


def plot_loo_temp_data_comparison(
        comm_wr, comm_cr, figsize=(15, 15), fpath=None
):
    """
    Create plot of LOOCV wrongly vs correctly classsified time series features.

    :param pd.DataFrame comm_wr: dataframe of the wrongly classified sequences
    :param pd.DataFrame comm_cr: dataframe of the correctly classified sequences
    :param tuple figsize: figure size; optional
    :param str fpath: full path to save the figure to; optional
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for i in range(2):
        if i == 0:
            sns.lineplot(
                x="date", y="comm_med", hue="Anxiety_Group", data=comm_cr, ax=axs[i, 0]
            )
            sns.despine(offset=10, trim=True, ax=axs[i, 0])
            sns.lineplot(
                x="date", y="comm_med", hue="Anxiety_Group", data=comm_wr, ax=axs[i, 1]
            )
            sns.despine(offset=10, trim=True, ax=axs[i, 1])

            for j, ax in enumerate(axs[i].ravel()):
                # plot vertical line at lockdown start date
                ax = plot_lockdown_start(ax)

                # format the ticks
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                if j == 0:
                    ax.set(
                        xlabel="",
                        ylabel="Communication app usage",
                        title="Correctly classified set",
                    )
                    ax.get_legend().remove()
                else:
                    ax.set(xlabel="", ylabel="", title="Misclassified set")
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.8))
        elif i == 1:
            sns.lineplot(
                x="date", y="social_med", hue="Anxiety_Group", data=comm_cr, ax=axs[i, 0]
            )
            sns.despine(offset=10, trim=True, ax=axs[i, 0])
            sns.lineplot(
                x="date", y="social_med", hue="Anxiety_Group", data=comm_wr, ax=axs[i, 1]
            )
            sns.despine(offset=10, trim=True, ax=axs[i, 1])

            for j, ax in enumerate(axs[i].ravel()):
                # plot vertical line at lockdown start date
                ax = plot_lockdown_start(ax)

                # format the ticks
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                ax.get_legend().remove()

                if j == 0:
                    ax.set(
                        xlabel="",
                        ylabel="Social app usage",
                    )
                else:
                    ax.set(xlabel="", ylabel="")

    if fpath is not None:
        fig.savefig(fpath, format='pdf', dpi=300)


def plot_loo_ema_data_comparison(
        ema_wr, ema_cr, ema_cols, figsize=(15, 25), fpath=None, max_chars=10
):
    """
    Create plot of LOOCV wrongly vs correctly classified time series features.

    :param pd.DataFrame ema_wr:
    :param pd.DataFrame ema_cr:
    :param list ema_cols:
    :param tuple figsize: figure size; optional
    :param str fpath: full path to save the figure to; optional
    :param int max_chars: maximum number of characters in a row when writing the labels; optional
    """
    fig, axs = plt.subplots(len(ema_cols), 2, figsize=figsize)

    for i in range(len(ema_cols)):
        sorted_values = np.sort(ema_cr[ema_cols[i]].unique())
        sns.countplot(
            y=ema_cols[i], hue="Anxiety group", order=sorted_values, data=ema_cr, orient="h", ax=axs[i, 0]
        )
        sns.countplot(
            y=ema_cols[i], hue="Anxiety group", order=sorted_values, data=ema_wr, orient="h", ax=axs[i, 1]
        )

        for j, ax in enumerate(axs[i].ravel()):
            if j == 0:
                ax.set(xlabel="", ylabel=ema_cols[i])
                if i == 0:
                    ax.set(title="Correctly classified set")
            else:
                ax.set(xlabel="", ylabel="")
                if i == 0:
                    ax.set(title="Misclassified set")

            ax.set_yticklabels(map(f(max_chars), ax.get_yticklabels()))

            if (i == 0 and j == 0) or (i > 0):
                ax.get_legend().remove()
            else:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.8))

            sns.despine(offset=10, trim=True, ax=ax)

    if fpath is not None:
        fig.savefig(fpath, format='pdf', dpi=300)


#  --------------------------------------------------- ML utils ------------------------------------------------------ #
def pad_features(df, feat_cols):
    """
    Return features of data, where each observation sequence is padded with 0's or truncated to the input seq_length.
    `features` array should be a 2D with as many rows as there are sequences, and as many columns as the
    max sequence length. If more then one column is used, the output is 3D.

    :param pd.DataFrame df: dataframe containing the temporal data
    :param list feat_cols: list of column labels to consider
    :return: padded features
    """
    # compute maximum sequence length
    seq_length = max([len(df[df.user == pid]) for pid in df.user.unique()])

    # getting the correct rows x cols shape
    if isinstance(feat_cols, str):
        features = np.zeros((df.user.nunique(), seq_length), dtype=np.float32)

        for i, row in enumerate(df.groupby("user")[feat_cols].apply(np.array).values):
            features[i, -len(row):] = row[:seq_length]
    else:
        features = np.zeros((df.user.nunique(), seq_length, len(feat_cols)), dtype=np.float32)

        for i, pid in enumerate(df.user.unique()):
            patient_seq = df[df.user == pid][feat_cols].values
            features[i, -len(patient_seq):, :] = patient_seq[:seq_length]

    return features


def train_test_split_kfold(t_features, nt_features, labels, train_idx, test_idx):
    """
    Split up data into test and training sets for the K-fold cross-validation.

    :param np.ndarray t_features: temporal features
    :param np.ndarray nt_features: static features
    :param np.array labels: target labels
    :param list train_idx: train indices
    :param list test_idx: test indices
    :return: split training and test data loaders
    """

    # split up the data
    train_x_t, test_x_t = (
        t_features[train_idx].astype(np.float32),
        t_features[test_idx].astype(np.float32),
    )
    train_x_s, test_x_s = (
        nt_features[train_idx].astype(np.float32),
        nt_features[test_idx].astype(np.float32),
    )
    train_y, test_y = (
        labels[train_idx].astype(np.float32),
        labels[test_idx].astype(np.float32),
    )

    # create Tensor datasets
    train_data = TensorDataset(
        torch.from_numpy(train_x_t),
        torch.from_numpy(train_x_s),
        torch.from_numpy(train_y),
    )
    test_data = TensorDataset(
        torch.from_numpy(test_x_t), torch.from_numpy(test_x_s), torch.from_numpy(test_y)
    )

    # data loaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=10, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=len(test_x_t))

    return train_loader, test_loader


#  --------------------------------------------------- I/O methods --------------------------------------------------- #
def save_figures_to_pdf(figures, file_path):
    """
    Save multiple figures into a single PDF.

    :param list figures: list of figure objects to write to file
    :param str file_path: full path to save the figure to
    """
    pp = PdfPages(file_path)
    for fig in figures:
        pp.savefig(fig)
    pp.close()


# ---------------------- Data set generation for HMMs -------------------- #
def df_to_list(df, columns, by_year=False):
    """
    Function to transform a dataframe into a list of observation sequences grouped by patient.

    :param pd.DataFrame df: dataframe containing the patient data
    :param list columns: list of columns to keep for training
    :return: list of observation sequences grouped by patient
    """
    res = []
    # for each patient
    for unq in df.user.unique():
        df_patient = df[df.user == unq]
        # extract observations
        if not by_year:
            res.append(df_patient[columns].to_numpy())
        else:
            years = df_patient['date'].dt.year.unique().tolist()
            for year in years:
                res.append(df_patient[df_patient.date.dt.year == year][columns].to_numpy())
    return res
