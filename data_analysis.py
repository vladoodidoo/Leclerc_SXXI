from typing import List

import pandas as pd
from pandas._config import display
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
SAVE_PATH = "output/"

class EDA():
    def __init__(self, df, num_cols, cat_cols, display=True, prefix=""):
        self.df = df
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.display = display
        self.prefix = prefix

        print(num_cols)
        print(cat_cols)

    def label_count_plot(self, cat_col) -> None:
        """
    
        :param df: The pd.DataFrame
        :param cat_col: The name of the columns containing the
                          categorical data to count.
        :return:
        """
        df = self.df

        sns.countplot(y=cat_col, data=df, palette="Set2",
                      order=df[cat_col].value_counts().index)
        plt.savefig(SAVE_PATH + self.prefix + "label_count.png")

        plt.show()
    
    
    def get_pairplot(self) -> None:
        df = self.df
        num_cols = self.num_cols

        sns.pairplot(df[num_cols], diag_kind="kde")
        plt.savefig(SAVE_PATH + self.prefix + "pairplot.png")


        plt.show()
    
    
    def get_corr_matrix(self) -> None:
        df = self.df
        num_cols = self.num_cols

        sns.heatmap(df[num_cols].corr(), annot=True, cmap='viridis')
        plt.savefig(SAVE_PATH + self.prefix + "correlation_matrix.png")


        plt.show()
    
    
    def numerical_analysis(self, hue: str) -> None:
        """
    
        :param df: The pd.DataFrame
    
        :param num_columns: The list of all numerical columns
                            that will be taken into account for
                            analysis.
    
        :param hue: A string containing the name of the column
                    that will be used as hue for sns.kdeplot
    
        :return: None
        """

    
        self.get_corr_matrix()
        self.get_pairplot()
    
        df = self.df
        num_columns = self.num_cols

        ncols = 1
        nrows = (len(num_columns) // ncols) + (len(num_columns) % ncols != 0)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 15))
        for idx in range(len(num_columns)):
            ax = axs[idx] if len(axs.shape) == 1 else axs[idx // ncols, idx % ncols]
            sns.kdeplot(data=df, x=num_columns[idx], hue=hue,
                        ax=ax,
                        fill=True, palette="Set2")
        fig.savefig(SAVE_PATH + self.prefix + "KDE_plots.png")
        fig.show()
    
    
    def categorical_analysis(self, hue: str) -> None:
        """
        :param df: pd.DataFrame
    
        :param cat_columns: A string list containing the
                            categorical columns of the
                            pd.DataFrame
    
        :param hue: The hue for the sns.countplot
    
        :return: None
        """
        df = self.df
        cat_columns = self.cat_cols

        ncols = 1
        nrows = len(cat_columns) // ncols + (len(cat_columns) % ncols != 0)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 15))
        for idx in range(len(cat_columns)):
            ax = axs[idx] if len(axs.shape) == 1 else axs[idx // ncols, idx % ncols]
            sns.countplot(data=df, x=cat_columns[idx], hue=hue,
                          ax=ax, palette="Set2")
            ax.tick_params(axis='x', rotation=70)
        fig.savefig(SAVE_PATH + self.prefix + "categorical.png")
        fig.show()
    
    
    def explore_data(self) -> None:
        """
        :param df: A pd.DataFrame containing all the data.
                   The DataFrame needs to have a column named `label`
                   containing all the categorical labels.
    
        :param num_columns: The list of all the numerical columns
                            in the df DataFrame.
    
        :param cat_columns: The list of all the categorical columns
                            inf the df DataFrame
    
        :return: None, plots the data analysis
        """
        self.label_count_plot("label")
        self.numerical_analysis("label")
        self.categorical_analysis("label")
