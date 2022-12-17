#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# %% Class to Generate Dummy Data
# ----------------------------------------------------------------------------
class GenerateDate:
    def __init__(self, n_sample: int, n_analyst: int,
                 n_manager: int, f_name: str = None):
        self.df = self._create_data(n_sample, n_analyst, n_manager)

    def _create_data(self, n_sample: int, n_analyst: int, n_manager: int):
        lob: List[str] = ['AML', 'Risk', 'Personal', 'Commercial', 'IT', 'HR']
        activity: List[str] = ['Test', 'Review', 'QA', 'Ad-Hoc', 'Other']
        product: List[str] = ['Deposit', 'Insurance', 'Operational',
                              'Technology', 'Info Sec', 'Governance',
                              'Reputational']
        rating: List[str] = ['Critical', 'High', 'Medium', 'Low']

        df: pd.DataFrame = pd.DataFrame({
            'LOB': [np.random.choice(lob, p=[0.1, 0.1, 0.4, 0.1, 0.2, 0.1])
                    for i in range(n_sample)],
            'Activity':[np.random.choice(activity, p=[0.4, 0.3, 0.1, 0.1, 0.1])
                    for i in range(n_sample)],
            'Product':[np.random.choice(product,
                                        p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1])
                    for i in range(n_sample)],
            'Rating':[np.random.choice(rating, p=[0.1, 0.2, 0.3, 0.4])
                    for i in range(n_sample)],
            'Number_of_Analysts': [n_analyst] * n_sample,
            'Number_of_Managers': [n_manager] * n_sample
        })

        df['Rank']: pd.Series = df.reset_index(drop=False).groupby(
            ['LOB', 'Product', 'Activity']
        )['index'].rank(method='first', ascending=True)

        df['Activity_Name']: pd.Series = \
            df['LOB'] + ' - ' + df['Product'] + ': ' + df['Activity'] + \
                'Activity #' + df['Rank'].astype(int).astype(str)
        return df

    def get_df(self):
        return self.df

# ----------------------------------------------------------------------------
# %% Data Class for Activity Types
# ----------------------------------------------------------------------------
@dataclass
class Activity:
    lob: str
    product: str
    activity: str
    rating: str
    rating_num: int
    hours_required: float
    activity_name: str = None
    analyst_hours_required: float = 0
    manager_hours_required: float = 0

    def __post_init__(self):
        self._create_activity_name()
        self._compute_resource_hours_required()

    def _create_activity_name(self):
        self.activity_name = \
            f'{self.lob} - {self.product}: {self.activity}'

    def _compute_resource_hours_required(self):
        self.analyst_hours_required = self.hours_required * 0.75
        self.manager_hours_required = self.hours_required * 0.25

@dataclass
class Test(Activity):
    activity: str = 'Activity'
    rating: str = 'Critical'
    rating_num: int = 1
    hours_required: float = 200

@dataclass
class Review(Activity):
    activity: str = 'Review'
    rating: str = 'High'
    rating_num: int = 2
    hours_required: float = 150

@dataclass
class QA(Activity):
    activity: str = 'QA'
    rating: str = 'Medium'
    rating_num: int = 3
    hours_required: float = 40

@dataclass
class AdHoc(Activity):
    activity: str = 'Ad-Hoc'
    rating: str = 'Low'
    rating_num: int = 1
    hours_required: float = 20

@dataclass
class Other(Activity):
    activity: str = 'Other'
    rating: str = 'Low'
    rating_num: int = 1
    hours_required: float = 25

# ---------------------------------------------------------------------------
# %% Capacity Model
# ----------------------------------------------------------------------------
class CapacityModel:
    def __init__(self, data: pd.DataFrame, n_analyst: int,
                 n_manager: int, f_name: str=None):
        self.f_name: str = f_name
        self.df: pd.DataFrame = data
        self.df_final: pd.DataFrame = None
        self.total_analysts, self.total_managers = \
            self._compute_available_resources(n_analyst, n_manager)

    def _compute_available_resources(self, n_analyst: int, n_manager: int):
        total_analysts: int = 1760 * n_analyst
        total_managers: int = n_manager * 1760 * 0.25
        return total_analysts, total_managers

    def _process_data(self, df: pd.DataFrame):
        def compute_required_resources(df: pd.DataFrame) -> pd.DataFrame:
            # Calculate the analyst to manager
            df['Analyst_Hours_Required'] = df['Total_Hours_Required'] * (1-1/20)
            df['Manager_Hours_Required'] = df['Total_Hours_Required'] * (1/20)
            return df
        def compute_required_hours(activity, product):
            # Using product weight (complexity of testing effort depending on
            # product type) and hours required for each activity/task type,
            # calculate the total hours required to complete each activity
            activity_weight: Dict[str, int] = {
                'Test': 200, 'Review': 100, 'QA': 40,
                'Ad-Hoc': 20, 'Other': 10
            }
            product_weight: Dict[str, float] = {
                'Deposit': 1.0, 'Insurance': 1.0, 'Operational': 0.75,
                'Technology': 0.65, 'Info Sec': 0.5, 'Governance': 0.5,
                'Reputational': 0.25
            }
            return activity_weight[activity] * product_weight[product]

        df['Total_Hours_Required'] = df[['Activity', 'Product']].apply(
            lambda x: compute_required_hours(*x), axis=1)
        df['Rating_Number'] = df['Rating'].map({
            'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4
            })
        temp_df: pd.DataFrame = (compute_required_resources(df)
                   .sort_values(by=['Rating_Number', 'Total_Hours_Required'],
                                ascending=True)
                   .reset_index(drop=True))
        return temp_df


    def _compute_burndown(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the burndown of hours required for each activity by the
        # resources available
        def compute_burndown(required, available, label):
            total = np.insert(np.array(required), 0, available)
            series = pd.Series(np.subtract.accumulate(total, axis=0, out=total)[1:])
            series.name = label
            return series

        analyst_burndown = compute_burndown(
            df['Analyst_Hours_Required'], self.total_analysts, 'Analyst_Hours_Burndown'
        )
        director_burndown = compute_burndown(
            df['Manager_Hours_Required'], self.total_managers, 'Manager_Hours_Burndown'
        )
        df_final = pd.concat([df, analyst_burndown, director_burndown], axis=1)

        # Activity can only be completed if both analyst
        # and manager resources are available
        df_final['Can_Be_Completed'] = df_final.apply(
            lambda x: 'Yes' if (x['Analyst_Hours_Required'] >= 0) and
            (x['Manager_Hours_Required'] >= 0) else 'No', axis=1
        )
        return df_final

    def process(self):
        df: pd.DataFrame = deepcopy(self.df)
        df: pd.DataFrame = self._process_data(df)
        self.df_final = self._compute_burndown(df)
        return self

    def export_data(self):
        f_name: str = f'{self.f}.xlsx' if self.f is not None else 'sample_data.xlsx'
        self.df_final.to_excel(f_name)

    def get_df(self):
        return self.df_final

def compute_monthly_hours(n_resource: int, analyst: bool=True) -> float:
    fte_yearly_hours: float = 1760 * n_resource
    working_days: float = 260
    if not analyst:
        # Manager available at 1/4 of analyst capacity
        return round(fte_yearly_hours * 0.25 / (working_days/12), 2)
    return round(fte_yearly_hours / (working_days/12), 2)


def forecast(df: pd.DataFrame):
    total_analyst_hours_req: float = df['Analyst_Hours_Required'].sum()
    monthly_analyst_hours: float = total_analyst_hours_req/12

    total_manager_hours_req: float = df['Manager_Hours_Required'].sum()
    monthly_manager_hours: float = total_manager_hours_req/12

    one_analyst: float = compute_monthly_hours(1, True)
    one_manager: float = compute_monthly_hours(1, False)

    print(f'{monthly_analyst_hours//one_analyst} analysts are needed per month.')
    print(f'{monthly_manager_hours//one_manager} managers are needed per month.')


def main():
    print('\nStarting execution of model...\n')
    n_sample = 250
    n_analyst = 10
    n_manager = 2
    data = GenerateDate(n_sample, n_analyst, n_manager).get_df()
    df = CapacityModel(data, n_analyst, n_manager).process().get_df()
    forecast(df)
    print('\nModel ran successfully!')

if __name__ == '__main__':
    main()