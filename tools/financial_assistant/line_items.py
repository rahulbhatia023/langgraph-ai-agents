from datetime import date
from enum import Enum
from typing import Dict, Union, List, Optional, Annotated

import requests
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation

FINANCIAL_DATASETS_BASE_URL = "https://api.financialdatasets.ai/"


class LineItem(str, Enum):
    consolidated_income = "consolidated_income"
    cost_of_revenue = "cost_of_revenue"
    dividends_per_common_share = "dividends_per_common_share"
    earnings_per_share = "earnings_per_share"
    earnings_per_share_diluted = "earnings_per_share_diluted"
    ebit = "ebit"
    ebit_usd = "ebit_usd"
    earnings_per_share_usd = "earnings_per_share_usd"
    gross_profit = "gross_profit"
    income_tax_expense = "income_tax_expense"
    interest_expense = "interest_expense"
    net_income = "net_income"
    net_income_common_stock = "net_income_common_stock"
    net_income_common_stock_usd = "net_income_common_stock_usd"
    net_income_discontinued_operations = "net_income_discontinued_operations"
    net_income_non_controlling_interests = "net_income_non_controlling_interests"
    operating_expense = "operating_expense"
    operating_income = "operating_income"
    preferred_dividends_impact = "preferred_dividends_impact"
    research_and_development = "research_and_development"
    revenue = "revenue"
    revenue_usd = "revenue_usd"
    selling_general_and_administrative_expenses = (
        "selling_general_and_administrative_expenses"
    )
    weighted_average_shares = "weighted_average_shares"
    weighted_average_shares_diluted = "weighted_average_shares_diluted"

    # Balance Sheet fields
    accumulated_other_comprehensive_income = "accumulated_other_comprehensive_income"
    cash_and_equivalents = "cash_and_equivalents"
    cash_and_equivalents_usd = "cash_and_equivalents_usd"
    current_assets = "current_assets"
    current_debt = "current_debt"
    current_investments = "current_investments"
    current_liabilities = "current_liabilities"
    deferred_revenue = "deferred_revenue"
    deposit_liabilities = "deposit_liabilities"
    goodwill_and_intangible_assets = "goodwill_and_intangible_assets"
    inventory = "inventory"
    investments = "investments"
    non_current_assets = "non_current_assets"
    non_current_debt = "non_current_debt"
    non_current_investments = "non_current_investments"
    non_current_liabilities = "non_current_liabilities"
    outstanding_shares = "outstanding_shares"
    property_plant_and_equipment = "property_plant_and_equipment"
    retained_earnings = "retained_earnings"
    shareholders_equity = "shareholders_equity"
    shareholders_equity_usd = "shareholders_equity_usd"
    tax_assets = "tax_assets"
    tax_liabilities = "tax_liabilities"
    total_assets = "total_assets"
    total_debt = "total_debt"
    total_debt_usd = "total_debt_usd"
    total_liabilities = "total_liabilities"
    trade_and_non_trade_payables = "trade_and_non_trade_payables"
    trade_and_non_trade_receivables = "trade_and_non_trade_receivables"

    # Cash Flow Statement fields
    business_acquisitions_and_disposals = "business_acquisitions_and_disposals"
    capital_expenditure = "capital_expenditure"
    change_in_cash_and_equivalents = "change_in_cash_and_equivalents"
    depreciation_and_amortization = "depreciation_and_amortization"
    dividends_and_other_cash_distributions = "dividends_and_other_cash_distributions"
    effect_of_exchange_rate_changes = "effect_of_exchange_rate_changes"
    investment_acquisitions_and_disposals = "investment_acquisitions_and_disposals"
    issuance_or_purchase_of_equity_shares = "issuance_or_purchase_of_equity_shares"
    issuance_or_repayment_of_debt_securities = (
        "issuance_or_repayment_of_debt_securities"
    )
    net_cash_flow_from_financing = "net_cash_flow_from_financing"
    net_cash_flow_from_investing = "net_cash_flow_from_investing"
    net_cash_flow_from_operations = "net_cash_flow_from_operations"
    share_based_compensation = "share_based_compensation"


class SearchLineItems(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers to search for.")
    line_items: List[LineItem] = Field(
        ..., description="List of financial line items to retrieve."
    )
    period: str = Field(
        default="ttm",
        description="The time period for the financial data. Valid values are 'annual', 'quarterly', or 'ttm' (trailing twelve months).",
    )
    limit: int = Field(
        default=1,
        description="The maximum number of results to return per ticker. Must be a positive integer.",
    )
    start_date: Optional[date] = Field(
        None, description="The start date for the financial data in YYYY-MM-DD format."
    )
    end_date: Optional[date] = Field(
        None, description="The end date for the financial data in YYYY-MM-DD format."
    )


class SearchLineItemsTool(BaseTool):
    financial_datasets_api_key: str = Field(
        ..., description="Financial Datasets API key."
    )
    name: str = "search-line-items"
    description: str = (
        "Search for financial line items across multiple tickers and time periods."
    )
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = SearchLineItems
    return_direct: bool = True

    def _run(
        self,
        tickers: List[str],
        line_items: List[str],
        period: str = "ttm",
        limit: int = 1,
        start_date: str = None,
        end_date: str = None,
    ) -> Union[Dict, str]:
        url = f"{FINANCIAL_DATASETS_BASE_URL}financials/search/line-items"

        payload = {
            "tickers": tickers,
            "line_items": line_items,
            "period": period,
            "limit": limit,
        }

        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date

        try:
            response = requests.post(
                url,
                json=payload,
                headers={
                    "X-API-Key": self.financial_datasets_api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            return {"search_results": [], "error": str(e)}
