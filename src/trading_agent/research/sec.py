from __future__ import annotations

from functools import cached_property
from typing import Any

import httpx


class SecClient:
    def __init__(self, user_agent: str) -> None:
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    @cached_property
    def _ticker_map(self) -> dict[str, dict[str, Any]]:
        response = httpx.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=self.headers,
            timeout=30.0,
        )
        response.raise_for_status()
        rows = response.json().values()
        return {row["ticker"].upper(): row for row in rows}

    def get_cik(self, symbol: str) -> str | None:
        row = self._ticker_map.get(symbol.upper())
        if not row:
            return None
        return f"{int(row['cik_str']):010d}"

    async def get_company_summary(self, symbol: str) -> dict[str, Any]:
        cik = self.get_cik(symbol)
        if not cik:
            return {}

        async with httpx.AsyncClient(timeout=30.0, headers=self.headers) as client:
            submissions_response = await client.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json"
            )
            submissions_response.raise_for_status()
            submissions = submissions_response.json()

            facts_response = await client.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            )
            facts_response.raise_for_status()
            facts = facts_response.json()

        recent_forms = submissions.get("filings", {}).get("recent", {})
        revenue = self._latest_fact_value(
            facts,
            "us-gaap",
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        )
        net_income = self._latest_fact_value(facts, "us-gaap", ["NetIncomeLoss"])
        return {
            "cik": cik,
            "entity_name": facts.get("entityName") or submissions.get("name"),
            "latest_revenue": revenue,
            "latest_net_income": net_income,
            "recent_filings": [
                {
                    "form": form,
                    "filing_date": filing_date,
                    "accession_number": accession,
                }
                for form, filing_date, accession in zip(
                    recent_forms.get("form", [])[:5],
                    recent_forms.get("filingDate", [])[:5],
                    recent_forms.get("accessionNumber", [])[:5],
                    strict=False,
                )
            ],
        }

    def _latest_fact_value(
        self,
        facts: dict[str, Any],
        taxonomy: str,
        concepts: list[str],
    ) -> dict[str, Any] | None:
        taxonomy_facts = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            units = taxonomy_facts.get(concept, {}).get("units", {})
            usd_rows = units.get("USD", [])
            if usd_rows:
                latest = sorted(
                    usd_rows,
                    key=lambda row: (row.get("end") or "", row.get("filed") or ""),
                    reverse=True,
                )[0]
                return {
                    "concept": concept,
                    "value": latest.get("val"),
                    "period_end": latest.get("end"),
                    "filed": latest.get("filed"),
                    "form": latest.get("form"),
                }
        return None
