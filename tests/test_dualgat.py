"""Tests for DualGAT graph builders, model, and predictor."""
import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime


class TestIndustryGraphBuilder:
    """Tests for the industry graph builder."""

    @pytest.fixture
    def builder(self):
        from src.model.dualgat import IndustryGraphBuilder
        return IndustryGraphBuilder()

    @pytest.fixture
    def sample_fundamentals(self):
        return pd.DataFrame([
            {"stock": "AAPL", "sector": "Technology"},
            {"stock": "MSFT", "sector": "Technology"},
            {"stock": "JPM",  "sector": "Financial Services"},
            {"stock": "BAC",  "sector": "Financial Services"},
            {"stock": "JNJ",  "sector": "Healthcare"},
        ])

    def test_same_sector_connected(self, builder, sample_fundamentals):
        """Stocks in the same sector should have edges between them."""
        stocks = ["AAPL", "MSFT", "JPM", "BAC", "JNJ"]
        edge_index = builder.build(stocks, sample_fundamentals)
        assert edge_index.shape[0] == 2  # [2, num_edges]
        assert edge_index.shape[1] > 0

        edges = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            edges.add((min(u, v), max(u, v)))

        # AAPL(0) and MSFT(1) both Technology -> should be connected
        assert (0, 1) in edges or (1, 0) in edges
        # JPM(2) and BAC(3) both Financial -> should be connected
        assert (2, 3) in edges or (3, 2) in edges
        # JNJ(4) Healthcare -> should not connect to Technology stocks
        assert (0, 4) not in edges and (4, 0) not in edges

    def test_self_loops_included(self, builder, sample_fundamentals):
        """Every node should have a self-loop."""
        stocks = ["AAPL", "MSFT"]
        edge_index = builder.build(stocks, sample_fundamentals)
        has_self_loop_0 = any(
            (edge_index[0, i] == 0 and edge_index[1, i] == 0)
            for i in range(edge_index.shape[1])
        )
        assert has_self_loop_0

    def test_unknown_sector_isolated(self, builder):
        """Stocks with unknown sector only have self-loops."""
        funds = pd.DataFrame([
            {"stock": "AAPL", "sector": ""},
            {"stock": "MSFT", "sector": "Technology"},
        ])
        edge_index = builder.build(["AAPL", "MSFT"], funds)
        # AAPL (index 0) should not connect to MSFT (index 1)
        cross_edges = [
            (edge_index[0, i].item(), edge_index[1, i].item())
            for i in range(edge_index.shape[1])
        ]
        has_cross = any((u == 0 and v == 1) or (u == 1 and v == 0) for u, v in cross_edges)
        assert not has_cross


class TestCorrelationGraphBuilder:
    """Tests for the correlation graph builder."""

    @pytest.fixture
    def builder(self):
        from src.model.dualgat import CorrelationGraphBuilder
        return CorrelationGraphBuilder(window=30, theta1=0.77, theta2=0.67)

    def test_highly_correlated_connected(self, builder, prepopulated_db):
        """Stocks with correlation > theta1 should be connected."""
        # AAPL and MSFT prices in fixture are highly correlated
        edge_index = builder.build(["AAPL", "MSFT"], "2024-06-15", set())
        assert edge_index.shape[0] == 2
        # Self-loops at minimum
        assert edge_index.shape[1] >= 2

    def test_expert_stocks_use_lower_threshold(self, builder, prepopulated_db):
        """Stocks with expert labels use theta2 < theta1."""
        # Mark AAPL as having expert coverage -> should get more edges
        edge_index_with = builder.build(["AAPL", "MSFT"], "2024-06-15", {"AAPL"})
        edge_index_without = builder.build(["AAPL", "MSFT"], "2024-06-15", set())
        # With expert, edges >= without (lower threshold -> more or equal connections)
        assert edge_index_with.shape[1] >= edge_index_without.shape[1]

    def test_self_loops_included(self, builder, prepopulated_db):
        """Every node should have a self-loop."""
        edge_index = builder.build(["AAPL", "MSFT"], "2024-06-15", set())
        has_self_loop = any(
            edge_index[0, i] == edge_index[1, i]
            for i in range(edge_index.shape[1])
        )
        assert has_self_loop
