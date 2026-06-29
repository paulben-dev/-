"""Tests for DualGAT graph builders, model, and predictor."""
import pytest
import torch
import numpy as np
import pandas as pd


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
        n = len(stocks)
        for node in range(n):
            has_self_loop = any(
                (edge_index[0, i] == node and edge_index[1, i] == node)
                for i in range(edge_index.shape[1])
            )
            assert has_self_loop, f"Node {node} missing self-loop"

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
        stocks = ["AAPL", "MSFT"]
        edge_index = builder.build(stocks, "2024-06-15", set())
        n = len(stocks)
        for node in range(n):
            has_self_loop = any(
                edge_index[0, i] == edge_index[1, i] == node
                for i in range(edge_index.shape[1])
            )
            assert has_self_loop, f"Node {node} missing self-loop"


class TestDualGATModel:
    """Unit tests for the DualGAT neural network."""

    @pytest.fixture
    def model(self):
        from src.model.dualgat import DualGATModel
        return DualGATModel(in_dim=3, hidden=64, out_dim=32, heads=4, dropout=0.2)

    @pytest.fixture
    def dummy_graph(self):
        """Simple 5-node fully connected graph."""
        n = 5
        sources, targets = [], []
        for i in range(n):
            sources.append(i)
            targets.append(i)  # self-loop
            for j in range(n):
                if i != j:
                    sources.append(i)
                    targets.append(j)
        return torch.tensor([sources, targets], dtype=torch.long)

    def test_forward_output_shape(self, model, dummy_graph):
        """forward() returns [N] predictions."""
        x = torch.randn(5, 3)
        out = model(x, dummy_graph, dummy_graph)
        assert out.shape == (5,)
        assert out.dtype == torch.float32

    def test_different_graphs_produce_different_outputs(self, model):
        """Different graph structures should change predictions."""
        x = torch.randn(3, 3)
        # Graph A: fully connected
        ga = torch.tensor([[0,0,0,1,1,1,2,2,2], [0,1,2,0,1,2,0,1,2]], dtype=torch.long)
        # Graph B: only self-loops (isolated)
        gb = torch.tensor([[0,1,2], [0,1,2]], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            out_a = model(x, ga, ga)
            out_b = model(x, gb, gb)
        # Different graph structure → different outputs (except by coincidence)
        assert not torch.allclose(out_a, out_b)

    def test_deterministic_in_eval_mode(self, model, dummy_graph):
        """Same input + same graph → same output in eval mode."""
        model.eval()
        x = torch.randn(5, 3)
        with torch.no_grad():
            out1 = model(x, dummy_graph, dummy_graph)
            out2 = model(x, dummy_graph, dummy_graph)
        assert torch.allclose(out1, out2)

    def test_dropout_active_in_train_mode(self, model, dummy_graph):
        """Dropout produces non-deterministic outputs during training."""
        model.train()
        x = torch.randn(10, 3)
        out1 = model(x, dummy_graph, dummy_graph)
        out2 = model(x, dummy_graph, dummy_graph)
        assert not torch.allclose(out1, out2)
