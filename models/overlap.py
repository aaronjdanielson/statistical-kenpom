"""Who has played whom: is one rating system for a whole league-season supported by games?

A ridge rating system puts every team of a league-season on one scale. Inside a group
of teams linked by games, the games set that scale. Between groups that never met
(qualifying groups, the early rounds of a cup) only the prior does: the model assumes
the groups are equally strong, and its posterior sd does not show that the assumption,
not a game, is doing the work. These statistics show it.

    groups(team, opp)        connected groups of teams
    pairs_met_share(...)     share of team pairs that have played each other
    weakest_split(...)       the two most weakly linked halves of the largest group
                             (spectral bisection) and the games between them
    low_overlap(...)         the league-season flag
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

#: a connected league-season is low-overlap when its two most weakly linked halves are
#: joined by fewer than this many games (the rating gap between the halves then has a
#: standard error above about 3 points per 100 possessions: ~17 / sqrt(30)) ...
MIN_BRIDGE_GAMES = 30
#: ... and those games are under this share of the smaller half's games (so a pair of
#: teams that mostly play everyone else is not a "half")
MAX_BRIDGE_SHARE = 0.20


def _matrix(team: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Teams (sorted) and the symmetric games-played matrix, one count per game.
    *team*, *opp*: one entry per game (not per team-game row)."""
    teams = np.unique(np.concatenate([team, opp]))
    a, b = np.searchsorted(teams, team), np.searchsorted(teams, opp)
    W = np.zeros((len(teams), len(teams)))
    np.add.at(W, (a, b), 1.0)
    np.add.at(W, (b, a), 1.0)
    return teams, W


def groups(team: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(teams, group index per team, number of groups). Group 0 is the largest; ties go
    to the group holding the smallest team id."""
    teams, W = _matrix(team, opp)
    n, lab = connected_components(sparse.csr_matrix(W), directed=False)
    size = np.bincount(lab, minlength=n)
    first = np.array([teams[lab == k].min() for k in range(n)])
    order = np.lexsort((first, -size))                 # largest first, then smallest team id
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return teams, rank[lab], int(n)


def pairs_met_share(team: np.ndarray, opp: np.ndarray) -> float:
    teams, W = _matrix(team, opp)
    T = len(teams)
    return float((W > 0).sum() / 2 / (T * (T - 1) / 2)) if T > 1 else 1.0


def weakest_split(team: np.ndarray, opp: np.ndarray) -> tuple[int, float]:
    """(bridge games, bridge share) for the largest group: split it in two by the sign of
    the Fiedler vector of the normalised Laplacian (the standard relaxation of the
    sparsest cut); bridge games = games between the halves; bridge share = those games
    as a share of all games involving the smaller half (its conductance)."""
    teams, lab, _ = groups(team, opp)
    keep = np.isin(team, teams[lab == 0])
    teams, W = _matrix(team[keep], opp[keep])
    T = len(teams)
    if T < 4:
        return int(W.sum() / 2), 1.0
    d = W.sum(1)
    L = np.eye(T) - W / np.sqrt(np.outer(d, d))
    _, vec = np.linalg.eigh(L)
    side = vec[:, 1] / np.sqrt(d) > 0
    if side.all() or not side.any():
        return int(W.sum() / 2), 1.0
    cross = W[np.ix_(side, ~side)].sum()
    inside = min(W[np.ix_(side, side)].sum(), W[np.ix_(~side, ~side)].sum()) / 2
    return int(cross), float(cross / (cross + inside))


def low_overlap(n_groups: int, bridge_games: int, bridge_share: float) -> bool:
    return bool(n_groups > 1 or (bridge_games < MIN_BRIDGE_GAMES and bridge_share < MAX_BRIDGE_SHARE))
