"""Arena: pit two agents against each other and watch or tally the games.

    python arena.py --p1 dqn:checkpoint/best_model.pth --p2 minimax:4 --games 100
    python arena.py --p1 dqn:checkpoint/best_model.pth --p2 random --render gui
    python arena.py --p1 minimax:2 --p2 minimax:5 --games 6 --render text --delay 0.3

An agent is named by a spec string:

    random              uniformly random legal move
    minimax[:depth]     the alpha-beta searcher from minimax.py (default depth 4)
    dqn:<checkpoint>    greedy policy from a saved Connect4Model checkpoint
    <path>.pth          shorthand for dqn:<path>

Players only need `select_action(state, env, action_mask) -> int`, the same
interface `eval.py` uses, so anything that works there works here.

Seats alternate every game -- moving first is a real advantage in Connect 4, so
an even number of games gives both agents the same number of opens.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

import environment
import minimax

try:
    import tkinter as tk
except ImportError:  # headless box without Tk: --render gui is unavailable
    tk = None


# --------------------------------------------------------------- the players


class RandomPlayer:
    """Uniformly random legal move."""

    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, env, action_mask):
        return int(self.rng.choice(np.flatnonzero(np.asarray(action_mask))))


def build_player(spec, device=None, seed=None):
    """Turn a spec string into something with `select_action` and a `name`."""
    kind, _, arg = spec.partition(":")
    kind = kind.lower()

    if kind in ("random", "rand"):
        return RandomPlayer(seed=seed)

    if kind == "minimax":
        depth = int(arg) if arg else minimax.DEPTH
        player = minimax.MinimaxAgent(depth=depth)
        player.name = f"minimax-{depth}"
        return player

    if kind in ("dqn", "agent"):
        if not arg:
            raise SystemExit(f"'{spec}' needs a checkpoint, e.g. dqn:checkpoint/best_model.pth")
        return _load_dqn(arg, device)

    if spec.endswith((".pth", ".pt")):  # a bare path means a DQN checkpoint
        return _load_dqn(spec, device)

    raise SystemExit(f"unknown agent spec '{spec}' (expected random, minimax[:depth] or dqn:<path>)")


def _load_dqn(path, device=None):
    # Imported lazily so random-vs-minimax runs need no torch.
    from agent import Agent

    if not Path(path).exists():
        raise SystemExit(f"checkpoint not found: {path}")

    player = Agent(inference_only=True, device=device)
    player.load(path)
    player.epsilon = 0.0  # greedy: the arena measures the policy, not exploration
    player.name = Path(path).stem
    return player


# ------------------------------------------------------------------- scoring


class Results:
    """Running tally of an arena run, from player 1's point of view."""

    def __init__(self, names):
        self.names = names
        self.wins = [0, 0]
        self.draws = 0
        self.first_games = [0, 0]   # games each agent opened
        self.first_wins = [0, 0]    # ... and won while opening
        self.illegal = [0, 0]       # games thrown away on an illegal move
        self.lengths = []

    @property
    def games(self):
        return self.wins[0] + self.wins[1] + self.draws

    def record(self, info, p1_player):
        """`info` is the env's final info dict; `p1_player` is p1's seat (1/-1)."""
        winner = info["winner"]
        opener = 0 if p1_player == environment.PLAYER else 1
        self.first_games[opener] += 1

        if winner == 0:
            self.draws += 1
        else:
            idx = 0 if winner == p1_player else 1
            self.wins[idx] += 1
            if idx == opener:
                self.first_wins[idx] += 1
            if info.get("invalid_action"):
                self.illegal[1 - idx] += 1

        self.lengths.append(info["move_count"])

    def scoreline(self):
        return (
            f"{self.names[0]} {self.wins[0]} – {self.wins[1]} {self.names[1]}"
            f"  (draws {self.draws})"
        )

    def summary(self):
        games = max(self.games, 1)
        width = max(len(name) for name in self.names)
        lines = [
            f"=== {self.games} games: {self.names[0]} vs {self.names[1]} ===",
        ]
        for idx, name in enumerate(self.names):
            second_games = self.games - self.first_games[idx]
            second_wins = self.wins[idx] - self.first_wins[idx]
            lines.append(
                f"{name:<{width}}  {self.wins[idx]:>4} wins ({self.wins[idx] / games:6.1%})"
                f"   first {self.first_wins[idx]}/{self.first_games[idx]}"
                f"   second {second_wins}/{second_games}"
            )
        lines.append(f"{'draws':<{width}}  {self.draws:>4}      ({self.draws / games:6.1%})")
        if self.lengths:
            lines.append(f"average game length: {sum(self.lengths) / len(self.lengths):.1f} moves")
        for idx, name in enumerate(self.names):
            if self.illegal[idx]:
                lines.append(f"{name} forfeited {self.illegal[idx]} game(s) on an illegal move")
        return "\n".join(lines)


# ----------------------------------------------------------------- game loop


def play_game(env, players, p1_player, on_move=None):
    """Play one full game; return the env's final info dict.

    `p1_player` is which seat (1 or -1) players[0] takes. `on_move` is called as
    `on_move(env, mover_idx, action, info)` after every move.
    """
    state, info = env.reset()

    while True:
        mover_idx = 0 if env.current_player == p1_player else 1
        action = players[mover_idx].select_action(state, env, info["action_mask"])
        state, _reward, terminated, truncated, info = env.step(action)

        if on_move is not None:
            on_move(env, mover_idx, action, info)

        if terminated or truncated:
            return info


def seat_for_game(game_index, swap=True):
    """Seat for player 1 in game `game_index`; alternate so both open equally."""
    if not swap:
        return environment.PLAYER
    return environment.PLAYER if game_index % 2 == 0 else environment.OPPONENT


def run_arena(players, games=100, render="none", delay=0.0, swap=True, seed=None,
              progress_every=0):
    """Play `games` games and return the `Results`.

    render: "none" (tally only) or "text" (print the board after every move).
    """
    env = environment.Connect4Env(seed=seed)
    results = Results([players[0].name, players[1].name])

    for game in range(games):
        p1_player = seat_for_game(game, swap)

        on_move = None
        if render == "text":
            print(f"\n--- game {game + 1}/{games} · "
                  f"{players[0].name if p1_player == environment.PLAYER else players[1].name}"
                  f" opens ---")

            def on_move(env, mover_idx, action, info, _p1=p1_player):
                print(f"\n{players[mover_idx].name} plays column {action + 1}")
                print(render_text(env, _p1, [p.name for p in players]))
                if delay:
                    time.sleep(delay)

        info = play_game(env, players, p1_player, on_move=on_move)
        results.record(info, p1_player)

        if render == "text":
            winner = info["winner"]
            if winner == 0:
                outcome = "draw"
            else:
                outcome = f"{players[0 if winner == p1_player else 1].name} wins"
            if info.get("invalid_action"):
                outcome += " (opponent played an illegal column)"
            print(f"\n>>> {outcome} in {info['move_count']} moves · {results.scoreline()}")
        elif progress_every and (game + 1) % progress_every == 0:
            print(f"[{game + 1}/{games}] {results.scoreline()}", flush=True)

    return results


# ------------------------------------------------------------ text rendering

_TEXT_PIECES = ("O", "X")  # players[0], players[1]


def render_text(env, p1_player, names):
    """ASCII board, labelled with which symbol belongs to which agent."""
    rows = []
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            piece = env.board[row, col]
            if piece == 0:
                cells.append(".")
            else:
                cells.append(_TEXT_PIECES[0 if piece == p1_player else 1])
        rows.append("| " + " ".join(cells) + " |")

    header = "  " + " ".join(str(c + 1) for c in range(env.cols))
    legend = f"  {_TEXT_PIECES[0]} = {names[0]}   {_TEXT_PIECES[1]} = {names[1]}"
    return "\n".join([header, *rows, header, legend])


# ------------------------------------------------------------- gui rendering

CELL = 76
DISC = 29
MARGIN = 14
HEADER = 8

COLOR_BG = "#111827"
COLOR_BOARD = "#1d4ed8"
COLOR_EMPTY = "#0b1220"
COLOR_P1 = "#f43f5e"
COLOR_P2 = "#facc15"
COLOR_TEXT = "#e5e7eb"
COLOR_MUTED = "#94a3b8"
COLOR_WIN = "#ffffff"

GAME_BREAK_MS = 1200  # pause on the final position before the next game starts


class ArenaGUI:
    """Watch two agents play a match, one move per tick."""

    def __init__(self, players, games, delay_ms=450, swap=True, seed=None):
        self.players = players
        self.games = games
        self.delay_ms = delay_ms
        self.swap = swap

        self.env = environment.Connect4Env(seed=seed)
        self.results = Results([players[0].name, players[1].name])
        self.game_index = 0
        self.p1_player = seat_for_game(0, swap)
        self.win_line = ()
        self.paused = False
        self.finished = False
        self._after_id = None

        self.root = tk.Tk()
        self.root.title(f"Arena · {players[0].name} vs {players[1].name}")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        width = self.env.cols * CELL + 2 * MARGIN
        height = self.env.rows * CELL + 2 * MARGIN

        self.title = tk.Label(
            self.root, text="", font=("Helvetica", 15, "bold"),
            fg=COLOR_TEXT, bg=COLOR_BG,
        )
        self.title.pack(pady=(10, HEADER))

        self.canvas = tk.Canvas(
            self.root, width=width, height=height,
            bg=COLOR_BG, highlightthickness=0,
        )
        self.canvas.pack(padx=10)

        self.status = tk.Label(
            self.root, text="", font=("Helvetica", 13, "bold"),
            fg=COLOR_TEXT, bg=COLOR_BG,
        )
        self.status.pack(pady=(8, 0))

        self.subtitle = tk.Label(
            self.root, text="", font=("Helvetica", 11),
            fg=COLOR_MUTED, bg=COLOR_BG,
        )
        self.subtitle.pack()

        controls = tk.Frame(self.root, bg=COLOR_BG)
        controls.pack(pady=10)
        self.pause_button = tk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=4)
        tk.Button(controls, text="Skip game", command=self.skip_game).pack(side="left", padx=4)
        tk.Button(controls, text="Quit", command=self.root.destroy).pack(side="left", padx=4)

        self.root.bind("<space>", lambda _e: self.toggle_pause())
        self.root.bind("<q>", lambda _e: self.root.destroy())

        self.start_game()

    # ------------------------------------------------------------ game flow

    def run(self):
        self.root.mainloop()
        return self.results

    def start_game(self):
        self.env.reset()
        self.p1_player = seat_for_game(self.game_index, self.swap)
        self.win_line = ()
        self.draw()
        self._update_labels()
        self._schedule(self.delay_ms)

    def step(self):
        self._after_id = None
        if self.finished or self.paused:
            return

        state = self.env._observation()
        mover_idx = 0 if self.env.current_player == self.p1_player else 1
        action = self.players[mover_idx].select_action(state, self.env, self.env.action_mask())
        _state, _reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self._end_game(info)
        else:
            self.draw()
            self._update_labels()
            self._schedule(self.delay_ms)

    def _end_game(self, info):
        if info["winner"] != 0 and not info.get("invalid_action"):
            self.win_line = _winning_cells(self.env.board, *info["last_move"]) or ()
        self.results.record(info, self.p1_player)
        self.draw()
        self._announce(info)

        self.game_index += 1
        if self.game_index >= self.games:
            self.finished = True
            self.pause_button.configure(state="disabled")
        else:
            self._schedule(GAME_BREAK_MS, self.start_game)

    def skip_game(self):
        """Play the current game out instantly and move on."""
        if self.finished:
            return
        self._cancel()
        while not self.env.terminated:
            state = self.env._observation()
            mover_idx = 0 if self.env.current_player == self.p1_player else 1
            action = self.players[mover_idx].select_action(
                state, self.env, self.env.action_mask()
            )
            _state, _reward, terminated, _truncated, info = self.env.step(action)
            if terminated:
                self._end_game(info)
                return

    def toggle_pause(self):
        if self.finished:
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        if self.paused:
            self._cancel()
            self._update_labels()
        else:
            self._schedule(self.delay_ms)

    def _schedule(self, delay_ms, callback=None):
        self._cancel()
        if self.finished or self.paused:
            return
        self._after_id = self.root.after(delay_ms, callback or self.step)

    def _cancel(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    # ---------------------------------------------------------------- text

    def _color(self, player):
        return COLOR_P1 if player == self.p1_player else COLOR_P2

    def _update_labels(self):
        opener = 0 if self.p1_player == environment.PLAYER else 1
        self.title.configure(
            text=f"game {min(self.game_index + 1, self.games)}/{self.games}"
                 f" · {self.players[opener].name} opens"
        )
        if self.paused:
            self.status.configure(text="paused", fg=COLOR_MUTED)
        else:
            mover_idx = 0 if self.env.current_player == self.p1_player else 1
            color = COLOR_P1 if mover_idx == 0 else COLOR_P2
            self.status.configure(text=f"{self.players[mover_idx].name} to move", fg=color)
        self.subtitle.configure(text=self.results.scoreline())

    def _announce(self, info):
        winner = info["winner"]
        if winner == 0:
            self.status.configure(text="draw", fg=COLOR_TEXT)
        else:
            idx = 0 if winner == self.p1_player else 1
            self.status.configure(
                text=f"{self.players[idx].name} wins",
                fg=COLOR_P1 if idx == 0 else COLOR_P2,
            )

        note = f"{info['move_count']} moves"
        if info.get("invalid_action"):
            note = "illegal move — forfeited · " + note
        if self.game_index + 1 >= self.games:
            note = "match over · " + note
        self.subtitle.configure(text=f"{note} · {self.results.scoreline()}")

    # ------------------------------------------------------------- drawing

    def draw(self):
        c = self.canvas
        c.delete("all")
        c.create_rectangle(
            MARGIN, MARGIN,
            MARGIN + self.env.cols * CELL, MARGIN + self.env.rows * CELL,
            fill=COLOR_BOARD, outline="",
        )

        wins = set(self.win_line)
        last = self.env.last_move
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                x = MARGIN + col * CELL + CELL / 2
                y = MARGIN + row * CELL + CELL / 2
                piece = self.env.board[row, col]
                fill = COLOR_EMPTY if piece == 0 else self._color(piece)

                outline, width = "", 0
                if (row, col) in wins:
                    outline, width = COLOR_WIN, 4
                elif last == (row, col):
                    outline, width = COLOR_TEXT, 2

                c.create_oval(
                    x - DISC, y - DISC, x + DISC, y + DISC,
                    fill=fill, outline=outline, width=width,
                )


def _winning_cells(board, row, col):
    from play import winning_cells  # shared with the human-play window

    return winning_cells(board, row, col)


# ---------------------------------------------------------------------- cli


def main():
    parser = argparse.ArgumentParser(
        description="Play two agents against each other, with or without a board to watch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "agent specs:\n"
            "  random               uniformly random legal move\n"
            "  minimax[:depth]      alpha-beta search (default depth "
            f"{minimax.DEPTH})\n"
            "  dqn:<checkpoint>     greedy policy from a saved checkpoint\n"
            "  <path>.pth           shorthand for dqn:<path>\n"
        ),
    )
    parser.add_argument("--p1", required=True, help="first agent spec")
    parser.add_argument("--p2", required=True, help="second agent spec")
    parser.add_argument("-n", "--games", type=int, default=100, help="games to play")
    parser.add_argument(
        "--render", choices=["none", "text", "gui"], default="none",
        help="none: results only (default); text: boards in the terminal; gui: a window",
    )
    parser.add_argument(
        "--delay", type=float, default=0.45,
        help="seconds between moves when rendering (default 0.45)",
    )
    parser.add_argument(
        "--no-swap", action="store_true",
        help="keep p1 opening every game instead of alternating seats",
    )
    parser.add_argument("--device", default=None, help="torch device for dqn agents")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--progress-every", type=int, default=0,
        help="print the running score every N games (render=none only)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    players = [
        build_player(args.p1, device=args.device, seed=args.seed),
        build_player(args.p2, device=args.device, seed=None if args.seed is None else args.seed + 1),
    ]
    if players[0].name == players[1].name:
        players[0].name += " (1)"
        players[1].name += " (2)"

    if args.render == "gui":
        if tk is None:
            raise SystemExit("tkinter is not available; use --render text or --render none")
        gui = ArenaGUI(
            players, games=args.games, delay_ms=int(args.delay * 1000),
            swap=not args.no_swap, seed=args.seed,
        )
        results = gui.run()
        if results.games:
            print(results.summary())
        return

    results = run_arena(
        players, games=args.games, render=args.render, delay=args.delay,
        swap=not args.no_swap, seed=args.seed, progress_every=args.progress_every,
    )
    print()
    print(results.summary())


if __name__ == "__main__":
    main()
