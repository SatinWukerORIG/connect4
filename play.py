"""Play Connect 4 against an agent in a window.

    python play.py                       # play the random agent
    python play.py --second              # let the agent open
    python play.py --agent dqn --checkpoint checkpoints/dqn.pt

Agents only need an `act(obs, action_mask) -> int` method. `obs` is the
(2, 6, 7) plane stack from the env, already flipped to the mover's perspective,
so the same agent plays either seat. Add a trained policy by pointing
--checkpoint at its weights.
"""

import argparse
import tkinter as tk

import numpy as np

from environment import Connect4Env, _DIRECTIONS

CELL = 88          # pixel size of one board square
DISC = 34          # disc radius
CHUTE = 70         # height of the hover strip above the board
MARGIN = 14

COLOR_BG = "#111827"
COLOR_BOARD = "#1d4ed8"
COLOR_EMPTY = "#0b1220"
COLOR_HUMAN = "#f43f5e"
COLOR_AGENT = "#facc15"
COLOR_TEXT = "#e5e7eb"
COLOR_MUTED = "#94a3b8"
COLOR_WIN = "#ffffff"

AGENT_DELAY_MS = 450   # pause before the agent replies, so moves are readable


class RandomAgent:
    """Uniformly random legal move."""

    name = "Random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, action_mask):
        return int(self.rng.choice(np.flatnonzero(action_mask)))


class DQNAgent:
    """Greedy policy from a trained Connect4Model checkpoint.

    Accepts a bare state_dict or a training checkpoint that wraps one -- Agent.save
    stores it under "model", other loops use "model_state_dict" or "state_dict".
    """

    name = "DQN"

    WEIGHT_KEYS = ("model", "model_state_dict", "state_dict")

    def __init__(self, checkpoint, device="cpu"):
        import torch  # imported lazily so the random agent needs no torch

        from model import Connect4Model

        self.torch = torch
        self.device = torch.device(device)
        self.model = Connect4Model().to(self.device)

        blob = torch.load(checkpoint, map_location=self.device)
        if isinstance(blob, dict):
            for key in self.WEIGHT_KEYS:
                if key in blob:
                    blob = blob[key]
                    break
        self.model.load_state_dict(blob)
        self.model.eval()

    def act(self, obs, action_mask):
        torch = self.torch
        with torch.no_grad():
            state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q = self.model(state.unsqueeze(0)).squeeze(0)
            mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask, float("-inf"))
            return int(torch.argmax(q).item())


def winning_cells(board, row, col):
    """Cells of the line through (row, col), or None if it isn't a win."""
    rows, cols = board.shape
    player = board[row, col]
    if player == 0:
        return None
    for dr, dc in _DIRECTIONS:
        line = [(row, col)]
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                line.append((r, c))
                r += sign * dr
                c += sign * dc
        if len(line) >= 4:
            return line
    return None


class Connect4GUI:
    def __init__(self, agent, human_first=True, seed=None):
        self.env = Connect4Env(seed=seed)
        self.agent = agent
        self.human_player = 1 if human_first else -1

        self.game_over = False
        self.busy = False          # agent's move is pending; ignore clicks
        self.hover_col = None
        self.win_line = ()
        self.score = {"you": 0, "agent": 0, "draw": 0}

        self.root = tk.Tk()
        self.root.title("Connect 4")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        width = self.env.cols * CELL + 2 * MARGIN
        height = self.env.rows * CELL + CHUTE + 2 * MARGIN

        self.canvas = tk.Canvas(
            self.root, width=width, height=height,
            bg=COLOR_BG, highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=(10, 0))
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(None))

        self.status = tk.Label(
            self.root, text="", font=("Helvetica", 15, "bold"),
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
        tk.Button(controls, text="New game", command=self.new_game).pack(side="left", padx=4)
        tk.Button(controls, text="Swap who starts", command=self._swap_sides).pack(side="left", padx=4)
        tk.Button(controls, text="Quit", command=self.root.destroy).pack(side="left", padx=4)

        self.root.bind("<n>", lambda _e: self.new_game())
        self.root.bind("<q>", lambda _e: self.root.destroy())
        for col in range(self.env.cols):
            self.root.bind(str(col + 1), lambda _e, c=col: self._human_move(c))

        self.new_game()

    # ------------------------------------------------------------- game flow

    def run(self):
        self.root.mainloop()

    def new_game(self):
        self.env.reset()
        self.game_over = False
        self.busy = False
        self.win_line = ()
        self.draw()
        self._schedule_agent()

    def _swap_sides(self):
        self.human_player = -self.human_player
        self.new_game()

    def _human_move(self, col):
        if self.game_over or self.busy:
            return
        if self.env.current_player != self.human_player:
            return
        if not self.env.is_valid_action(col):
            self.subtitle.configure(text="That column is full.")
            return
        self._play(col)

    def _agent_move(self):
        self.busy = False
        if self.game_over or self.env.current_player == self.human_player:
            return
        obs = self.env._observation()
        col = self.agent.act(obs, self.env.action_mask())
        if not self.env.is_valid_action(col):
            # A broken policy forfeits rather than crashing the window.
            self.subtitle.configure(text=f"{self.agent.name} chose illegal column {col + 1}.")
        self._play(col)

    def _play(self, col):
        mover = self.env.current_player
        _obs, reward, terminated, _truncated, info = self.env.step(col)

        if terminated:
            self.game_over = True
            if info["winner"] != 0 and not info.get("invalid_action"):
                self.win_line = winning_cells(self.env.board, *info["last_move"]) or ()
            self._record(info["winner"])
            self.draw()
            self._announce(info, reward, mover)
        else:
            self.draw()
            self._schedule_agent()

    def _schedule_agent(self):
        """Hand the turn to the agent after a short, visible pause."""
        if self.game_over or self.env.current_player == self.human_player:
            self._update_status()
            return
        self.busy = True
        self._update_status()
        self.root.after(AGENT_DELAY_MS, self._agent_move)

    def _record(self, winner):
        if winner == 0:
            self.score["draw"] += 1
        elif winner == self.human_player:
            self.score["you"] += 1
        else:
            self.score["agent"] += 1

    # ----------------------------------------------------------------- text

    def _update_status(self):
        if self.env.current_player == self.human_player:
            self.status.configure(text="Your turn", fg=COLOR_HUMAN)
        else:
            self.status.configure(text=f"{self.agent.name} is thinking…", fg=COLOR_AGENT)
        self.subtitle.configure(text=self._scoreline())

    def _announce(self, info, reward, mover):
        winner = info["winner"]
        if winner == 0:
            self.status.configure(text="Draw", fg=COLOR_TEXT)
        elif winner == self.human_player:
            self.status.configure(text="You win!", fg=COLOR_HUMAN)
        else:
            self.status.configure(text=f"{self.agent.name} wins", fg=COLOR_AGENT)

        # `reward` belongs to whoever just moved; show it from your seat.
        yours = reward if mover == self.human_player else -reward
        note = f"reward {yours:+.0f} · {info['move_count']} moves"
        if info.get("invalid_action"):
            note = "illegal move — game forfeited · " + note
        self.subtitle.configure(text=f"{note} · {self._scoreline()} · press N for a new game")

    def _scoreline(self):
        s = self.score
        return f"you {s['you']} – {s['agent']} {self.agent.name}  (draws {s['draw']})"

    # -------------------------------------------------------------- drawing

    def _color(self, player):
        return COLOR_HUMAN if player == self.human_player else COLOR_AGENT

    def _cell_center(self, row, col):
        return (
            MARGIN + col * CELL + CELL / 2,
            MARGIN + CHUTE + row * CELL + CELL / 2,
        )

    def _on_motion(self, event):
        col = int((event.x - MARGIN) // CELL)
        inside = 0 <= col < self.env.cols and event.x >= MARGIN
        self._set_hover(col if inside else None)

    def _set_hover(self, col):
        if col != self.hover_col:
            self.hover_col = col
            self.draw()

    def _on_click(self, event):
        col = int((event.x - MARGIN) // CELL)
        if 0 <= col < self.env.cols and event.x >= MARGIN:
            self._human_move(col)

    def draw(self):
        c = self.canvas
        c.delete("all")

        self._draw_chute()

        board_top = MARGIN + CHUTE
        c.create_rectangle(
            MARGIN, board_top,
            MARGIN + self.env.cols * CELL, board_top + self.env.rows * CELL,
            fill=COLOR_BOARD, outline="",
        )

        wins = set(self.win_line)
        last = self.env.last_move
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                x, y = self._cell_center(row, col)
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

    def _draw_chute(self):
        """Ghost disc above the column under the cursor."""
        if self.game_over or self.busy or self.hover_col is None:
            return
        if self.env.current_player != self.human_player:
            return
        if not self.env.is_valid_action(self.hover_col):
            return
        x = MARGIN + self.hover_col * CELL + CELL / 2
        y = MARGIN + CHUTE / 2
        self.canvas.create_oval(
            x - DISC, y - DISC, x + DISC, y + DISC,
            fill=self._color(self.human_player), outline="",
        )


def build_agent(args):
    if args.agent == "random":
        return RandomAgent(seed=args.seed)
    if not args.checkpoint:
        raise SystemExit("--agent dqn needs --checkpoint path/to/weights.pt")
    return DQNAgent(args.checkpoint, device=args.device)


def main():
    parser = argparse.ArgumentParser(description="Play Connect 4 against an agent.")
    parser.add_argument("--agent", choices=["random", "dqn"], default="random")
    parser.add_argument("--checkpoint", help="weights for --agent dqn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--second", action="store_true", help="let the agent move first")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    Connect4GUI(build_agent(args), human_first=not args.second, seed=args.seed).run()


if __name__ == "__main__":
    main()
