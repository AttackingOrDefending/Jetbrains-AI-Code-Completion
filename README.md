## Code used

Taken from my project [lidraughts-bot](https://github.com/AttackingOrDefending/lidraughts-bot).

## How to Run

Run `python3 main.py` on linux/mac or `py main.py` on windows.

## Results

See `output.txt` for exact results.

Total testcases: 33
Errors: 10
Correct suggestions (manual): 8
Exact matches (automatic): 3
CHRF (automatic): 26.6749
Good suggestions (whether one result is a substring of the other) (automatic): 5
Similar length (automatic): 10

The best metric was `Good suggestions (whether one result is a substring of the other)` as considered correct 5 of the 8 outputs that I manually reviewed as correct.

### Comparison (my review)

Testcase 1:
```
Model output:
sleep(1)
Correct output:
sleep(60)
```

Manual result: Correct

Testcase 2:
```
Model output:
move_part_to_move[move_part.id] = move_part
Correct output:
api_post(ENDPOINTS["move"].format(game_id, move_part),
```

Manual result: Wrong

Testcase 3:
```
Model output:
self.url, self.event_stream_url)
Correct output:
baseUrl, ENDPOINTS["stream_event"])
```

Manual result: Wrong

Testcase 4:
```
Model output:
url, f"game/{game_id}")
Correct output:
baseUrl, ENDPOINTS["stream"].format(game_id))
```

Manual result: Wrong

Testcase 5:
```
Model output:
login(challenge_id, reason)
Correct output:
api_post(ENDPOINTS["decline"].format(challenge_id), data=f"reason={reason}",
```

Manual result: Wrong

Testcase 6:
```
Model output:
api_delete(ENDPOINTS["profile"])
Correct output:
set_user_agent(profile["username"])
```

Manual result: Wrong

Testcase 7:
```
Model output:
get_ongoing_games_from_db()
Correct output:
api_get(ENDPOINTS["playing"])["nowPlaying"]
```

Manual result: Wrong

Testcase 8:
```
error
Model output:
<fim_prefix>def set_user_agent(self, username):
        self.<fim_suffix>
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
Correct output:
header.update({"User-Agent": f"lidraughts-bot/{self.version} user:{username}"})
```

Manual result: Error

Testcase 9:
```
Model output:
user_by_id(user_id)
Correct output:
api_get(ENDPOINTS["status"], params={"ids": user_id})
```

Manual result: Wrong

Testcase 10:
```
error
Model output:
<fim_prefix>def reset_connection(self):
        self.session.close()
        self.<fim_suffix>
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.headers.update(self.header)
        self.session.
Correct output:
session = requests.Session()
```

Manual result: Error

Testcase 11:
```
error
Model output:
<fim_prefix>def first_search(self, board, movetime, draw_offered):
        # No pondering after the first move since a different clock is used afterwards.<fim_suffix>
        return self.search_for(board, movetime, draw_offered)
    def search_for(self, board, movetime, draw_offered):
        # No pondering after the first move since a different clock is used afterwards.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because the search is done in a different way.
        # This is because
Correct output:
```

Manual result: Error

Testcase 12:
```
Model output:
get("movetime") is not None:
Correct output:
movetime is None or time_limit.movetime > movetime_sec:
```

Manual result: Wrong

Testcase 13:
```
error
Model output:
<fim_prefix>def offer_draw_or_resign(self, result, board):
        def actual(score):
            if "cp" in score:
                return score.get("cp", float("nan"))
            else:
                win = score.<fim_suffix>
                if win > 0:
                    return 10000 - win
                else:
                    return -10000 - win
            else:
                return 0
        if result.get("cp", float("nan")) > 0:
            return 10000 - result.get("cp", float("nan"))
        else:
            return -10000 - result.get("cp", float("nan"))
    def offer_draw_or_resign_with_score(self, result, board):
        def actual(score):
            if "cp" in score:
                return score.get("cp", float("nan"))
            else:
                win = score.get("win", float("nan"))
                if win > 0:
                    return 10000 - win
                else:
                    return -10000 - win
            else:
                return 0
        if result.get("cp", float("nan")) > 0:
            return 10000 - result.get("cp", float("nan"))
        else:
            return -10000 - result.get("cp", float("nan"))
    def offer_draw_or_resign_with_score_with_win(self, result, board):
        def actual(score):
            if "cp" in score:
                return score.get("cp", float("nan"))
            else:
                win = score.get("win", float("nan"))
                if win > 0:
                    return 10000 - win
                else:
                    return -10000 - win
            else:
                return
Correct output:
get("win", float("nan"))
```

Manual result: Error

Testcase 14:
```
error
Model output:
<fim_prefix>def process_playresult(self, board, result):
        self.last_move_info = result.info.copy()
        self.move_commentary.append(self.last_move_info.copy())
        if self.comment_start_index is None:
            self.comment_start_index = len(board.move_stack)
        self.<fim_suffix>
        result = self.offer_draw_or_resign(result, board)
        self.last_move_info["ponderpv"] = self.last_move_info.get("pv", "")[1:-1].split()
        self.print_stats()
        return result
    def print_stats(self):
        print("Last move info:")
        print(self.last_move_info)
        print("Commentary:")
        print(self.comment_start_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
        print("Commentary start index:")
        print(self.comment_start_index)
        print("Commentary end index:")
        print(self.comment_end_index)
Correct output:
scores.append(self.last_move_info.get("score", {"win": 1}))
```

Manual result: Error

Testcase 15:
```
error
Model output:
<fim_prefix>def add_null_comment(self):
        if self.<fim_suffix>
            self.move_commentary.append(None)
        else:
            self.move_commentary.append(self.commentary)
    def add_commentary(self, commentary):
        self.commentary = commentary
        self.move_commentary.append(commentary)
    def add_commentary_to_commentary(self, commentary):
        self.commentary = commentary
        self.move_commentary.append(commentary)
    def add_commentary_to_commentary_from_commentary(self, commentary):
        self.commentary = commentary
        self.move_commentary.append(commentary)
    def add_comment
Correct output:
comment_start_index is not None:
```

Manual result: Error

Testcase 16:
```
Model output:
stats:
Correct output:
get_stats():
```

Manual result: Wrong

Testcase 17:
```
Model output:
search_board(board, time_limit, ponder, draw_offered)
Correct output:
engine.play(board, time_limit, ponder=ponder)
```

Manual result: Wrong

Testcase 18:
```
Model output:
engine.get_time_limit(time_limit)
Correct output:
add_go_commands(time_limit)
```

Manual result: Wrong

Testcase 19:
```
Model output:
engine.get_time_limit(time_limit)
Correct output:
add_go_commands(time_limit)
```

Manual result: Wrong

Testcase 20:
```
Model output:
exit(0)
Correct output:
debug("Recieved SIGINT. Terminating client.")
```

Manual result: Wrong

Testcase 21:
```
Model output:
account_id:
Correct output:
upgrade_to_bot_account() is None:
```

Manual result: Wrong

Testcase 22:
```
Model output:
sleep(period)
Correct output:
sleep(period)
```

Manual result: Correct

Testcase 23:
```
Model output:
Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
Correct output:
Formatter("%(message)s")
```

Manual result: Correct

Testcase 24:
```
Model output:
getLogger()
Correct output:
getLogger()
```

Manual result: Correct

Testcase 25:
```
Model output:
debug("choose_move_time")
Correct output:
info(f"Searching for time {search_time}")
```

Manual result: Correct

Testcase 26:
```
Model output:
state[f"{wb}inc"] = pre_move_time
Correct output:
state[f"{wb}time"] = max(0, game.state[f"{wb}time"] - overhead)
```

Manual result: Correct

Testcase 27:
```
error
Model output:
<fim_prefix>def start_pondering(engine, board, game, can_ponder, best_move, start_time, move_overhead, move_overhead_inc):
    if not can_ponder or best_move.<fim_suffix>
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
Correct output:
ponder is None:
```

Manual result: Error

Testcase 28:
```
error
Model output:
<fim_prefix>def ponder_thread_func(game, engine, board, wtime, btime, winc, binc):
        global ponder_results
        best_move = engine.<fim_suffix>
        ponder_results[game.id] = best_move
        engine.update_board(board)
        engine.update_time(wtime, btime)
        engine.update_inc(winc, binc)
        engine.update_engine_state(engine.state)
        engine.update_engine_score(engine.score)
        engine.update_engine_time(engine.time)
        engine.update_engine_inc(engine.inc)
        engine.update_engine_engine_state(engine.engine_state)
        engine.update_engine_engine_score(engine.engine_score)
        engine.update_engine_engine_time(engine.engine_time)
        engine.update_engine_inc(engine.inc)
        engine.update_engine_engine_state(engine.engine_state)
        engine.update_engine_engine_score(engine.engine_score)
        engine.update_engine
Correct output:
search_with_ponder(board, wtime, btime, winc, binc, True, False)
```

Manual result: Error

Testcase 29:
```
error
Model output:
<fim_prefix>def fake_thinking(config, board, game):
    if config.get("fake_think_time") and len(board.move_stack) > 9:
        delay = min(game.clock_initial, game.my_remaining_seconds()) * 0.015
        accel = 1 - max(0, min(100, len(board.<fim_suffix>
        sleep = min(5, delay * accel)
        time.sleep(sleep)
        board.move_stack.pop()
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move_stack.append(move)
        board.move
Correct output:
move_stack) - 20)) / 150
```

Manual result: Error

Testcase 30:
```
error
Model output:
<fim_prefix>def print_move_number(board):
    logger.<fim_suffix>
    logger.info(f"move: {len(board.move_stack) // 2 + 1}")
    logger.info(f"board: {board}")
    logger.info(f"board.move_stack: {board.move_stack}")
    logger.info(f"board.move_stack.pop(): {board.move_stack.pop()}")
    logger.info(f"board.move_stack.popleft(): {board.move_stack.popleft()}")
    logger.info(f"board.move_stack.popright(): {board.move_stack.popright()}")
    logger.info(f"board.move_stack.pop(): {board.move_stack.pop()
Correct output:
info("")
```

Manual result: Error

Testcase 31:
```
Model output:
get("winner")
Correct output:
state.get("winner")
```

Manual result: Correct

Testcase 32:
```
Model output:
  //  /
Correct output:
  || ._)  lidraughts-bot {__version__}
```

Manual result: Wrong

Testcase 33:
```
Model output:
add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
Correct output:
add_argument("-v", action="store_true", help="Make output more verbose. Include all communication with lichess.")
```

Manual result: Correct
