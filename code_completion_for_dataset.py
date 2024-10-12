import requests
from urllib.parse import urljoin
from requests.exceptions import ConnectionError, HTTPError, ReadTimeout
from http.client import RemoteDisconnected
import backoff
import logging
import time

ENDPOINTS = {
    "profile": "/api/account",
    "playing": "/api/account/playing",
    "stream": "/api/bot/game/stream/{}",
    "stream_event": "/api/stream/event",
    "game": "/api/bot/game/{}",
    "move": "/api/bot/game/{}/move/{}",
    "chat": "/api/bot/game/{}/chat",
    "abort": "/api/bot/game/{}/abort",
    "accept": "/api/challenge/{}/accept",
    "decline": "/api/challenge/{}/decline",
    "upgrade": "/api/bot/account/upgrade",
    "resign": "/api/bot/game/{}/resign",
    "export": "/game/export/{}",
    "status": "/api/users/status"
}


logger = logging.getLogger(__name__)


def rate_limit_check(response):
    if response.status_code == 429:
        logger.warning("Rate limited. Waiting 1 minute until next request.")
        time.sleep(60)
        return True
    return False


# docs: https://lidraughts.org/api
class Lidraughts:
    def __init__(self, token, url, version, logging_level):
        self.version = version
        self.header = {
            "Authorization": f"Bearer {token}"
        }
        self.baseUrl = url
        self.session = requests.Session()
        self.session.headers.update(self.header)
        self.set_user_agent("?")
        self.logging_level = logging_level

    def is_final(exception):
        return isinstance(exception, HTTPError) and exception.response.status_code < 500

    @backoff.on_exception(backoff.constant,
                          (RemoteDisconnected, ConnectionError, HTTPError, ReadTimeout),
                          max_time=60,
                          interval=0.1,
                          giveup=is_final,
                          backoff_log_level=logging.DEBUG,
                          giveup_log_level=logging.DEBUG)
    def api_get(self, path, raise_for_status=True, get_raw_text=False, params=None):
        logging.getLogger("backoff").setLevel(self.logging_level)
        url = urljoin(self.baseUrl, path)
        response = self.session.get(url, timeout=2, params=params)
        if rate_limit_check(response) or raise_for_status:
            response.raise_for_status()
        return response.text if get_raw_text else response.json()

    @backoff.on_exception(backoff.constant,
                          (RemoteDisconnected, ConnectionError, HTTPError, ReadTimeout),
                          max_time=60,
                          interval=0.1,
                          giveup=is_final,
                          backoff_log_level=logging.DEBUG,
                          giveup_log_level=logging.DEBUG)
    def api_post(self, path, data=None, headers=None, params=None, raise_for_status=True):
        logging.getLogger("backoff").setLevel(self.logging_level)
        url = urljoin(self.baseUrl, path)
        response = self.session.post(url, data=data, headers=headers, params=params, timeout=2)
        if rate_limit_check(response) or raise_for_status:
            response.raise_for_status()
        return response.json()

    def get_game(self, game_id):
        return self.api_get(ENDPOINTS["game"].format(game_id))

    def upgrade_to_bot_account(self):
        return self.api_post(ENDPOINTS["upgrade"])

    def make_move(self, game_id, move):
        for move_part in move.move.li_api_move:
            self.api_post(ENDPOINTS["move"].format(game_id, move_part),
                          params={"offeringDraw": str(move.draw_offered).lower()})

    def chat(self, game_id, room, text):
        payload = {"room": room, "text": text}
        return self.api_post(ENDPOINTS["chat"].format(game_id), data=payload)

    def abort(self, game_id):
        return self.api_post(ENDPOINTS["abort"].format(game_id))

    def get_event_stream(self):
        url = urljoin(self.baseUrl, ENDPOINTS["stream_event"])
        return requests.get(url, headers=self.header, stream=True)

    def get_game_stream(self, game_id):
        url = urljoin(self.baseUrl, ENDPOINTS["stream"].format(game_id))
        return requests.get(url, headers=self.header, stream=True)

    def accept_challenge(self, challenge_id):
        return self.api_post(ENDPOINTS["accept"].format(challenge_id))

    def decline_challenge(self, challenge_id, reason="generic"):
        return self.api_post(ENDPOINTS["decline"].format(challenge_id), data=f"reason={reason}",
                             headers={"Content-Type": "application/x-www-form-urlencoded"})

    def get_profile(self):
        profile = self.api_get(ENDPOINTS["profile"])
        self.set_user_agent(profile["username"])
        return profile

    def get_ongoing_games(self):
        ongoing_games = self.api_get(ENDPOINTS["playing"])["nowPlaying"]
        return ongoing_games

    def resign(self, game_id):
        self.api_post(ENDPOINTS["resign"].format(game_id))

    def set_user_agent(self, username):
        self.header.update({"User-Agent": f"lidraughts-bot/{self.version} user:{username}"})
        self.session.headers.update(self.header)

    def get_game_pgn(self, game_id):
        return self.api_get(ENDPOINTS["export"].format(game_id), get_raw_text=True, params={"literate": "true"})

    def is_online(self, user_id):
        user = self.api_get(ENDPOINTS["status"], params={"ids": user_id})
        return user and user[0].get("online")

    def reset_connection(self):
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update(self.header)


import os
import draughts
import draughts.engine
import subprocess
import logging
from enum import Enum

logger = logging.getLogger(__name__)


def create_engine(config, variant, initial_time):
    cfg = config["engine"]
    engine_path = os.path.join(cfg["dir"], cfg["name"])
    engine_working_dir = cfg.get("working_dir") or os.getcwd()
    engine_type = cfg.get("protocol")
    engine_options = cfg.get("engine_options")
    draw_or_resign = cfg.get("draw_or_resign") or {}
    commands = [engine_path, cfg["engine_argument"]]
    if engine_options:
        for k, v in engine_options.items():
            commands.append(f"--{k}={v}")

    stderr = None if cfg.get("silence_stderr", False) else subprocess.DEVNULL

    if engine_type == "hub":
        Engine = HubEngine
    elif engine_type == "dxp":
        Engine = DXPEngine
    elif engine_type == "cb":
        Engine = CBEngine
    elif engine_type == "homemade":
        Engine = getHomemadeEngine(cfg["name"])
    else:
        raise ValueError(
            f"    Invalid engine type: {engine_type}. Expected hub, dxp, cb, or homemade.")
    options = cfg.get(f"{engine_type}_options") or {}
    options["variant"] = variant
    options["initial-time"] = initial_time
    logger.debug(f"Starting engine: {' '.join(commands)}")
    return Engine(commands, options, stderr, draw_or_resign, cwd=engine_working_dir)


class Termination(str, Enum):
    MATE = "mate"
    TIMEOUT = "outoftime"
    RESIGN = "resign"
    ABORT = "aborted"
    DRAW = "draw"


class GameEnding(str, Enum):
    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    INCOMPLETE = "*"


def translate_termination(termination, board, winner_color):
    if termination == Termination.MATE:
        return f"{winner_color.title()} mates"
    elif termination == Termination.TIMEOUT:
        return "Time forfeiture"
    elif termination == Termination.RESIGN:
        resigner = "black" if winner_color == "white" else "white"
        return f"{resigner.title()} resigns"
    elif termination == Termination.ABORT:
        return "Game aborted"
    elif termination == Termination.DRAW:
        if board.is_fifty_moves():
            return "50-move rule"
        elif board.is_repetition():
            return "Threefold repetition"
        else:
            return "Draw by agreement"
    elif termination:
        return termination
    else:
        return ""


PONDERPV_CHARACTERS = 12  # the length of ", ponderpv: "
MAX_CHAT_MESSAGE_LEN = 140  # maximum characters in a chat message


class EngineWrapper:
    def __init__(self, options, draw_or_resign):
        self.scores = []
        self.draw_or_resign = draw_or_resign
        self.go_commands = options.pop("go_commands", {}) or {}
        self.last_move_info = {}
        self.move_commentary = []
        self.comment_start_index = None

    def search_for(self, board, movetime, draw_offered):
        return self.search(board, draughts.engine.Limit(movetime=movetime / 1000), False, draw_offered)

    def first_search(self, board, movetime, draw_offered):
        # No pondering after the first move since a different clock is used afterwards.
        return self.search_for(board, movetime, draw_offered)

    def search_with_ponder(self, board, wtime, btime, winc, binc, ponder, draw_offered):
        if board.whose_turn() == draughts.WHITE:
            time = wtime
            inc = winc
        else:
            time = btime
            inc = binc
        time_limit = draughts.engine.Limit(time=time / 1000,
                                           inc=inc / 1000)
        return self.search(board, time_limit, ponder, draw_offered)

    def add_go_commands(self, time_limit):
        movetime = self.go_commands.get("movetime")
        if movetime is not None:
            movetime_sec = float(movetime) / 1000
            if time_limit.movetime is None or time_limit.movetime > movetime_sec:
                time_limit.movetime = movetime_sec
        time_limit.depth = self.go_commands.get("depth")
        time_limit.nodes = self.go_commands.get("nodes")
        return time_limit

    def offer_draw_or_resign(self, result, board):
        def actual(score):
            if "cp" in score:
                return score.get("cp", float("nan"))
            else:
                win = score.get("win", float("nan"))
                if win > 0:
                    return 10000 - win
                else:
                    return -10000 - win

        can_offer_draw = self.draw_or_resign.get("offer_draw_enabled", False)
        draw_offer_moves = self.draw_or_resign.get("offer_draw_moves", 5)
        draw_score_range = self.draw_or_resign.get("offer_draw_score", 0)
        draw_max_piece_count = self.draw_or_resign.get("offer_draw_pieces", 10)
        pieces_on_board = len(board.board.pieces)
        enough_pieces_captured = pieces_on_board <= draw_max_piece_count

        if can_offer_draw and len(self.scores) >= draw_offer_moves and enough_pieces_captured:
            scores = self.scores[-draw_offer_moves:]

            def score_near_draw(score):
                return abs(actual(score)) <= draw_score_range
            if len(scores) == len(list(filter(score_near_draw, scores))):
                result.draw_offered = True

        resign_enabled = self.draw_or_resign.get("resign_enabled", False)
        min_moves_for_resign = self.draw_or_resign.get("resign_moves", 3)
        resign_score = self.draw_or_resign.get("resign_score", -1000)

        if resign_enabled and len(self.scores) >= min_moves_for_resign:
            scores = self.scores[-min_moves_for_resign:]

            def score_near_loss(score):
                return actual(score) <= resign_score
            if len(scores) == len(list(filter(score_near_loss, scores))):
                result.resigned = True
        return result

    def search(self, board, time_limit, ponder, draw_offered):
        pass

    def process_playresult(self, board, result):
        self.last_move_info = result.info.copy()
        self.move_commentary.append(self.last_move_info.copy())
        if self.comment_start_index is None:
            self.comment_start_index = len(board.move_stack)
        self.scores.append(self.last_move_info.get("score", {"win": 1}))
        result = self.offer_draw_or_resign(result, board)
        self.last_move_info["ponderpv"] = self.last_move_info.get("pv", "")[1:-1].split()
        self.print_stats()
        return result

    def comment_index(self, move_stack_index):
        if self.comment_start_index is None:
            return -1
        else:
            return move_stack_index - self.comment_start_index

    def comment_for_board_index(self, index):
        comment_index = self.comment_index(index)
        if comment_index < 0 or comment_index % 2 != 0:
            return None

        try:
            return self.move_commentary[comment_index // 2]
        except IndexError:
            return None

    def add_null_comment(self):
        if self.comment_start_index is not None:
            self.move_commentary.append(None)

    def print_stats(self):
        for line in self.get_stats():
            logger.info(line)

    def get_stats(self, for_chat=False):
        info = self.last_move_info.copy()
        stats = ["depth", "nps", "nodes", "score", "ponderpv"]
        if for_chat:
            bot_stats = [f"{stat}: {info[stat]}" for stat in stats if stat in info and stat != "ponderpv"]
            len_bot_stats = len(", ".join(bot_stats)) + PONDERPV_CHARACTERS
            ponder_pv = info["ponderpv"]
            ponder_pv = ponder_pv.split()
            try:
                while len(" ".join(ponder_pv)) + len_bot_stats > MAX_CHAT_MESSAGE_LEN:
                    ponder_pv.pop()
                if ponder_pv[-1].endswith("."):
                    ponder_pv.pop()
                info["ponderpv"] = " ".join(ponder_pv)
            except IndexError:
                pass
        return [f"{stat}: {info[stat]}" for stat in stats if stat in info]

    def get_opponent_info(self, game):
        pass

    def name(self):
        return self.engine.id.get("name", "")

    def report_game_result(self, game, board):
        pass

    def stop(self):
        pass

    def quit(self):
        pass

    def kill_process(self):
        self.engine.kill_process()

    def ponderhit(self):
        pass


class HubEngine(EngineWrapper):
    def __init__(self, commands, options, stderr, draw_or_resign, **popen_args):
        super().__init__(options, draw_or_resign)
        self.engine = draughts.engine.HubEngine(commands, **popen_args)

        if "bb-size" in options and options["bb-size"] == "auto":
            if "variant" in options and options["variant"] != "normal":
                variant = f'_{options["variant"]}'
            else:
                variant = ""
            for number in range(1, 7):
                path = os.path.realpath(f"./data/bb{variant}/{number + 1}")
                if not os.path.isdir(path):
                    break
            else:
                number += 1
            if number == 1:
                number = 0
            options["bb-size"] = number

        self.engine.configure(options)
        self.engine.init()

    def search(self, board, time_limit, ponder, draw_offered):
        time_limit = self.add_go_commands(time_limit)
        result = self.engine.play(board, time_limit, ponder=ponder)
        return self.process_playresult(board, result)

    def stop(self):
        self.engine.stop()

    def quit(self):
        self.engine.quit()

    def ponderhit(self):
        self.engine.ponderhit()


class DXPEngine(EngineWrapper):
    def __init__(self, commands, options, stderr, draw_or_resign, **popen_args):
        super().__init__(options, draw_or_resign)
        self.engine = draughts.engine.DXPEngine(commands, options=options, **popen_args)

    def search(self, board, time_limit, ponder, draw_offered):
        if ponder:
            return draughts.engine.PlayResult(None, None)
        time_limit = self.add_go_commands(time_limit)
        result = self.engine.play(board)
        return self.process_playresult(board, result)

    def quit(self):
        self.engine.quit()


class CBEngine(EngineWrapper):
    def __init__(self, commands, options, stderr, draw_or_resign, **popen_args):
        super().__init__(options, draw_or_resign)
        self.engine = draughts.engine.CheckerBoardEngine(commands)
        self.engine.configure(options)

    def search(self, board, time_limit, ponder, draw_offered):
        if ponder:
            return draughts.engine.PlayResult(None, None)
        time_limit = self.add_go_commands(time_limit)
        result = self.engine.play(board, time_limit)
        return self.process_playresult(board, result)


def getHomemadeEngine(name):
    import strategies
    return getattr(strategies, name)


import argparse
import draughts
import draughts.engine
import engine_wrapper
import model
import json
import lidraughts
import logging
import logging.handlers
import multiprocessing
import signal
import time
import backoff
import sys
import threading
import os
import copy
from config import load_config
from conversation import Conversation, ChatLine
from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, ReadTimeout
from rich.logging import RichHandler
from collections import defaultdict
from http.client import RemoteDisconnected

logger = logging.getLogger(__name__)

__version__ = "1.2.0"

terminated = False


def signal_handler(signal, frame):
    global terminated
    logger.debug("Recieved SIGINT. Terminating client.")
    terminated = True


signal.signal(signal.SIGINT, signal_handler)


def is_final(exception):
    return isinstance(exception, HTTPError) and exception.response.status_code < 500


def upgrade_account(li):
    if li.upgrade_to_bot_account() is None:
        return False

    logger.info("Succesfully upgraded to Bot Account!")
    return True


def watch_control_stream(control_queue, li):
    while not terminated:
        try:
            response = li.get_event_stream()
            lines = response.iter_lines()
            for line in lines:
                if line:
                    event = json.loads(line.decode("utf-8"))
                    control_queue.put_nowait(event)
                else:
                    control_queue.put_nowait({"type": "ping"})
        except Exception:
            pass


def do_correspondence_ping(control_queue, period):
    while not terminated:
        time.sleep(period)
        control_queue.put_nowait({"type": "correspondence_ping"})


def logging_configurer(level, filename):
    console_handler = RichHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    all_handlers = [console_handler]

    if filename:
        file_handler = logging.FileHandler(filename, delay=True)
        FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
        file_formatter = logging.Formatter(FORMAT)
        file_handler.setFormatter(file_formatter)
        all_handlers.append(file_handler)

    logging.basicConfig(level=level,
                        handlers=all_handlers,
                        force=True)


def logging_listener_proc(queue, configurer, level, log_filename):
    configurer(level, log_filename)
    logger = logging.getLogger()
    while not terminated:
        try:
            logger.handle(queue.get())
        except Exception:
            pass


def game_logging_configurer(queue, level):
    if sys.platform == "win32":
        h = logging.handlers.QueueHandler(queue)
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(h)
        root.setLevel(level)


def game_error_handler(error):
    logger.exception("Game ended due to error:", exc_info=error)


def start(li, user_profile, config, logging_level, log_filename, one_game=False):
    challenge_config = config["challenge"]
    max_games = challenge_config.get("concurrency", 1)
    logger.info(f"You're now connected to {config['url']} and awaiting challenges.")
    manager = multiprocessing.Manager()
    challenge_queue = manager.list()
    control_queue = manager.Queue()
    control_stream = multiprocessing.Process(target=watch_control_stream, args=[control_queue, li])
    control_stream.start()
    correspondence_cfg = config.get("correspondence") or {}
    correspondence_checkin_period = correspondence_cfg.get("checkin_period", 600)
    correspondence_pinger = multiprocessing.Process(target=do_correspondence_ping,
                                                    args=[control_queue, correspondence_checkin_period])
    correspondence_pinger.start()
    correspondence_queue = manager.Queue()
    correspondence_queue.put("")
    startup_correspondence_games = [game["gameId"] for game in li.get_ongoing_games() if game["perf"] == "correspondence"]
    wait_for_correspondence_ping = False
    last_check_online_time = time.time()

    busy_processes = 0
    queued_processes = 0

    logging_queue = manager.Queue()
    logging_listener = multiprocessing.Process(target=logging_listener_proc,
                                               args=(logging_queue, logging_configurer, logging_level, log_filename))
    logging_listener.start()

    def log_proc_count(change, queued, used):
        symbol = "+++" if change == "Freed" else "---"
        logger.info(f"{symbol} Process {change}. Total Queued: {queued}. Total Used: {used}")

    play_game_args = [li,
                      None,  # will hold the game id
                      control_queue,
                      user_profile,
                      config,
                      challenge_queue,
                      correspondence_queue,
                      logging_queue,
                      game_logging_configurer,
                      logging_level]

    with multiprocessing.pool.Pool(max_games + 1) as pool:
        while not terminated:
            try:
                event = control_queue.get()
                if event.get("type") != "ping":
                    logger.debug(f"Event: {event}")
            except InterruptedError:
                continue

            if event.get("type") is None:
                logger.warning("Unable to handle response from lidraughts.org:")
                logger.warning(event)
                if event.get("error") == "Missing scope":
                    logger.warning('Please check that the API access token for your bot has the scope "Play games with the bot'
                                   ' API".')
                continue

            if event["type"] == "terminated":
                break
            elif event["type"] == "local_game_done":
                busy_processes -= 1
                log_proc_count("Freed", queued_processes, busy_processes)
                if one_game:
                    break
            elif event["type"] == "challenge":
                chlng = model.Challenge(event["challenge"])
                is_supported, decline_reason = chlng.is_supported(challenge_config)
                if is_supported:
                    challenge_queue.append(chlng)
                    if challenge_config.get("sort_by", "best") == "best":
                        list_c = list(challenge_queue)
                        list_c.sort(key=lambda c: -c.score())
                        challenge_queue = list_c
                else:
                    li.decline_challenge(chlng.id, reason=decline_reason)
            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                if game_id in startup_correspondence_games:
                    logger.info(f'--- Enqueue {config["url"] + game_id}')
                    correspondence_queue.put(game_id)
                    startup_correspondence_games.remove(game_id)
                else:
                    if queued_processes > 0:
                        queued_processes -= 1
                    busy_processes += 1
                    log_proc_count("Used", queued_processes, busy_processes)
                    play_game_args[1] = game_id
                    pool.apply_async(play_game, play_game_args, error_callback=game_error_handler)

            is_correspondence_ping = event["type"] == "correspondence_ping"
            is_local_game_done = event["type"] == "local_game_done"
            if (is_correspondence_ping or (is_local_game_done and not wait_for_correspondence_ping)) and not challenge_queue:
                if is_correspondence_ping and wait_for_correspondence_ping:
                    correspondence_queue.put("")

                wait_for_correspondence_ping = False
                while (busy_processes + queued_processes) < max_games:
                    game_id = correspondence_queue.get()
                    # stop checking in on games if we have checked in on all games since the last correspondence_ping
                    if not game_id:
                        if is_correspondence_ping and not correspondence_queue.empty():
                            correspondence_queue.put("")
                        else:
                            wait_for_correspondence_ping = True
                            break
                    else:
                        busy_processes += 1
                        log_proc_count("Used", queued_processes, busy_processes)
                        play_game_args[1] = game_id
                        pool.apply_async(play_game, play_game_args, error_callback=game_error_handler)

            # Keep processing the queue until empty or max_games is reached.
            while (queued_processes + busy_processes) < max_games and challenge_queue:
                chlng = challenge_queue.pop(0)
                try:
                    logger.info(f"Accept {chlng}")
                    queued_processes += 1
                    li.accept_challenge(chlng.id)
                    log_proc_count("Queued", queued_processes, busy_processes)
                except (HTTPError, ReadTimeout) as exception:
                    if isinstance(exception, HTTPError) and exception.response.status_code == 404:
                        logger.info(f"Skip missing {chlng}")
                    queued_processes -= 1

            if time.time() > last_check_online_time + 60 * 60:  # 1 hour.
                if not li.is_online(user_profile["id"]):
                    logger.info("Will reset connection with lichess")
                    li.reset_connection()
                last_check_online_time = time.time()

            control_queue.task_done()

    logger.info("Terminated")
    control_stream.terminate()
    control_stream.join()
    correspondence_pinger.terminate()
    correspondence_pinger.join()
    logging_listener.terminate()
    logging_listener.join()


ponder_results = {}


@backoff.on_exception(backoff.expo, BaseException, max_time=600, giveup=is_final)
def play_game(li,
              game_id,
              control_queue,
              user_profile,
              config,
              challenge_queue,
              correspondence_queue,
              logging_queue,
              game_logging_configurer,
              logging_level):
    game_logging_configurer(logging_queue, logging_level)
    logger = logging.getLogger(__name__)

    response = li.get_game_stream(game_id)
    lines = response.iter_lines()

    # Initial response of stream will be the full game info. Store it
    initial_state = json.loads(next(lines).decode("utf-8"))
    logger.debug(f"Initial state: {initial_state}")
    abort_time = config.get("abort_time", 20)
    game = model.Game(initial_state, user_profile["username"], li.baseUrl, abort_time)

    initial_time = (game.state["wtime"] if game.my_color == "white" else game.state["btime"]) / 1000
    variant = parse_variant(game.variant_name)
    engine = engine_wrapper.create_engine(config, variant, initial_time)
    conversation = Conversation(game, engine, li, __version__, challenge_queue)

    logger.info(f"+++ {game}")

    is_correspondence = game.perf_name == "Correspondence"
    correspondence_cfg = config.get("correspondence") or {}
    correspondence_move_time = correspondence_cfg.get("move_time", 60) * 1000
    correspondence_disconnect_time = correspondence_cfg.get("disconnect_time", 300)

    engine_cfg = config["engine"]
    ponder_cfg = correspondence_cfg if is_correspondence else engine_cfg
    can_ponder = ponder_cfg.get("ponder", False)
    move_overhead = config.get("move_overhead", 1000)
    move_overhead_inc = config.get("move_overhead_inc", 100)
    delay_seconds = config.get("rate_limiting_delay", 0)/1000

    greeting_cfg = config.get("greeting") or {}
    keyword_map = defaultdict(str, me=game.me.name, opponent=game.opponent.name)

    def get_greeting(greeting):
        return str(greeting_cfg.get(greeting) or "").format_map(keyword_map)
    hello = get_greeting("hello")
    goodbye = get_greeting("goodbye")
    hello_spectators = get_greeting("hello_spectators")
    goodbye_spectators = get_greeting("goodbye_spectators")

    board = draughts.Game(game.variant_name.lower(), game.initial_fen)
    moves, old_moves = [], []
    ponder_thread = None
    ponder_li_one = None

    first_move = True
    disconnect_time = 0
    prior_game = None
    while not terminated:
        move_attempted = False
        try:
            if first_move:
                upd = game.state
                first_move = False
            else:
                binary_chunk = next(lines)
                upd = json.loads(binary_chunk.decode("utf-8")) if binary_chunk else None
            logger.debug(f"Game state: {upd}")

            u_type = upd["type"] if upd else "ping"
            if u_type == "chatLine":
                conversation.react(ChatLine(upd), game)
            elif u_type == "gameState":
                game.state = upd

                start_time = time.perf_counter_ns()
                if upd["moves"] and len(upd["moves"].split()[-1]) != 4:
                    continue
                moves = upd["moves"].split()
                moves_to_get = len(moves) - len(old_moves)
                if moves_to_get > 0:
                    for move in moves[-moves_to_get:]:
                        board.push_str_move(move)
                old_moves = moves

                if len(board.move_stack) == 0:
                    disconnect_time = correspondence_disconnect_time

                if not is_game_over(board) and is_engine_move(game, prior_game, board):
                    disconnect_time = correspondence_disconnect_time
                    if len(board.move_stack) < 2:
                        conversation.send_message("player", hello)
                        conversation.send_message("spectator", hello_spectators)
                    fake_thinking(config, board, game)
                    print_move_number(board)

                    draw_offered = check_for_draw_offer(game)

                    if len(board.move_stack) < 2:
                        best_move = choose_first_move(engine, board, draw_offered)
                    elif is_correspondence:
                        best_move = choose_move_time(engine, board, correspondence_move_time, draw_offered)
                    else:
                        best_move = get_pondering_results(ponder_thread, ponder_li_one, game, board, engine)
                        if best_move.move is None:
                            best_move = choose_move(engine, board, game, draw_offered, start_time, move_overhead,
                                                    move_overhead_inc)
                    move_attempted = True
                    if best_move.resigned and len(board.move_stack) >= 2:
                        li.resign(game.id)
                    else:
                        li.make_move(game.id, best_move)
                    ponder_thread, ponder_li_one = start_pondering(engine, board, game, can_ponder, best_move,
                                                                   start_time, move_overhead, move_overhead_inc)
                    time.sleep(delay_seconds)
                elif is_game_over(board):
                    engine.report_game_result(game, board)
                    tell_user_game_result(game, board)
                    conversation.send_message("player", goodbye)
                    conversation.send_message("spectator", goodbye_spectators)

                wb = "w" if board.whose_turn() == draughts.WHITE else "b"
                terminate_time = (upd[f"{wb}time"] + upd[f"{wb}inc"]) / 1000 + 60
                game.ping(abort_time, terminate_time, disconnect_time)
                prior_game = copy.deepcopy(game)
            elif u_type == "ping":
                if is_correspondence and not is_engine_move(game, prior_game, board) and game.should_disconnect_now():
                    break
                elif game.should_abort_now():
                    logger.info(f"Aborting {game.url()} by lack of activity")
                    li.abort(game.id)
                    break
                elif game.should_terminate_now():
                    logger.info(f"Terminating {game.url()} by lack of activity")
                    if game.is_abortable():
                        li.abort(game.id)
                    break
        except (HTTPError, ReadTimeout, RemoteDisconnected, ChunkedEncodingError, ConnectionError):
            if move_attempted:
                continue
            if game.id not in (ongoing_game["gameId"] for ongoing_game in li.get_ongoing_games()):
                break
        except StopIteration:
            break

    engine.stop()
    engine.quit()

    try:
        print_pgn_game_record(li, config, game, board, engine)
    except Exception:
        logger.exception("Error writing game record:")

    if is_correspondence and not is_game_over(board):
        logger.info(f"--- Disconnecting from {game.url()}")
        correspondence_queue.put(game_id)
    else:
        logger.info(f"--- {game.url()} Game over")

    control_queue.put_nowait({"type": "local_game_done"})


def parse_variant(variant):
    variant = variant.lower()

    if variant in ["standard", "from position"]:
        return "normal"
    elif variant == "breakthrough":
        return "bt"
    elif variant == "antidraughts":
        return "losing"
    elif variant == "frysk!":
        return "frisian"
    else:
        return variant


def choose_move_time(engine, board, search_time, draw_offered):
    logger.info(f"Searching for time {search_time}")
    return engine.search_for(board, search_time, draw_offered)


def choose_first_move(engine, board, draw_offered):
    # need to hardcode first movetime (10000 ms) since Lidraughts has 30 sec limit.
    search_time = 10000
    logger.info(f"Searching for time {search_time}")
    return engine.first_search(board, search_time, draw_offered)


def choose_move(engine, board, game, draw_offered, start_time, move_overhead, move_overhead_inc):
    pre_move_time = int((time.perf_counter_ns() - start_time) / 1e6)
    overhead = pre_move_time + move_overhead
    wb = "w" if board.whose_turn() == draughts.WHITE else "b"
    game.state[f"{wb}time"] = max(0, game.state[f"{wb}time"] - overhead)
    game.state[f"{wb}inc"] = max(0, game.state[f"{wb}inc"] - move_overhead_inc)
    logger.info("Searching for wtime {wtime} btime {btime}".format_map(game.state))
    return engine.search_with_ponder(board, game.state["wtime"], game.state["btime"], game.state["winc"],
                                     game.state["binc"], False, draw_offered)


def start_pondering(engine, board, game, can_ponder, best_move, start_time, move_overhead, move_overhead_inc):
    if not can_ponder or best_move.ponder is None:
        return None, None

    ponder_board = board.copy()
    for move in best_move.move.board_move:
        ponder_board.move(move)
    for move in best_move.ponder.board_move:
        ponder_board.move(move)

    wtime = game.state["wtime"]
    btime = game.state["btime"]
    winc = game.state["winc"]
    binc = game.state["binc"]
    setup_time = int((time.perf_counter_ns() - start_time) / 1000000)
    if board.whose_turn() == draughts.WHITE:
        wtime = wtime - move_overhead - setup_time + winc
        winc = winc - move_overhead_inc
    else:
        btime = btime - move_overhead - setup_time + binc
        binc = binc - move_overhead_inc

    def ponder_thread_func(game, engine, board, wtime, btime, winc, binc):
        global ponder_results
        best_move = engine.search_with_ponder(board, wtime, btime, winc, binc, True, False)
        ponder_results[game.id] = best_move

    logger.info(f"Pondering for wtime {wtime} btime {btime}")
    ponder_thread = threading.Thread(target=ponder_thread_func, args=(game, engine, ponder_board, wtime, btime, winc, binc))
    ponder_thread.start()
    return ponder_thread, best_move.ponder.li_one_move


def get_pondering_results(ponder_thread, ponder_li_one, game, board, engine):
    no_move = draughts.engine.PlayResult(None, None)
    if ponder_thread is None:
        return no_move

    move_li_one = board.move_stack[-1].li_one_move
    if ponder_li_one == move_li_one:
        engine.ponderhit()
        ponder_thread.join()
        return ponder_results[game.id]
    else:
        engine.stop()
        ponder_thread.join()
        return no_move


def check_for_draw_offer(game):
    return game.state.get(f"{game.opponent_color[0]}draw", False)


def fake_thinking(config, board, game):
    if config.get("fake_think_time") and len(board.move_stack) > 9:
        delay = min(game.clock_initial, game.my_remaining_seconds()) * 0.015
        accel = 1 - max(0, min(100, len(board.move_stack) - 20)) / 150
        sleep = min(5, delay * accel)
        time.sleep(sleep)


def print_move_number(board):
    logger.info("")
    logger.info(f"move: {len(board.move_stack) // 2 + 1}")


def is_engine_move(game, prior_game, board):
    return game_changed(game, prior_game) and game.is_white == (board.whose_turn() == draughts.WHITE)


def is_game_over(board):
    return board.is_over()


def game_changed(current_game, prior_game):
    if prior_game is None:
        return True

    return current_game.state["moves"] != prior_game.state["moves"]


def tell_user_game_result(game, board):
    winner = game.state.get("winner")
    termination = game.state.get("status")

    winning_name = game.white.name if winner == "white" else game.black.name
    losing_name = game.white.name if winner == "black" else game.black.name

    if winner is not None:
        logger.info(f"{winning_name} won!")
    elif termination == engine_wrapper.Termination.DRAW:
        logger.info("Game ended in draw.")
    else:
        logger.info("Game adjourned.")

    if termination == engine_wrapper.Termination.MATE:
        logger.info("Game won by checkmate.")
    elif termination == engine_wrapper.Termination.TIMEOUT:
        logger.info(f"{losing_name} forfeited on time.")
    elif termination == engine_wrapper.Termination.RESIGN:
        logger.info(f"{losing_name} resigned.")
    elif termination == engine_wrapper.Termination.ABORT:
        logger.info("Game aborted.")
    elif termination == engine_wrapper.Termination.DRAW:
        if board.is_fifty_moves():
            logger.info("Game drawn by 50-move rule.")
        elif board.is_repetition():
            logger.info("Game drawn by threefold repetition.")
        else:
            logger.info("Game drawn by agreement.")
    elif termination:
        logger.info(f"Game ended by {termination}")


def print_pgn_game_record(li, config, game, board, engine):
    game_directory = config.get("pgn_directory")
    if not game_directory:
        return

    try:
        os.mkdir(game_directory)
    except FileExistsError:
        pass

    game_file_name = f"{game.white.name} vs {game.black.name} - {game.id}.pgn"
    game_file_name = "".join(c for c in game_file_name if c not in '<>:"/\\|?*')
    game_path = os.path.join(game_directory, game_file_name)

    lidraughts_game_record = li.get_game_pgn(game.id)

    with open(game_path, "w") as game_record_destination:
        game_record_destination.write(lidraughts_game_record)


def intro():
    return fr"""
    .   _/|
    .  // o\
    .  || ._)  lidraughts-bot {__version__}
    .  //__\
    .  )___(   Play on Lidraughts with a bot
    """


def start_lichess_bot():
    parser = argparse.ArgumentParser(description="Play on Lidraughts with a bot")
    parser.add_argument("-u", action="store_true", help="Upgrade your account to a bot account.")
    parser.add_argument("-v", action="store_true", help="Make output more verbose. Include all communication with lichess.")
    parser.add_argument("--config", help="Specify a configuration file (defaults to ./config.yml)")
    parser.add_argument("-l", "--logfile", help="Record all console output to a log file.", default=None)
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.v else logging.INFO
    logging_configurer(logging_level, args.logfile)
    logger.info(intro(), extra={"highlighter": None})
    CONFIG = load_config(args.config or "./config.yml")
    li = lidraughts.Lidraughts(CONFIG["token"], CONFIG["url"], __version__, logging_level)

    user_profile = li.get_profile()
    username = user_profile["username"]
    is_bot = user_profile.get("title") == "BOT"
    logger.info(f"Welcome {username}!")

    if args.u and not is_bot:
        is_bot = upgrade_account(li)

    if is_bot:
        start(li, user_profile, CONFIG, logging_level, args.logfile)
    else:
        logger.error(f"{username} is not a bot account. Please upgrade it to a bot account!")


if __name__ == "__main__":
    try:
        start_lichess_bot()
    except Exception:
        logger.exception("Quitting lichess-bot due to an error:")
