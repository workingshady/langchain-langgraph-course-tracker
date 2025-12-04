import sys


# Enhanced color and style codes for better logging
class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def _log(msg: str, prefix: str, color: str, stream=sys.stdout):
    formatted = f"{color}{prefix}  {msg}{Colors.END}"
    print(formatted, file=stream)


def log_info(message: str, *, color: str = Colors.CYAN):
    """Log info message with customizable color."""
    _log(message, "[i]", color, stream=sys.stdout)


def log_success(message: str):
    """Log success message in green."""
    _log(message, "[✔]", Colors.GREEN, stream=sys.stdout)


def log_error(message: str):
    """Log error message in red, to stderr."""
    _log(message, "[✖]", Colors.RED, stream=sys.stderr)


def log_warning(message: str):
    """Log warning message in yellow."""
    _log(message, "[!]", Colors.YELLOW, stream=sys.stdout)


def log_header(message: str):
    """Log a prominent header."""
    border = '=' * 70
    header = f"{Colors.BOLD}{Colors.PURPLE}{border}{Colors.END}\n"
    header += f"{Colors.BOLD}{Colors.PURPLE}[#] {message.upper()}{Colors.END}\n"
    header += f"{Colors.BOLD}{Colors.PURPLE}{border}{Colors.END}"
    print("\n" + header + "\n")

