import logging
from enum import Enum
from typing import Tuple, List

from boba_python_utils.common_utils.typing_helper import solve_key_value_pairs, is_none_or_empty_str, is_str
from boba_python_utils.general_utils.external.colorama import Fore, Back, Style
from boba_python_utils.general_utils.external.colorama import (
    init as colorama_init,
)
import sys
from boba_python_utils.general_utils.general import (
    is_basic_type,
    is_class,
)

colorama_init()

HPRINT_TITLE_COLOR = Fore.YELLOW
HPRINT_MESSAGE_COLOR = Fore.CYAN
EPRINT_TITLE_COLOR = Fore.MAGENTA
EPRINT_MESSAGE_COLOR = Fore.RED

# region non-colored messages
DEFAULT_TITLE_DECORATION = '===='


def get_title_with_decoration(title: str, title_decor: str = DEFAULT_TITLE_DECORATION) -> str:
    return f'{title_decor}{title}{title_decor}'


def get_titled_message_str(
        title: str,
        content: str = '',
        start: str = '',
        end: str = '\n',
        replacement_for_empty_content='n/a'
):
    """
    Gets string for a titled message.
    Args:
        title: the title text of the message.
        content: the content text of the message.
        start: adds this string to the start of the output (in front of the title).
        end: adds this string to the end of the output (right after the content).
        replacement_for_empty_content: replacement string to display for `content` if
            the `content` is empty.

    Returns: a titled message string.

    """
    if content is None or content == '':
        content = replacement_for_empty_content
    return (
        f'{start}{title}{end}'
        if (not content)
        else f'{start}{title}: {content}{end}'
    )


def get_pairs_message_str(
        *args,
        title=None,
        comment=None,
        sep='\t',
        start='',
        end='\n'
):
    return (
            (f'[{title}]{sep}' if title else '') +
            (f'[{comment}]{sep}' if comment else '') +
            sep.join(  # noqa: E126
                (
                    get_titled_message_str(
                        title=arg[0],
                        content=arg[1],
                        start=start,
                        end='',
                    )
                    for arg in args
                )
            )
            + end
    )


# endregion

# region colored messages
def get_cprint_titled_message_str(
        title: str,
        content: str = '',
        title_color=Fore.CYAN,
        content_color=Fore.WHITE,
        start: str = '',
        end: str = '\n',
        replacement_for_empty_content='n/a'
) -> str:
    """
    Gets a colored titled message string for terminal print.
    Args:
        title: the title text of the message.
        content: the content text of the message.
        title_color: the title text color.
        content_color: the content text color.
        start: adds this string to the start of the output (in front of the title).
        end: adds this string to the end of the output (right after the content).
        replacement_for_empty_content: replacement string to display for `content` if
            the `content` is empty.

    Returns: a colored titled string for terminal print according to the arguments.

    """
    if is_none_or_empty_str(content):
        content = replacement_for_empty_content

    if content is not None and not is_str(content):
        # ! do the conversion first; string formatting does not support some classes
        content = str(content)

    return (
        f'{start}{title_color}{title}{Fore.WHITE}{end}'
        if (not content)
        else f'{start}{title_color}{title}: {content_color}{content}{Fore.WHITE}{end}'
    )


def get_cprint_section_title_str(
        title: str,
        title_color=Fore.YELLOW,
        title_style=Style.BOLD,
        title_decoration=DEFAULT_TITLE_DECORATION,
) -> str:
    return (
        f'\n{title_style}{title_color}'
        f'{get_title_with_decoration(title, title_decoration)}'
        f'{Style.RESET_ALL}{Fore.WHITE}\n\n'
        if title else ''
    )


def get_cprint_section_separator(
        title_color=Fore.YELLOW,
        title_style=Style.BOLD
):
    return (
        f'\n{title_style}{title_color}'
        f'----'
        f'{Style.RESET_ALL}{Fore.WHITE}\n'
    )


def get_cprint_pairs_message_str(
        *args,
        title=None,
        comment=None,
        first_color=Fore.CYAN,
        second_color=Fore.WHITE,
        title_color=Fore.YELLOW,
        title_style=Style.BOLD,
        title_decoration=DEFAULT_TITLE_DECORATION,
        sep='\t',
        start='',
        end='\n',
        replacement_for_empty_content='n/a',
        output_title_and_contents: List = None
):
    key_value_pairs = list(solve_key_value_pairs(args))
    if output_title_and_contents is not None:
        ks, vs = zip(*key_value_pairs)
        if title:
            if not output_title_and_contents:
                output_title_and_contents.append(('title', *ks))
            output_title_and_contents.append(tuple((title, *vs)))
        else:
            if not output_title_and_contents:
                output_title_and_contents.append(ks)
            output_title_and_contents.append(vs)

    return (
            get_cprint_section_title_str(
                title=title,
                title_color=title_color,
                title_style=title_style,
                title_decoration=title_decoration
            ) +
            (
                f'{comment}\n' if comment else ''
            ) +
            sep.join(
                (
                    get_cprint_titled_message_str(
                        title=k,
                        content=v,
                        title_color=first_color,
                        content_color=second_color,
                        start=start,
                        end='',
                        replacement_for_empty_content=replacement_for_empty_content
                    )
                    for k, v in key_value_pairs
                )
            )
            + (
                get_cprint_section_separator(title_color, title_style)
                if title else ''
            )
            + end
    )


def get_pair_strs_for_color_print_and_regular_print(
        *args,
        first_color=Fore.CYAN,
        second_color=Fore.WHITE,
        sep: str = ' ',
        end: str = '\n'
) -> Tuple[str, str]:
    # ! DEPRECATED
    # we will move to Universal Logging that support colored terminal print out and
    # message logging at the same time
    colored_strs, uncolored_strs = [], []
    for arg_idx, arg in enumerate(args):
        colored_strs.append(
            get_cprint_titled_message_str(
                title=arg[0],
                content=arg[1],
                title_color=first_color,
                content_color=second_color,
                end='',
            )
        )
        uncolored_strs.append(
            f'{arg[0]}: {arg[1]},' if arg_idx != len(args) - 1 else f'{arg[0]}: {arg[1]}'
        )

    return sep.join(colored_strs) + end, sep.join(uncolored_strs) + end


def _get_cprint_str(
        text: str,
        color_quote: str = '`',
        color: str = Fore.CYAN,
        bk_color: str = Fore.WHITE,
        end: str = '\n',
):
    if not isinstance(text, str):
        text = str(text)
    output = [Fore.WHITE]
    color_start: bool = True
    prev_color_quote: bool = False
    for c in text:
        if c == color_quote:
            if prev_color_quote:
                output.append('`')
                prev_color_quote = False
                color_start = True
            elif color_start:
                prev_color_quote = True
                color_start = False
            else:
                output.append(bk_color)
                color_start = True
        else:
            if prev_color_quote:
                output.append(color)
            output.append(c)
            prev_color_quote = False
    if end is not None:
        output.append(end)
    output.append(Fore.WHITE)
    return ''.join(output)


# endregion

# region cprint
def cprint(text, color_place_holder='`', color=Fore.CYAN, bk_color=Fore.WHITE, end='\n'):
    print(
        _get_cprint_str(
            text=text, color_quote=color_place_holder, color=color, bk_color=bk_color, end=end
        )
    )


def cprint_message(
        title, content='', title_color=Fore.CYAN, content_color=Fore.WHITE, start='', end='\n'
):
    print(get_cprint_titled_message_str(title, content, title_color, content_color, start, end))


def cprint_pairs(
        *args,
        title=None,
        comment=None,
        first_color=Fore.CYAN,
        second_color=Fore.WHITE,
        title_color=Fore.YELLOW,
        title_stype=Style.BOLD,
        title_decoration=DEFAULT_TITLE_DECORATION,
        sep=' ',
        start='',
        end='\n',
        replacement_for_empty_content='n/a',
        logger: logging.Logger = None,
        output_title_and_contents: List = None
):
    print(
        get_cprint_pairs_message_str(
            *args,
            title=title,
            comment=comment,
            first_color=first_color,
            second_color=second_color,
            title_color=title_color,
            title_style=title_stype,
            title_decoration=title_decoration,
            sep=sep,
            start=start,
            end=end,
            replacement_for_empty_content=replacement_for_empty_content,
            output_title_and_contents=output_title_and_contents
        )
    )
    if logger is not None:
        logger.info(
            msg=get_pairs_message_str(
                *args,
                title=title,
                comment=comment,
                sep=sep,
                start=start,
                end=end
            )
        )


# endregion

# region hprint

def get_pairs_str_for_hprint_and_regular_print(
        *args, sep: str = ' ', end: str = '\n'
) -> Tuple[str, str]:
    return get_pair_strs_for_color_print_and_regular_print(
        *args,
        first_color=HPRINT_MESSAGE_COLOR,
        second_color=Fore.WHITE,
        sep=sep,
        end=end
    )


def hprint(msg, color_quote='`', end=''):
    """
    Print the message `msg`, highlighting texts enclosed
        by a pair of `color_quote`s (by default the backtick `) with the cyan color.
    Use two backticks '``' to escape the color quote.
    :param msg: the message to print.
    :param color_quote: the character used to mark the beginning
            and the end of each piece of texts to highlight.
    :param end: string appended at the end of the message, newline by default.
    """
    cprint(
        text=msg,
        color_place_holder=color_quote,
        color=HPRINT_MESSAGE_COLOR,
        bk_color=Fore.WHITE,
        end=end
    )


def get_hprint_section_title_str(title: str) -> str:
    return get_cprint_section_title_str(
        title=title,
        title_color=HPRINT_TITLE_COLOR,
        title_style=Style.BOLD,
        title_decoration=DEFAULT_TITLE_DECORATION
    )


def get_hprint_section_separator() -> str:
    return get_cprint_section_separator(
        title_color=HPRINT_TITLE_COLOR,
        title_style=Style.BOLD
    )


def hprint_section_title(title: str):
    print(get_hprint_section_title_str(title))


def hprint_section_separator():
    print(get_hprint_section_separator())


def get_hprint_message_str(title, content='', start='', end=''):
    return get_cprint_titled_message_str(
        title=title,
        content=content,
        title_color=HPRINT_MESSAGE_COLOR,
        content_color=Fore.WHITE,
        start=start,
        end=end,
    )


def hprint_pairs(
        *args,
        title=None,
        comment=None,
        sep=' ',
        start='',
        end='',
        logger:
        logging.Logger = None,
        replacement_for_empty_content='n/a',
        output_title_and_contents: List = None
):
    cprint_pairs(
        *args,
        title=title,
        comment=comment,
        first_color=HPRINT_MESSAGE_COLOR,
        second_color=Fore.WHITE,
        title_color=HPRINT_TITLE_COLOR,
        title_stype=Style.BOLD,
        sep=sep,
        start=start,
        end=end,
        replacement_for_empty_content=replacement_for_empty_content,
        logger=logger,
        output_title_and_contents=output_title_and_contents
    )


def hprint_message(
        *msg_pairs,
        title: str = '',
        content: str = '',
        start: str = '',
        end: str = '',
        sep: str = '\n',
        logger: logging.Logger = None,
        replacement_for_empty_content: str = 'n/a',
        output_title_and_contents: List = None
):
    """
    Highlight-print one or more messages.

    If there is only one message to print, then specify `title` and `content`,
    and `title` will be highlighted. We may also specify both `title` and `content`
    as unnamed arguments.

    Examples:

        >>> hprint_message(title='title', content='this is a message')
        >>> hprint_message('title', 'this is a message')

    If there are multiple messages, specify them as a sequence of tuples;
    we can omit the tuple brackets for convenience.

    In this case, `title` can be used to specify a big title for all the messages,
    and `content` will be displayed as a piece of text following the title.

    Examples:
        >>> hprint_message(
        ...   ('title1', 'this is message1'),
        ...   ('title2', 'this is message2'),
        ...   ('title3', 'this is message3')
        ... )

        >>> hprint_message(
        ...   'title1', 'this is message1',
        ...   'title2', 'this is message2',
        ...   'title3', 'this is message3'
        ... )

        >>> hprint_message(
        ...   'title1', 'this is message1',
        ...   'title2', 'this is message2',
        ...   'title3', 'this is message3',
        ...   title='Big Title',
        ...   content='This is a comment for the messages.'
        ... )

    A non-string `content` object will be converted to string.

    Examples:
        >>> hprint_message(
        ...   'metric1', 1,
        ...   'metric2', 4,
        ...   'metric3', 23.45,
        ...   title='Performance Metrics',
        ...   content='All the performance metrics should be non-zero.'
        ... )

    Args:
        *msg_pairs: specify a sequence of tuples if we have more than one messages;
            if this is not a sequence of tuples, every two adjacent objects will be treated
            as a tuple.
        title: specify the title for a single message,
            or specify the big title for multiple messages.
        content: specify the content of a single message,
            or specify a comment for multiple messages.
        start: this is added to the start of each message.
        end: this is added to the end of each message.
        logger: provide a logger to log the current message.

    """
    if msg_pairs:
        hprint_pairs(
            *solve_key_value_pairs(*msg_pairs),
            title=title,
            comment=content,
            sep=sep,
            start=start,
            end=end,
            replacement_for_empty_content=replacement_for_empty_content,
            logger=logger,
            output_title_and_contents=output_title_and_contents
        )
    else:
        if output_title_and_contents is not None:
            if title:
                output_title_and_contents.append((title, content))
            else:
                output_title_and_contents.append(content)
        print(
            get_hprint_message_str(
                title=title,
                content=content,
                start=start,
                end=end)
        )
        if logger is not None:
            logger.info(
                msg=get_titled_message_str(
                    title=title,
                    content=content,
                    start=start,
                    end=end,
                    replacement_for_empty_content=replacement_for_empty_content
                )
            )


# endregion

# region eprint
def eprint(text, color_quote='`', end='\n'):
    """
    Print the message `msg` with the , highlighting texts enclosed
        by a pair of `color_quote`s (by default the backtick `) with the red color.
    :param msg: the message to print.
    :param color_quote: the character used to mark the beginning
            and the end of each piece of texts to highlight.
    :param end: string appended at the end of the message, newline by default.
    """
    cprint(
        text=text, color_place_holder=color_quote, color=Fore.RED, bk_color=Fore.MAGENTA, end=end
    )


def get_eprint_message_str(title, content='', start='', end=''):
    return get_cprint_titled_message_str(
        title=title,
        content=content,
        title_color=EPRINT_TITLE_COLOR,
        content_color=EPRINT_MESSAGE_COLOR,
        start=start,
        end=end,
    )


def eprint_pairs(
        *args,
        title=None,
        comment=None,
        sep=' ',
        start='',
        end='',
        logger:
        logging.Logger = None,
        replacement_for_empty_content='n/a'
):
    cprint_pairs(
        *args,
        title=title,
        comment=comment,
        first_color=EPRINT_MESSAGE_COLOR,
        second_color=Fore.WHITE,
        title_color=EPRINT_TITLE_COLOR,
        title_stype=Style.BOLD,
        sep=sep,
        start=start,
        end=end,
        replacement_for_empty_content=replacement_for_empty_content,
        logger=logger
    )


def eprint_message(
        *msg_pairs,
        title='',
        content='',
        start='',
        end='',
        sep='\n',
        logger: logging.Logger = None,
        replacement_for_empty_content='n/a'
):
    if msg_pairs:
        eprint_pairs(
            *solve_key_value_pairs(*msg_pairs),
            title=title,
            comment=content,
            sep=sep,
            start=start,
            end=end,
            replacement_for_empty_content=replacement_for_empty_content,
            logger=logger
        )
    else:
        print(
            get_eprint_message_str(
                title=title,
                content=content,
                start=start,
                end=end)
        )
        if logger is not None:
            logger.error(
                msg=get_titled_message_str(
                    title=title,
                    content=content,
                    start=start,
                    end=end,
                    replacement_for_empty_content=replacement_for_empty_content
                )
            )


# endregion

# region wprint

def wprint(text, color_quote='`', end='\n'):
    """
    Print the message `msg` with the , highlighting texts enclosed
        by a pair of `color_quote`s (by default the backtick `) with the red color.
    :param msg: the message to print.
    :param color_quote: the character used to mark the beginning
            and the end of each piece of texts to highlight.
    :param end: string appended at the end of the message, newline by default.
    """
    cprint(
        text=text, color_place_holder=color_quote, color=Fore.RED, bk_color=Fore.YELLOW, end=end
    )


def get_eprint_message_str(title, content='', start='', end=''):
    return get_cprint_titled_message_str(
        title=title,
        content=content,
        title_color=EPRINT_TITLE_COLOR,
        content_color=EPRINT_MESSAGE_COLOR,
        start=start,
        end=end,
    )


def eprint_pairs(
        *args,
        title=None,
        comment=None,
        sep=' ',
        start='',
        end='',
        logger:
        logging.Logger = None,
        replacement_for_empty_content='n/a'
):
    cprint_pairs(
        *args,
        title=title,
        comment=comment,
        first_color=EPRINT_MESSAGE_COLOR,
        second_color=Fore.WHITE,
        title_color=EPRINT_TITLE_COLOR,
        title_stype=Style.BOLD,
        sep=sep,
        start=start,
        end=end,
        replacement_for_empty_content=replacement_for_empty_content,
        logger=logger
    )


def eprint_message(
        *msg_pairs,
        title='',
        content='',
        start='',
        end='',
        sep='\n',
        logger: logging.Logger = None,
        replacement_for_empty_content='n/a'
):
    if msg_pairs:
        eprint_pairs(
            *solve_key_value_pairs(*msg_pairs),
            title=title,
            comment=content,
            sep=sep,
            start=start,
            end=end,
            replacement_for_empty_content=replacement_for_empty_content,
            logger=logger
        )
    else:
        print(
            get_eprint_message_str(
                title=title,
                content=content,
                start=start,
                end=end)
        )
        if logger is not None:
            logger.error(
                msg=get_titled_message_str(
                    title=title,
                    content=content,
                    start=start,
                    end=end,
                    replacement_for_empty_content=replacement_for_empty_content
                )
            )


# endregion

def wprint_message(title, content='', start='', end='\n'):
    cprint_message(
        title, content, title_color=Fore.MAGENTA, content_color=Fore.YELLOW, start=start, end=end
    )


class flogger(object):
    def __init__(self, path, print_terminal=True):
        self.terminal = sys.stdout
        self.log = open(path, "w")
        self.print_to_terminal = print_terminal

    def write(self, message):
        if self.print_to_terminal:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        pass

    def reset(self):
        self.flush()
        self.log.close()
        sys.stdout = self.terminal


def color_print_pair_str(
        pair_str: str,
        pair_delimiter=',',
        kv_delimiter=':',
        key_color=Fore.CYAN,
        value_color=Fore.WHITE,
        end='\n',
):
    pairs = pair_str.split(pair_delimiter)
    pair_count = len(pairs)
    for i in range(pair_count):
        kv_str = pairs[i].strip()
        if len(kv_str) > 0:
            k, v = kv_str.split(kv_delimiter, maxsplit=2)
            cprint_message(k, v, key_color, value_color, end=', ' if i != pair_count - 1 else end)


def _get_print_tag_str(tag):
    if is_class(tag):
        return tag.__module__ + '.' + tag.__name__
    elif is_basic_type(tag):
        return str(tag)
    else:
        return tag.__class__


def retrieve_and_print_attrs(obj, *attr_names):
    num_attr_names = len(attr_names)
    attr_vals = [None] * num_attr_names
    for i in range(num_attr_names):
        attr_name = attr_names[i]
        attr_val = getattr(obj, attr_name)
        hprint_message(attr_name, attr_val)
        attr_vals[i] = attr_val
    return tuple(attr_vals)


def print_attrs(obj):
    for attr in dir(obj):
        if attr[0] != '_':
            attr_val = getattr(obj, attr)
            if not callable(attr_val):
                hprint_message(attr, attr_val)


def hprint_message_pair_str(pair_str, pair_delimiter=',', kv_delimiter=':'):
    color_print_pair_str(
        pair_str,
        pair_delimiter=pair_delimiter,
        kv_delimiter=kv_delimiter,
        key_color=Fore.YELLOW,
        value_color=Fore.WHITE,
    )


def log_pairs(logging_fun, *args):
    msg = ' '.join(str(arg_tup[0]) + ' ' + str(arg_tup[1]) for arg_tup in args)
    logging_fun(msg)


def info_print(tag, content):
    if not hasattr(tag, '_verbose') or getattr(tag, '_verbose') is True:
        cprint_message(_get_print_tag_str(tag), content, title_color=Fore.CYAN)


def debug_print(tag, content):
    if not hasattr(tag, '_verbose') or getattr(tag, '_verbose') is True:
        cprint_message(_get_print_tag_str(tag), content, title_color=Fore.YELLOW)
