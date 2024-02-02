# Python-color-print
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/>

## Function Signature
```python
def print_Color(Input: str, colors: list, print_END: str = '\n', advanced_mode: bool = False):
```

## Parameters
- `Input` (str): The input string to be printed. In advanced mode, '~*' is used to separate different parts of the string to be printed in different colors.
- `colors` (list): A list of colors for the text. In non-advanced mode, only the first color in the list is used. In advanced mode, each color corresponds to a part of the input string separated by '~*'.
- `print_END` (str): The string appended after the final output, default is '\\n'.
- `advanced_mode` (bool): If True, enables advanced mode that allows multiple colors in one string. Default is False.

## Usage
In **normal mode**, you can print a string in a single color. For example:
```python
print_Color('Hello, World!', ['green']) 
```
This will print 'Hello, World!' in green.

In **advanced mode**, you can print different parts of a string in different colors. For example:
```python
print_Color('~*Hello in green~*Hello in red', ['green', 'red'], advanced_mode=True) 
```
This will print 'Hello in green' in green and 'Hello in red' in red.

## Special Characters
The '~*' characters are used as separators for different parts of the string that need to be printed in different colors when using advanced mode.

## Supported Colors
#### you can use the key word like 'black' and... to set the text color.
~~~
'black': '\x1b[0;30m',
'red': '\x1b[0;31m',
'green': '\x1b[0;32m',
'yellow': '\x1b[0;33m',
'blue': '\x1b[0;34m',
'magenta': '\x1b[0;35m',
'cyan': '\x1b[0;36m',
'white': '\x1b[0;37m',
'normal': '\x1b[0m',
'bg_black': '\x1b[40m',
'bg_red': '\x1b[41m',
'bg_green': '\x1b[42m',
'bg_yellow': '\x1b[43m',
'bg_blue': '\x1b[44m',
'bg_magenta': '\x1b[45m',
'bg_cyan': '\x1b[46m',
'bg_white': '\x1b[47m',
'bg_normal': '\x1b[49m',
'light_gray': '\x1b[0;90m',
'light_red': '\x1b[0;91m',
'light_green': '\x1b[0;92m',
'light_yellow': '\x1b[0;93m',
'light_blue': '\x1b[0;94m',
'light_magenta': '\x1b[0;95m',
'light_cyan': '\x1b[0;96m',
'light_white': '\x1b[0;97m',
'bg_light_gray': '\x1b[0;100m',
'bg_light_red': '\x1b[0;101m',
'bg_light_green': '\x1b[0;102m',
'bg_light_yellow': '\x1b[0;103m',
'bg_light_blue': '\x1b[0;104m',
'bg_light_magenta': '\x1b[0;105m',
'bg_light_cyan': '\x1b[0;106m',
'bg_light_white': '\x1b[0;107m',
'underline': '\x1b[4m',
'bold': '\x1b[1m',
'blink': '\x1b[5m'
~~~
