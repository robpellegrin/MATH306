#!/usr/bin/python3

from sys import argv

"""
Author: Rob Pellegrin
Date    3/28/2025

Simple script to correct the indenting of #pragma directives. Hopefully this is
added to clang-format someday.

The path to the file to be processed should be passes as a single command line
argument.

"""


def main(filename):
    file_contents = []

    # Read the file being processed into a list.
    with open(filename, "r", encoding='UTF-8') as input_file:
        file_contents = input_file.readlines()

    # Loop over the list containing the file's contents.
    for pos in range(len(file_contents)):
        # The current line being processed.
        line = file_contents[pos]

        # Check if the current line contains a pragma directive. If it does,
        # count the leading white space characters of the next line so that
        # the same amount of white space can be appended to the line containing
        # the pgrama.
        if "#pragma" in line and pos + 1 < len(file_contents):
            # Count the leading whitespace of the next line
            leading_whitespace_count = len(
                file_contents[pos + 1]) - len(file_contents[pos + 1].lstrip())

            # Update the current line with the correct number of leading spaces
            file_contents[pos] = (" " * leading_whitespace_count) + line

    # Save the file.
    with open(f"{filename}", 'w', encoding='UTF-8') as output_file:
        for line in file_contents:
            output_file.write(line)


def verify_command_line_arg():
    # Make sure a single argument has been provided.
    if len(argv) != 2:
        print("Error: Expected a path to a single file as an argument.")
        exit

    # The argument provided should be a file with the suffix .c or .cpp. If
    # not, inform the user and exit.
    if ".cpp" not in argv[1] or ".c" not in argv[1]:
        print("Error: Expected .cpp or .c file")
        exit


if __name__ == "__main__":
    verify_command_line_arg()
    main(argv[1])
