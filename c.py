import sys
import subprocess

# --- Token Types and Reserved Keywords ---
TOKENS = {
    'IF': 'if',
    'ELSE': 'else',
    'PRINT': 'print',
    'LBRACE': '{',
    'RBRACE': '}',
    'LPAREN': '(',
    'RPAREN': ')',
    'SEMI': ';',
    'COMMA': ','
}

# --- Lexer ---
def lex(input_str):
    tokens = []
    i = 0
    while i < len(input_str):
        c = input_str[i]
        if c.isspace():
            i += 1
        elif c.isdigit():
            num = 0
            while i < len(input_str) and input_str[i].isdigit():
                num = num * 10 + int(input_str[i])
                i += 1
            tokens.append(('NUMBER', num))
        elif c.isalpha():
            ident = []
            while i < len(input_str) and (input_str[i].isalnum() or input_str[i] == '_'):
                ident.append(input_str[i])
                i += 1
            ident = ''.join(ident)
            if ident in TOKENS.values():
                tokens.append((ident.upper(), ident))
            else:
                raise ValueError(f"Unknown identifier: {ident}")
        elif c in '+-*/(){};,<>=!':
            if c == '=' and i+1 < len(input_str) and input_str[i+1] == '=':
                tokens.append(('OP', '=='))
                i += 2
            elif c == '!' and i+1 < len(input_str) and input_str[i+1] == '=':
                tokens.append(('OP', '!='))
                i += 2
            else:
                for tok, val in TOKENS.items():
                    if c == val:
                        tokens.append((tok, c))
                        break
                else:
                    tokens.append(('OP', c))
                i += 1
        else:
            raise ValueError(f"Invalid character: {c}")
    return tokens

# --- AST Nodes ---
class Node:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children or []

# --- Parser ---
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def consume(self, expected_type=None):
        if expected_type and self.tokens[self.pos][0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {self.tokens[self.pos][0]}")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self):
        return self.parse_block()

    def eof(self):
        return self.pos >= len(self.tokens)  # <-- Added this method

    def parse_block(self):
        nodes = []
        while self.peek() != 'RBRACE' and not self.eof():
            nodes.append(self.parse_statement())
        return Node('BLOCK', children=nodes)

    def parse_statement(self):
        token_type = self.peek()
        if token_type == 'IF':
            return self.parse_if()
        elif token_type == 'PRINT':
            return self.parse_print()
        else:
            expr = self.parse_expression()
            self.consume('SEMI')
            return Node('EXPR_STMT', children=[expr])

    def parse_if(self):
        self.consume('IF')
        self.consume('LPAREN')
        condition = self.parse_expression()
        self.consume('RPAREN')
        self.consume('LBRACE')
        then_block = self.parse_block()
        self.consume('RBRACE')
        else_block = None
        if self.peek() == 'ELSE':
            self.consume('ELSE')
            self.consume('LBRACE')
            else_block = self.parse_block()
            self.consume('RBRACE')
        return Node('IF', children=[condition, then_block, else_block])

    def parse_print(self):
        self.consume('PRINT')
        self.consume('LPAREN')
        expr = self.parse_expression()
        self.consume('RPAREN')
        self.consume('SEMI')
        return Node('PRINT', children=[expr])

    def parse_expression(self):
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_add_sub()
        while self.peek() in ['OP'] and self.tokens[self.pos][1] in ['==', '!=', '<', '>', '<=', '>=']:
            op = self.consume('OP')[1]
            right = self.parse_add_sub()
            left = Node('BINOP', op, [left, right])
        return left

    def parse_add_sub(self):
        left = self.parse_mul_div()
        while self.peek() == 'OP' and self.tokens[self.pos][1] in ['+', '-']:
            op = self.consume('OP')[1]
            right = self.parse_mul_div()
            left = Node('BINOP', op, [left, right])
        return left

    def parse_mul_div(self):
        left = self.parse_primary()
        while self.peek() == 'OP' and self.tokens[self.pos][1] in ['*', '/']:
            op = self.consume('OP')[1]
            right = self.parse_primary()
            left = Node('BINOP', op, [left, right])
        return left

    def parse_primary(self):
        token = self.tokens[self.pos]
        if token[0] == 'NUMBER':
            self.consume()
            return Node('NUMBER', token[1])
        elif token[0] == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        else:
            raise ValueError(f"Unexpected token: {token}")

# --- Code Generator ---
class CodeGenerator:
    def __init__(self):
        self.asm = []
        self.label_count = 0

    def new_label(self):
        self.label_count += 1
        return f"L{self.label_count}"

    def generate(self, node):
        if node.type == 'BLOCK':
            for child in node.children:
                self.generate(child)
        elif node.type == 'IF':
            self.generate_if(node)
        elif node.type == 'PRINT':
            self.generate_print(node)
        elif node.type == 'EXPR_STMT':
            self.generate(node.children[0])
        elif node.type == 'BINOP':
            self.generate_binop(node)
        elif node.type == 'NUMBER':
            self.asm.append(f"mov ${node.value}, %rax")

    def generate_if(self, node):
        condition, then_block, else_block = node.children
        else_label = self.new_label()
        end_label = self.new_label()

        # Generate condition
        self.generate(condition)
        self.asm.append(f"cmp $0, %rax")
        self.asm.append(f"je {else_label if else_block else end_label}")

        # Generate then block
        self.generate(then_block)
        self.asm.append(f"jmp {end_label}")

        # Generate else block
        if else_block:
            self.asm.append(f"{else_label}:")
            self.generate(else_block)

        self.asm.append(f"{end_label}:")

    def generate_print(self, node):
        self.generate(node.children[0])
        self.asm.append("mov %rax, %rsi")
        self.asm.append("lea .message(%rip), %rdi")
        self.asm.append("xor %eax, %eax")
        self.asm.append("call printf@PLT")

    def generate_binop(self, node):
        op = node.value
        left, right = node.children
        self.generate(left)
        self.asm.append("push %rax")
        self.generate(right)
        self.asm.append("pop %rcx")

        if op == '+':
            self.asm.append("add %rcx, %rax")
        elif op == '-':
            self.asm.append("sub %rcx, %rax")
        elif op == '*':
            self.asm.append("imul %rcx, %rax")
        elif op == '/':
            self.asm.append("xor %edx, %edx")
            self.asm.append("idiv %rcx")
        elif op in ['==', '!=', '<', '>', '<=', '>=']:
            self.asm.append("cmp %rax, %rcx")
            jump_map = {
                '==': 'je',
                '!=': 'jne',
                '<': 'jl',
                '>': 'jg',
                '<=': 'jle',
                '>=': 'jge'
            }
            jump_instr = jump_map[op]
            true_label = self.new_label()
            end_label = self.new_label()
            self.asm.append(f"{jump_instr} {true_label}")
            self.asm.append("mov $0, %rax")
            self.asm.append(f"jmp {end_label}")
            self.asm.append(f"{true_label}:")
            self.asm.append("mov $1, %rax")
            self.asm.append(f"{end_label}:")

# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compiler.py <source_file>")
        sys.exit(1)

    # Read source file
    try:
        with open(sys.argv[1], 'r') as f:
            input_str = f.read()
    except FileNotFoundError:
        print(f"Error: File '{sys.argv[1]}' not found")
        sys.exit(1)

    # Lexing
    try:
        tokens = lex(input_str)
    except ValueError as e:
        print(f"Lexer error: {e}")
        sys.exit(1)

    # Parsing
    try:
        parser = Parser(tokens)
        ast = parser.parse()
    except ValueError as e:
        print(f"Parser error: {e}")
        sys.exit(1)

    # Code Generation
    generator = CodeGenerator()
    generator.generate(ast)

    # Generate assembly
    asm_code = f"""
    .global main
    .section .text
    main:
        {chr(10).join(generator.asm)}
        xor %eax, %eax
        ret
    .section .rodata
    .message:
        .string "%d\\n"
    """

    # Save and compile
    with open("output.s", "w") as f:
        f.write(asm_code)

    subprocess.run(["gcc", "-no-pie", "output.s", "-o", "output"])

    subprocess.run(["./output"])
    # remove output.s and output
    subprocess.run(["rm", "output.s"])
    subprocess.run(["rm", "output"])
