import sys
import subprocess

# Token types and reserved keywords
TOKENS = {
    'IF': 'if',
    'ELSE': 'else',
    'WHILE': 'while',
    'PRINT': 'print',
    'LET': 'let',
    'LBRACE': '{',
    'RBRACE': '}',
    'LPAREN': '(',
    'RPAREN': ')',
    'SEMI': ';',
    'COMMA': ',',
    'ASSIGN': '='
}

# Lexer
def lex(input_str):
    tokens = []
    i = 0
    while i < len(input_str):
        c = input_str[i]
        if c.isspace():
            i += 1
        elif c == '"':
            # String literal
            i += 1
            string = []
            while i < len(input_str) and input_str[i] != '"':
                if input_str[i] == '\\':
                    i += 1
                    if i >= len(input_str):
                        raise ValueError("Unterminated string")
                    string.append(input_str[i])
                else:
                    string.append(input_str[i])
                i += 1
            if i >= len(input_str):
                raise ValueError("Unterminated string")
            i += 1  # Skip closing "
            tokens.append(('STRING', ''.join(string)))
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
                tokens.append(('IDENT', ident))
        elif c in '+-*/(){};,<>=!':
            # Handle multi-character operators first
            if c == '=' and i+1 < len(input_str) and input_str[i+1] == '=':
                tokens.append(('OP', '=='))
                i += 2
            elif c == '!' and i+1 < len(input_str) and input_str[i+1] == '=':
                tokens.append(('OP', '!='))
                i += 2
            elif c == '<' and i+1 < len(input_str) and input_str[i+1] == '=':
                tokens.append(('OP', '<='))
                i += 2
            elif c == '>' and i+1 < len(input_str) and input_str[i+1] == '=':
                tokens.append(('OP', '>='))
                i += 2
            else:
                # Handle single-character tokens
                found = False
                for tok_type, tok_value in TOKENS.items():
                    if tok_value == c:
                        tokens.append((tok_type, c))
                        found = True
                        break
                if not found:
                    tokens.append(('OP', c))
                i += 1
        else:
            raise ValueError(f"Invalid character: {c}")
    return tokens

# AST Nodes
class Node:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children or []

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.vars = set()
    
    def peek(self):
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None
    
    def eof(self):
        return self.pos >= len(self.tokens)
    
    def consume(self, expected_type=None):
        if expected_type and self.tokens[self.pos][0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {self.tokens[self.pos][0]}")
        token = self.tokens[self.pos]
        self.pos += 1
        return token
    
    def parse(self):
        return self.parse_block()
    
    def parse_block(self):
        nodes = []
        while self.peek() not in ['RBRACE', None] and not self.eof():
            nodes.append(self.parse_statement())
        return Node('BLOCK', children=nodes)
    
    def parse_statement(self):
        token_type = self.peek()
        if token_type == 'LET':
            return self.parse_var_decl()
        elif token_type == 'WHILE':
            return self.parse_while()
        elif token_type == 'IF':
            return self.parse_if()
        elif token_type == 'PRINT':
            return self.parse_print()
        elif token_type == 'IDENT' and self.pos+1 < len(self.tokens) and self.tokens[self.pos+1][0] == 'ASSIGN':
            return self.parse_var_assign()
        else:
            expr = self.parse_expression()
            self.consume('SEMI')
            return Node('EXPR_STMT', children=[expr])
    
    def parse_var_decl(self):
        self.consume('LET')
        name = self.consume('IDENT')[1]
        self.consume('ASSIGN')
        expr = self.parse_expression()
        self.consume('SEMI')
        self.vars.add(name)
        return Node('VAR_DECL', name, [expr])
    
    def parse_var_assign(self):
        name = self.consume('IDENT')[1]
        self.consume('ASSIGN')
        expr = self.parse_expression()
        self.consume('SEMI')
        if name not in self.vars:
            raise ValueError(f"Undefined variable: {name}")
        return Node('VAR_ASSIGN', name, [expr])
    
    def parse_while(self):
        self.consume('WHILE')
        self.consume('LPAREN')
        condition = self.parse_expression()
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = self.parse_block()
        self.consume('RBRACE')
        return Node('WHILE', children=[condition, body])
    
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
        arg = self.parse_print_arg()
        self.consume('RPAREN')
        self.consume('SEMI')
        return Node('PRINT', children=[arg])
    
    def parse_print_arg(self):
        if self.peek() == 'STRING':
            token = self.consume('STRING')
            return Node('STRING', token[1])
        else:
            return self.parse_expression()
    
    def parse_expression(self):
        return self.parse_comparison()
    
    def parse_comparison(self):
        left = self.parse_add_sub()
        while self.peek() == 'OP' and self.tokens[self.pos][1] in ['==', '!=', '<', '>', '<=', '>=']:
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
        elif token[0] == 'IDENT':
            name = token[1]
            if name not in self.vars:
                raise ValueError(f"Undefined variable: {name}")
            self.consume()
            return Node('VAR', name)
        elif token[0] == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        else:
            raise ValueError(f"Unexpected token: {token}")

# Code Generator
class CodeGenerator:
    def __init__(self):
        self.asm = []
        self.label_count = 0
        self.vars = {}
        self.stack_size = 0
        self.strings = {}
        self.string_counter = 0
    
    def new_label(self):
        self.label_count += 1
        return f"L{self.label_count}"
    
    def new_string_label(self):
        self.string_counter += 1
        return f".STR{self.string_counter}"
    
    def prologue(self):
        return f"""
        push %rbp
        mov %rsp, %rbp
        sub ${self.stack_size}, %rsp
        """
    
    def epilogue(self):
        return f"""
        mov %rbp, %rsp
        pop %rbp
        ret
        """
    
    def generate(self, node):
        if node.type == 'BLOCK':
            for child in node.children:
                self.generate(child)
        elif node.type == 'VAR_DECL':
            var_name = node.value
            if var_name in self.vars:
                raise ValueError(f"Duplicate variable: {var_name}")
            self.vars[var_name] = self.stack_size
            self.stack_size += 8
            self.generate(node.children[0])
            self.asm.append(f"mov %rax, -{self.vars[var_name]+8}(%rbp)")
        elif node.type == 'VAR_ASSIGN':
            var_name = node.value
            self.generate(node.children[0])
            self.asm.append(f"mov %rax, -{self.vars[var_name]+8}(%rbp)")
        elif node.type == 'VAR':
            self.asm.append(f"mov -{self.vars[node.value]+8}(%rbp), %rax")
        elif node.type == 'WHILE':
            cond_label = self.new_label()
            end_label = self.new_label()
            self.asm.append(f"{cond_label}:")
            self.generate(node.children[0])
            self.asm.append("cmp $0, %rax")
            self.asm.append(f"je {end_label}")
            self.generate(node.children[1])
            self.asm.append(f"jmp {cond_label}")
            self.asm.append(f"{end_label}:")
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
        
        self.generate(condition)
        self.asm.append("cmp $0, %rax")
        self.asm.append(f"je {else_label if else_block else end_label}")
        self.generate(then_block)
        self.asm.append(f"jmp {end_label}")
        
        if else_block:
            self.asm.append(f"{else_label}:")
            self.generate(else_block)
        
        self.asm.append(f"{end_label}:")
    
    def generate_print(self, node):
        arg = node.children[0]
        
        if arg.type == 'STRING':
            label = self.new_string_label()
            self.strings[label] = arg.value
            self.asm.append(f"lea {label}(%rip), %rsi")
            self.asm.append("lea .fmt_str(%rip), %rdi")
        else:
            self.generate(arg)
            self.asm.append("mov %rax, %rsi")
            self.asm.append("lea .fmt_num(%rip), %rdi")
        
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

# Main
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compiler.py <source_file>")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], 'r') as f:
            input_str = f.read()
    except FileNotFoundError:
        print(f"Error: File '{sys.argv[1]}' not found")
        sys.exit(1)
    
    try:
        tokens = lex(input_str)
    except ValueError as e:
        print(f"Lexer error: {e}")
        sys.exit(1)
    
    try:
        parser = Parser(tokens)
        ast = parser.parse()
    except ValueError as e:
        print(f"Parser error: {e}")
        sys.exit(1)
    
    generator = CodeGenerator()
    generator.generate(ast)
    
    asm_code = f"""
    .global main
    .section .text
    main:
        {generator.prologue()}
        {chr(10).join(generator.asm)}
        {generator.epilogue()}
    .section .rodata
    .fmt_num:
        .string "%d\\n"
    .fmt_str:
        .string "%s\\n"
    """
    
    # Add string literals
    for label, content in generator.strings.items():
        escaped = content.replace('"', '\\"').replace('\\', '\\\\')
        asm_code += f"{label}:\n    .string \"{escaped}\"\n"
    
    with open("output.s", "w") as f:
        f.write(asm_code)
    
    # Compile and run
    compile_result = subprocess.run(["gcc", "-no-pie", "output.s", "-o", "output"])
    if compile_result.returncode == 0:
        print("Compilation successful. Running program:")
        subprocess.run(["./output"])
    else:
        print("Compilation failed")