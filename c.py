import sys
import subprocess
from llvmlite import ir, binding

TOKENS = {
    'IF': 'if', 'ELSE': 'else', 'WHILE': 'while', 'PRINT': 'print',
    'LET': 'let', 'LBRACE': '{', 'RBRACE': '}', 'LPAREN': '(',
    'RPAREN': ')', 'SEMI': ';', 'COMMA': ',', 'ASSIGN': '='
}

def lex(input_str):
    tokens = []
    i = 0
    while i < len(input_str):
        c = input_str[i]
        if c.isspace():
            i += 1
        elif c == '"':
            i += 1
            string = []
            while i < len(input_str) and input_str[i] != '"':
                if input_str[i] == '\\':
                    i += 1
                    if i >= len(input_str):
                        raise ValueError("Unterminated string")
                    if input_str[i] == 'n':
                        string.append('\n')
                    elif input_str[i] == 't':
                        string.append('\t')
                    elif input_str[i] == '\\':
                        string.append('\\')
                    elif input_str[i] == '"':
                        string.append('"')
                    else:
                        raise ValueError(f"Invalid escape sequence: \\{input_str[i]}")
                    i += 1
                else:
                    string.append(input_str[i])
                    i += 1
            if i >= len(input_str):
                raise ValueError("Unterminated string")
            i += 1
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

class Node:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children or []

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.vars = {}

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
        self.vars[name] = 'str' if expr.type == 'STRING' else 'int'
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
        arg = self.parse_expression()
        self.consume('RPAREN')
        self.consume('SEMI')
        return Node('PRINT', children=[arg])

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
        elif token[0] == 'STRING':
            self.consume()
            return Node('STRING', token[1])
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

class LLVMCodeGenerator:
    def __init__(self):
        self.module = ir.Module()
        self.builder = None
        self.func = None
        self.vars = {}
        self.printf = None
        self.string_counter = 0
        self.fmt_counter = 0  # Add format string counter

        # Initialize LLVM
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        # Create global format strings once
        self.fmt_num = self.create_global_fmt("%d")
        self.fmt_str = self.create_global_fmt("%s")

    def create_global_fmt(self, fmt):
        """Create a global format string with unique name"""
        self.fmt_counter += 1
        name = f".fmt{self.fmt_counter}"
        fmt_val = fmt + '\0'
        fmt_const = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt_val)),
                                bytearray(fmt_val.encode()))
        global_fmt = ir.GlobalVariable(self.module, fmt_const.type, name=name)
        global_fmt.linkage = 'internal'
        global_fmt.global_constant = True
        global_fmt.initializer = fmt_const
        return global_fmt

    def declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def generate(self, ast):
        self.declare_printf()

        # Create main function
        func_type = ir.FunctionType(ir.IntType(32), [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)

        self.visit(ast)

        # Return 0 from main
        self.builder.ret(ir.Constant(ir.IntType(32), 0))
        return str(self.module)

    def visit(self, node):
        method_name = f'visit_{node.type}'
        return getattr(self, method_name)(node)

    def visit_BLOCK(self, node):
        for child in node.children:
            self.visit(child)

    def visit_VAR_DECL(self, node):
        var_name = node.value
        expr = self.visit(node.children[0])

        # Allocate space on the stack
        if isinstance(expr.type, ir.IntType):
            ptr = self.builder.alloca(expr.type)
            self.vars[var_name] = ptr
            self.builder.store(expr, ptr)
        else:  # String type
            ptr = self.builder.alloca(expr.type)
            self.vars[var_name] = ptr
            self.builder.store(expr, ptr)

    def visit_VAR_ASSIGN(self, node):
        var_name = node.value
        expr = self.visit(node.children[0])
        ptr = self.vars[var_name]
        self.builder.store(expr, ptr)

    def visit_WHILE(self, node):
        loop_block = self.func.append_basic_block("loop")
        after_block = self.func.append_basic_block("after_loop")
        cond_block = self.func.append_basic_block("condition")

        # Initial jump to condition
        self.builder.branch(cond_block)

        # Condition block
        self.builder.position_at_end(cond_block)
        cond_value = self.visit(node.children[0])
        self.builder.cbranch(cond_value, loop_block, after_block)

        # Loop block
        self.builder.position_at_end(loop_block)
        self.visit(node.children[1])
        self.builder.branch(cond_block)

        # After loop block
        self.builder.position_at_end(after_block)

    def visit_IF(self, node):
        condition, then_block, else_block = node.children

        then_bb = self.func.append_basic_block("then")
        else_bb = self.func.append_basic_block("else") if else_block else None
        merge_bb = self.func.append_basic_block("merge")

        # Evaluate condition
        cond_value = self.visit(condition)

        # Create branch
        if else_block:
            self.builder.cbranch(cond_value, then_bb, else_bb)
        else:
            self.builder.cbranch(cond_value, then_bb, merge_bb)

        # Then block
        self.builder.position_at_end(then_bb)
        self.visit(then_block)
        self.builder.branch(merge_bb)

        # Else block
        if else_block:
            self.builder.position_at_end(else_bb)
            self.visit(else_block)
            self.builder.branch(merge_bb)

        # Merge block
        self.builder.position_at_end(merge_bb)

    def visit_PRINT(self, node):
        value = self.visit(node.children[0])

        # Get correct format string
        if isinstance(value.type, ir.IntType):
            fmt_ptr = self.builder.bitcast(self.fmt_num, ir.IntType(8).as_pointer())
        else:
            fmt_ptr = self.builder.bitcast(self.fmt_str, ir.IntType(8).as_pointer())

        # Call printf
        self.builder.call(self.printf, [fmt_ptr, value])

    def visit_BINOP(self, node):
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        op = node.value

        if op in ['+', '-', '*', '/']:
            if op == '+':
                return self.builder.add(left, right)
            elif op == '-':
                return self.builder.sub(left, right)
            elif op == '*':
                return self.builder.mul(left, right)
            elif op == '/':
                return self.builder.sdiv(left, right)
        elif op in ['<', '>', '<=', '>=', '==', '!=']:
            return self.builder.icmp_signed({
                '<': '<=', '>': '>', '<=': '<=', '>=': '>=',
                '==': '==', '!=': '!='
            }[op], left, right)
        else:
            raise ValueError(f"Unknown operator {op}")

    def visit_NUMBER(self, node):
        return ir.Constant(ir.IntType(32), node.value)

    def visit_STRING(self, node):
        # Generate unique name for each string
        self.string_counter += 1
        name = f".str{self.string_counter}"

        # Create global string constant
        str_val = node.value + '\0'
        str_const = ir.Constant(ir.ArrayType(ir.IntType(8), len(str_val)),
                               bytearray(str_val.encode()))
        global_str = ir.GlobalVariable(self.module, str_const.type, name=name)
        global_str.linkage = 'internal'
        global_str.global_constant = True
        global_str.initializer = str_const
        return self.builder.bitcast(global_str, ir.IntType(8).as_pointer())

    def visit_VAR(self, node):
        ptr = self.vars[node.value]
        return self.builder.load(ptr)

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

    generator = LLVMCodeGenerator()
    llvm_ir = generator.generate(ast)

    with open("output.ll", "w") as f:
        f.write(llvm_ir)

    compile_result = subprocess.run(["clang", "-Wno-override-module", "output.ll", "-o", "output"])
    if compile_result.returncode == 0:
        print("Output:")
        subprocess.run(["./output"])
    else:
        print("Compilation failed")
