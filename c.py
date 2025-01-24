import os
import sys
import subprocess
from llvmlite import ir, binding

TOKENS = {
    'IF': 'if', 'ELSE': 'else', 'WHILE': 'while', 'PRINT': 'print',
    'LET': 'let', 'FUNC': 'func', 'RETURN': 'return', 'INCLUDE': 'include',
    'LBRACE': '{', 'RBRACE': '}', 'LPAREN': '(', 'RPAREN': ')',
    'SEMI': ';', 'COMMA': ',', 'ASSIGN': '=', 'READ': 'read'
}

def lex(input_str):
    tokens = []
    i = 0
    while i < len(input_str):
        c = input_str[i]
        if c.isspace():
            i += 1
        elif c == '#':
            while i < len(input_str) and input_str[i] != '\n':
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
    def __init__(self, tokens, parent=None, file_dir=""):
        self.tokens = tokens
        self.pos = 0
        self.vars = {}
        self.functions = parent.functions.copy() if parent else {}
        self.parent = parent
        self.included_files = parent.included_files.copy() if parent else set()
        self.file_dir = file_dir

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
        nodes = []
        while not self.eof():
            nodes.append(self.parse_statement())
        return Node('BLOCK', children=nodes)

    def parse_block(self):
        nodes = []
        while self.peek() not in ['RBRACE', None] and not self.eof():
            nodes.append(self.parse_statement())
        return Node('BLOCK', children=nodes)

    def parse_statement(self):
        token_type = self.peek()
        if token_type == 'INCLUDE':
            return self.parse_include()
        elif token_type == 'LET':
            return self.parse_var_decl()
        elif token_type == 'FUNC':
            return self.parse_function()
        elif token_type == 'RETURN':
            return self.parse_return()
        elif token_type == 'WHILE':
            return self.parse_while()
        elif token_type == 'IF':
            return self.parse_if()
        elif token_type == 'PRINT':
            return self.parse_fn_with_one_arg('PRINT')
        elif token_type == 'IDENT' and self.pos+1 < len(self.tokens) and self.tokens[self.pos+1][0] == 'ASSIGN':
            return self.parse_var_assign()
        else:
            expr = self.parse_expression()
            self.consume('SEMI')
            return Node('EXPR_STMT', children=[expr])

    def parse_include(self):
        self.consume('INCLUDE')
        filename_token = self.consume('STRING')
        filename = filename_token[1]

        full_path = os.path.join(self.file_dir, filename)
        full_path = os.path.normpath(full_path)
        abs_path = os.path.abspath(full_path)

        if abs_path in self.included_files:
            return Node('EMPTY')
        self.included_files.add(abs_path)

        if not os.path.exists(full_path):
            raise ValueError(f"Included file not found: {full_path}")

        with open(full_path, 'r') as f:
            content = f.read()

        included_tokens = lex(content)
        included_dir = os.path.dirname(full_path)
        included_parser = Parser(included_tokens, parent=self, file_dir=included_dir)
        included_ast = included_parser.parse()

        self.functions.update(included_parser.functions)
        return Node('INCLUDE', children=included_ast.children)

    def parse_function(self):
        self.consume('FUNC')
        name = self.consume('IDENT')[1]

        self.functions[name] = {'params': [], 'body': None}

        self.consume('LPAREN')
        params = []
        while self.peek() != 'RPAREN':
            param = self.consume('IDENT')[1]
            params.append(param)
            self.vars[param] = 'int'
            if self.peek() == 'COMMA':
                self.consume('COMMA')
        self.consume('RPAREN')

        self.consume('LBRACE')
        body = self.parse_block()
        self.consume('RBRACE')

        self.functions[name] = {
            'params': params,
            'body': body,
            'return_type': 'int'
        }

        for param in params:
            del self.vars[param]

        return Node('FUNCTION', name, [params, body])

    def parse_return(self):
        self.consume('RETURN')
        expr = self.parse_expression()
        self.consume('SEMI')
        return Node('RETURN', children=[expr])

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

    def parse_fn_with_one_arg(self, ty):
        self.consume(ty)
        self.consume('LPAREN')
        arg = self.parse_expression()
        self.consume('RPAREN')
        self.consume('SEMI')
        return Node(ty, children=[arg])

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

    def parse_function_call(self):
        name = self.consume('IDENT')[1]
        if name != 'read' and name not in self.functions:
            raise ValueError(f"Undefined function: {name}")

        self.consume('LPAREN')
        args = []
        while self.peek() != 'RPAREN':
            args.append(self.parse_expression())
            if self.peek() == 'COMMA':
                self.consume('COMMA')
        self.consume('RPAREN')
        return Node('CALL', name, args)

    def parse_primary(self):
        token = self.tokens[self.pos]
        if token[0] == 'READ':
            # Handle read() function call
            self.consume('READ')
            self.consume('LPAREN')
            self.consume('RPAREN')
            return Node('CALL', 'read', [])
        elif token[0] == 'IDENT':
            if self.pos+1 < len(self.tokens) and self.tokens[self.pos+1][0] == 'LPAREN':
                return self.parse_function_call()
            name = token[1]
            if name not in self.vars:
                raise ValueError(f"Undefined variable: {name}")
            self.consume()
            return Node('VAR', name)
        elif token[0] == 'NUMBER':
            self.consume()
            return Node('NUMBER', token[1])
        elif token[0] == 'STRING':
            self.consume()
            return Node('STRING', token[1])
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
        self.functions = {}
        self.printf = None
        self.scanf = None
        self.string_counter = 0
        self.fmt_counter = 0

        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        self.fmt_num = self.create_global_fmt("%d")
        self.fmt_str = self.create_global_fmt("%s")

    def create_global_fmt(self, fmt):
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

    def declare_scanf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.scanf = ir.Function(self.module, scanf_ty, name="scanf")

    def generate(self, ast):
        self.declare_printf()
        self.declare_scanf()

        for child in ast.children:
            if child.type in ['FUNCTION', 'INCLUDE']:
                self.visit(child)

        func_type = ir.FunctionType(ir.IntType(32),
                                   [ir.IntType(32),
                                    ir.IntType(8).as_pointer().as_pointer()])
        main_func = ir.Function(self.module, func_type, name="main")
        self.func = main_func
        block = main_func.append_basic_block("entry")
        builder = ir.IRBuilder(block)

        old_builder = self.builder
        self.builder = builder
        for child in ast.children:
            if child.type not in ['FUNCTION', 'INCLUDE']:
                self.visit(child)
        self.builder = old_builder

        builder.ret(ir.Constant(ir.IntType(32), 0))
        self.func = None
        return str(self.module)

    def visit(self, node):
        method_name = f'visit_{node.type}'
        return getattr(self, method_name)(node)

    def visit_INCLUDE(self, node):
        for child in node.children:
            self.visit(child)

    def visit_FUNCTION(self, node):
        func_name = node.value
        params, body = node.children
        param_types = [ir.IntType(32) for _ in params]

        func_type = ir.FunctionType(ir.IntType(32), param_types)
        function = ir.Function(self.module, func_type, name=func_name)
        self.functions[func_name] = function

        entry_block = function.append_basic_block("entry")
        builder = ir.IRBuilder(entry_block)

        old_vars = self.vars.copy()
        for i, param in enumerate(params):
            ptr = builder.alloca(ir.IntType(32), name=param)
            builder.store(function.args[i], ptr)
            self.vars[param] = ptr

        old_builder = self.builder
        old_func = self.func
        self.builder = builder
        self.func = function
        self.visit(body)
        self.builder = old_builder
        self.func = old_func

        if not builder.block.is_terminated:
            builder.ret(ir.Constant(ir.IntType(32), 0))

        self.vars = old_vars

    def visit_RETURN(self, node):
        value = self.visit(node.children[0])
        self.builder.ret(value)

    def visit_CALL(self, node):
        if node.value == 'read':
            return self.visit_READ(node)
        func = self.functions[node.value]
        args = [self.visit(arg) for arg in node.children]
        return self.builder.call(func, args)

    def visit_READ(self, node):
        input_ptr = self.builder.alloca(ir.IntType(32))
        fmt = self.create_global_fmt("%d")
        fmt_ptr = self.builder.bitcast(fmt, ir.IntType(8).as_pointer())
        self.builder.call(self.scanf, [fmt_ptr, input_ptr])
        return self.builder.load(input_ptr)

    def visit_BLOCK(self, node):
        for child in node.children:
            self.visit(child)

    def visit_VAR_DECL(self, node):
        var_name = node.value
        expr = self.visit(node.children[0])
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

        if not self.builder.block.is_terminated:
            self.builder.branch(cond_block)

        self.builder.position_at_end(cond_block)
        cond_value = self.visit(node.children[0])
        self.builder.cbranch(cond_value, loop_block, after_block)

        self.builder.position_at_end(loop_block)
        self.visit(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_block)

        self.builder.position_at_end(after_block)

    def visit_IF(self, node):
        condition, then_block, else_block = node.children

        then_bb = self.func.append_basic_block("then")
        else_bb = self.func.append_basic_block("else") if else_block else None
        merge_bb = self.func.append_basic_block("merge")

        cond_value = self.visit(condition)

        if else_block:
            self.builder.cbranch(cond_value, then_bb, else_bb)
        else:
            self.builder.cbranch(cond_value, then_bb, merge_bb)

        self.builder.position_at_end(then_bb)
        self.visit(then_block)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        if else_block:
            self.builder.position_at_end(else_bb)
            self.visit(else_block)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        if not self.builder.block.is_terminated:
            self.builder.position_at_end(merge_bb)

    def visit_PRINT(self, node):
        value = self.visit(node.children[0])
        if isinstance(value.type, ir.IntType):
            fmt_ptr = self.builder.bitcast(self.fmt_num, ir.IntType(8).as_pointer())
        else:
            fmt_ptr = self.builder.bitcast(self.fmt_str, ir.IntType(8).as_pointer())
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
        self.string_counter += 1
        name = f".str{self.string_counter}"
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

    main_file = sys.argv[1]
    if not os.path.exists(main_file):
        print(f"Error: File '{main_file}' not found")
        sys.exit(1)

    main_dir = os.path.dirname(os.path.abspath(main_file))

    try:
        with open(main_file, 'r') as f:
            input_str = f.read()
    except FileNotFoundError:
        print(f"Error: File '{main_file}' not found")
        sys.exit(1)

    try:
        tokens = lex(input_str)
    except ValueError as e:
        print(f"Lexer error: {e}")
        sys.exit(1)

    try:
        parser = Parser(tokens, file_dir=main_dir)
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
