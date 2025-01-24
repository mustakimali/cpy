
    .global main
    .section .text
    main:
        
        push %rbp
        mov %rsp, %rbp
        sub $32, %rsp
        
        mov $0, %rax
mov %rax, -8(%rbp)
mov $1, %rax
mov %rax, -16(%rbp)
mov $0, %rax
mov %rax, -24(%rbp)
L1:
mov -24(%rbp), %rax
push %rax
mov $10, %rax
pop %rcx
cmp %rax, %rcx
jl L3
mov $0, %rax
jmp L4
L3:
mov $1, %rax
L4:
cmp $0, %rax
je L2
mov -8(%rbp), %rax
mov %rax, %rsi
lea .fmt_num(%rip), %rdi
xor %eax, %eax
call printf@PLT
mov -8(%rbp), %rax
push %rax
mov -16(%rbp), %rax
pop %rcx
add %rcx, %rax
mov %rax, -32(%rbp)
mov -16(%rbp), %rax
mov %rax, -8(%rbp)
mov -32(%rbp), %rax
mov %rax, -16(%rbp)
mov -24(%rbp), %rax
push %rax
mov $1, %rax
pop %rcx
add %rcx, %rax
mov %rax, -24(%rbp)
jmp L1
L2:
        
        mov %rbp, %rsp
        pop %rbp
        ret
        
    .section .rodata
    .fmt_num:
        .string "%d\n"
    .fmt_str:
        .string "%s\n"
    