#!/usr/bin/env julia

#Start Test Script
using TinySparse
using Base.Test

function test_pack()
    println("Testing sparse matrix pack...")
    pack(sprand(100,100,0.1))
    return true
end

function test_spmv()
    println("Testing out of place multiplication.")
    A = sprand(100,100,0.1)
    spA = pack(A)
    x = rand(100)
    if A*x == spA*x
        return true
    else
        return false
    end
end

function test_spmv_ip()
    println("Testing in place multiplication.")
    A = sprand(100,100,0.1)
    spA = pack(A)
    x = rand(100)
    b = Vector{Float64}(100)
    if A_mul_B!(b, A, x) == A_mul_B!(b, A, x)
        return true
    else
        return false
    end
end

function test_spmv_t_ip()
    println("Testing in place transposed multiplication.")
    A = sprand(100,100,0.1)
    spA = pack(A)
    x = rand(100)
    b = Vector{Float64}(100)
    if Ac_mul_B!(b, A, x) == Ac_mul_B!(b, A, x)
        return true
    else
        return false
    end
end

function run_tests()
    @test test_pack()
    @test test_spmv()
    @test test_spmv_ip()
    @test test_spmv_t_ip()
end

run_tests()