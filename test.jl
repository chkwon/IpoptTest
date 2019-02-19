using JuMP, Ipopt, Test

c = [10, 8, 6, 4, 2]
K = [5, 5, 5, 5, 5]
b = [1.2, 1.1, 1.0, 0.9, 0.8]
L = 20
g = 1.7
gg = 5000^(1/g)

model = Model(with_optimizer(Ipopt.Optimizer))

@variable(model, 0 <= x <= L)
@variable(model, y[1:4])
@variable(model, l[1:4])
@variable(model, w[1:4] >= 0 )
@variable(model, v[1:4] >= 0 )
@variable(model, Q >= 0)
@constraint(model, Q == x+y[1]+y[2]+y[3]+y[4])
@NLobjective(model, Min, c[1]*x + b[1]/(b[1]+1)*K[1]^(-1/b[1])*x^((1+b[1])/b[1])
                 - x*( gg*Q^(-1/g) ) )

@NLconstraint(model, cnstr[i=1:4], 0 == ( c[i+1] + K[i+1]^(-1/b[i+1])*y[i] ) - ( gg*Q^(-1/g) )
                                  - y[i]*( -1/g*gg*Q^(-1-1/g) ) - l[i] )

#
mpec_tol = 1e-8

for i in 1:4
  @NLconstraints(model, begin
    0 <= y[i] <= L
    l[i] == w[i] - v[i]
    sqrt( (L-y[i])^2 + (v[i])^2 + mpec_tol) - (L-y[i]) - v[i] == 0
    sqrt( (y[i])^2 + (w[i])^2 + mpec_tol) - (y[i]) - w[i] == 0
  end)
end

JuMP.optimize!(model)

@show JuMP.value.(y)
@show JuMP.value.(l)
@show JuMP.value(x)
@show JuMP.value(Q)

@show JuMP.objective_value(model)
@test isapprox(JuMP.objective_value(model), -6.11671, atol=1e-4)
@test isapprox( JuMP.value.(l)' * (L .- JuMP.value.(y)), 0, atol=1e-5)
