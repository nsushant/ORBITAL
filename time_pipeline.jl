# time_pipeline.jl
#
# End-to-end timing of a single low-thrust transfer through the full pipeline:
#   4.2 coasting-orbit estimate (golden-section)  →  4.3 continuous-thrust NLP (NLopt/SLSQP)
#
# Self-contained (Base Julia + NLopt). Run:
#   julia --project=. -e 'import Pkg; Pkg.add("NLopt")'   # one-time
#   julia --project=. time_pipeline.jl

using NLopt, Printf

const J2_P=1.0825267e-3; const RE_P=6378.137; const MU_P=398600.4418
const K_J2=3*J2_P*RE_P^2/(2*MU_P^3); const DAY=86400.0
raan_dot(a,I)=-1.5*J2_P*RE_P^2*sqrt(MU_P/a^7)*cos(I)
Vc_(a)=sqrt(MU_P/a)

# ── propagator (Section 3, validated) ────────────────────────────────────────
function propagate_phase(a0,I0,Ω0,β1,β2,ϑ,ε1,ε2,T,fmax)
    f1=ε1*fmax; f2=ε2*fmax; fc1=f1*cos(β1); fs1=f1*sin(β1); fc2=f2*cos(β2); fs2=f2*sin(β2)
    V0=sqrt(MU_P/a0); κ=(4ϑ*fc1+2*(π-2ϑ)*fc2)/π; tiny=1e-20
    a_var=abs(κ)>tiny; I_var=abs(fs1)>tiny
    a = a_var ? 4*MU_P/(2V0-κ*T)^2 : a0
    if !I_var; I=I0
    elseif !a_var; ξ=(2fs1*sin(ϑ))/(π*V0); I=I0+ξ*T
    else; z=log(2V0-κ*T); z0=log(2V0); ξ̄=-4fs1*sin(ϑ)/(π*κ); I=I0+ξ̄*(z-z0); end
    if !a_var && !I_var
        Ω=Ω0+((2cos(ϑ)*fs2)/(π*V0*sin(I0))-K_J2*V0^7*cos(I0))*T
    elseif !a_var && I_var
        ξ=(2fs1*sin(ϑ))/(π*V0); IT=I0+ξ*T
        L=log(((1+cos(IT))*(1-cos(I0)))/((1-cos(IT))*(1+cos(I0))))
        Ω=Ω0-K_J2*(V0^7/ξ)*(sin(IT)-sin(I0))-(cos(ϑ)*fs2/(π*ξ*V0))*L
    elseif a_var && !I_var
        z=log(2V0-κ*T); z0=log(2V0)
        Ω=Ω0-(4cos(ϑ)*fs2/(π*κ*sin(I0)))*(z-z0)+(K_J2*cos(I0)/(1024κ))*(exp(8z)-exp(8z0))
    else
        z=log(2V0-κ*T); z0=log(2V0); ξ̄=-4fs1*sin(ϑ)/(π*κ); IT=I0+ξ̄*(z-z0)
        L=log(((1+cos(IT))*(1-cos(I0)))/((1-cos(IT))*(1+cos(I0))))
        g(zz,II)=exp(8zz)*(8cos(II)+ξ̄*sin(II))/(64+ξ̄^2)
        Ω=Ω0+(2cos(ϑ)*fs2/(π*ξ̄*κ))*L+(K_J2/(128κ))*(g(z,IT)-g(z0,I0))
    end
    return a,I,Ω
end

# ── 4.2: coasting-orbit estimate (golden-section, Case 2 general) ─────────────
function golden_min(f,a,b;tol=1e-10,mx=500)
    ip=(sqrt(5.0)-1)/2; ip2=(3-sqrt(5.0))/2; a,b=min(a,b),max(a,b); h=b-a
    h<=tol && return (a+b)/2
    c=a+ip2*h; d=a+ip*h; fc=f(c); fd=f(d)
    for _ in 1:mx
        if fc<fd; b,d,fd=d,c,fc; h*=ip; c=a+ip2*h; fc=f(c)
        else; a,c,fc=c,d,fd; h*=ip; d=a+ip*h; fd=f(d) end
        h<=tol && break
    end
    return fc<fd ? (a+d)/2 : (c+b)/2
end
function coasting_orbit(a0,I0,af,If,dRAAN,Tf)
    ab=(a0+af)/2; Vb=(Vc_(a0)+Vc_(af))/2; Dat=af-a0; DIt=If-I0
    Ω̇0=raan_dot(a0,I0); Ω̇f=raan_dot(af,If); R=(dRAAN+(Ω̇f-Ω̇0)*Tf)/(Ω̇0*Tf)
    x1(x2)=-(R+tan(I0)*x2)/3.5
    J(x2)=(Vb*sqrt((x1(x2)*ab/(2ab))^2+x2^2)+Vb*sqrt(((Dat-x1(x2)*ab)/(2ab))^2+(DIt-x2)^2))
    x2=golden_min(J,min(0.0,DIt)-deg2rad(90),max(0.0,DIt)+deg2rad(90))
    return a0+x1(x2)*ab, I0+x2
end

# ── 4.3: continuous-thrust NLP (ξ1), seeded by phase inversion at ϑ=10° ───────
function invert(as,Is,at,It,ϑ,fmax)
    V0=Vc_(as); Vt=Vc_(at); dI=It-Is
    β1=atan(-dI*ϑ/(sin(ϑ)*log(Vt/V0)))
    (cos(β1)*(V0-Vt)<0) && (β1+=π); β1=mod(β1+π,2π)-π
    T=π*(V0-Vt)/(2*ϑ*fmax*cos(β1)); return β1,T
end
function fdgrad!(g,fn,x); f0=fn(x); @inbounds for i in eachindex(x); h=1e-7*max(1.0,abs(x[i])); xp=copy(x); xp[i]+=h; g[i]=(fn(xp)-f0)/h; end; f0; end
function continuous_dv(a0,I0,af,If,dRAAN,ac,Ic,Tf,fmax)
    Om0=0.0; OmT=dRAAN+raan_dot(af,If)*Tf
    traj(x)=begin
        a1,I1,O1=propagate_phase(a0,I0,Om0,x[1],0.0,x[2],1.0,0.0,x[3]*DAY,fmax)
        O2=O1+raan_dot(a1,I1)*(x[6]*DAY-x[3]*DAY)
        a3,I3,O3=propagate_phase(a1,I1,O2,x[4],0.0,x[5],1.0,0.0,Tf-x[6]*DAY,fmax); (a1,I1,a3,I3,O3)
    end
    rawobj(x)=(2fmax/π)*(x[2]*x[3]*DAY+x[5]*(Tf-x[6]*DAY))*1000
    rawcon(x)=(t=traj(x); [(t[1]-ac)/100,t[2]-Ic,(t[3]-af)/100,t[4]-If,t[5]-OmT])
    βt1,Tt=invert(a0,I0,ac,Ic,deg2rad(10),fmax); βa1,Tad=invert(ac,Ic,af,If,deg2rad(10),fmax)
    x0=[βt1,deg2rad(10),Tt/DAY,βa1,deg2rad(10),(Tf-Tad)/DAY]
    opt=Opt(:LD_SLSQP,6)
    opt.lower_bounds=[deg2rad(-179),deg2rad(0.3),0.05,deg2rad(-179),deg2rad(0.3),40.0]
    opt.upper_bounds=[deg2rad(179),deg2rad(89),60.0,deg2rad(179),deg2rad(89),99.95]
    opt.xtol_rel=1e-9; opt.maxeval=3000
    opt.min_objective=(x,g)->(length(g)>0 ? fdgrad!(g,rawobj,x) : rawobj(x))
    for k in 1:5; let k=k; ck=xx->rawcon(xx)[k]; equality_constraint!(opt,(x,g)->(length(g)>0 ? fdgrad!(g,ck,x) : ck(x)),1e-6); end; end
    (minf,minx,ret)=optimize(opt,x0); return minf
end

# ── full pipeline ────────────────────────────────────────────────────────────
function full_pipeline(a0,I0,RAAN0,af,If,RAANf,Tf_days;fmax=3.5e-6)
    Tf=Tf_days*DAY; dRAAN=RAANf-RAAN0
    ac,Ic=coasting_orbit(a0,I0,af,If,dRAAN,Tf)   # 4.2
    dv=continuous_dv(a0,I0,af,If,dRAAN,ac,Ic,Tf,fmax)  # 4.3
    return dv
end

# ── timing (Case 2 geometry: 800->900 km, 98->99 deg, RAAN 0->30) ─────────────
d=deg2rad
args=(RE_P+800,d(98.0),d(0.0),RE_P+900,d(99.0),d(30.0),100.0)
dv=full_pipeline(args...)                       # warm-up (JIT)
@printf("one transfer DeltaV = %.2f m/s\n",dv)
N=50; t=@elapsed for _ in 1:N; full_pipeline(args...); end
@printf("full pipeline (4.2 + 4.3): %.3f ms / transfer  (avg of %d)\n",t/N*1e3,N)
t2=@elapsed for _ in 1:N; coasting_orbit(args[1],args[2],args[4],args[5],args[6]-args[3],args[7]*DAY); end
@printf("  4.2 coasting-orbit only : %.4f ms / transfer\n",t2/N*1e3)
