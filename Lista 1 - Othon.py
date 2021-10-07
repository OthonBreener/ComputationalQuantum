# ## Método da bisseção
#
# É o método mais simples para encontrar raiz, depende de zerar uma raiz e ir repetidamente bisseccionando um intervalo e usando o teorema do valor médio.
# * Teorema do valor médio:$\newline$
# Uma função contínua $f: D \to \Re $ com D = [a,b], possui uma raiz $m \in \Re$ tal que $f(a) < m < f(b)$ quando existe um valor $c \in [a,b]$ tal que $f(c) = m$. $\newline$
# Se $f(a)$ e $f(b)$ possuem sinais opostos, existe uma raiz $c \in [a,b]$ tal que $f(c)$ = 0.
#
# * Aplicação númerica: $\newline$
# Escolha um intervalo apropriado $x \in [a_0 , b_0]$, tal que $f(a_0) \cdot f(b_0) < 0$ (eles devem ter sinais opostos), logo para n = 0,1,2,3.. $\newline$
# 1) Calcular o ponto médio no intervalo: $c_n = \frac{1}{2}(a_n + b_n)$ $\newline$
# 2) Verifique se convergiu para uma raiz com precisão aceitável, $f(c_n) = 0$. $\newline$
# 3) Se não convergiu, bisseccione o intervalo: $\newline$
# i. Se $f(a_n) \cdot f(c_n) > 0$ (As duas funções tem o mesmo sinal), o novo intervalo é: $[a_{n + 1}, b_{n + 1}] = [c_n , b_n]$. $\newline$
# ii. Se $f(a_n) \cdot f(c_n) < 0$ (As duas funções tem sinal diferente), o novo intervalo é: $[a_{n + 1}, b_{n + 1}] = [a_n , c_n]$. $\newline$
# 4) Retorne ao passo 1. $\newline$
#
# O número necessário de intereções (n) necessário para a convergência é dado por: $\newline$
#
# $n \geq \frac{log(b_0 - a_0) - log(\varepsilon)}{log(2)}$, onde $\varepsilon$ é a precisão.  $\newline$
# A convergência deve satisfazer $b_n - a_n \leq \varepsilon$ para um $\varepsilon << 1$.

# ## Questão 1
# Reproduza os resultados vistos em aula. Use os métodos de Newton e da bisseção para encontrar
# todas as raízes da equação: $\sin(x) = \alpha \cdot x $ para $\alpha = 0.1 $. São 7 raízes, encontre todas!

#bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from scipy import optimize

function = lambda x: np.sin(x) - 0.1*x

#Estimativa da raiz

x = np.linspace(-10, 10, 1000)
y1 = np.sin(x)
y2 = 0.1*x

plt.figure(figsize = (10, 4))

plt.subplot(121)
plt.plot(x, y1)
plt.plot(x, y2)
plt.ylim(-1.5, +1.5)
plt.grid()

plt.subplot(122)
plt.plot(x, function(x))
plt.ylim(-1.5, +1.5)
plt.xlim(-4, 4)
plt.grid()

plt.tight_layout()
plt.show()

#Função para o cálculo da raiz

def bisection(a0, b0, function, e = 1*10**(-6),m = 1000):
    '''
    Encontra a raiz de uma função f(x) usando o método de bisesseção.
    INPUT:
        a0,b0 = intervalo a0 < x < b0 para encontrar a raiz.
        f = função a ser utilizada
        e = precisão
        m = máximo de passos para encontrar o número de interações

    OUTPUT:
        Uma tupla coontendo a raiz e o número de intreações.

    '''
    n_interation = 0
    while (b0 - a0 > e) and (n_interation < m):
        n_interation += 1
        c0 = (a0 + b0)/2
        if function(a0)*function(c0) < 0:
            b0 = c0
        else:
            a0 = c0

    return (c0, n_interation)

#Resultado

root = bisection(0.1, 5, function)
print(root)

root = bisection(-2, 5, function)
print(root)

root = bisection(-10, -8, function)
print(root)

root = bisection(-8, -7, function)
print(root)

root = bisection(-4, 0, function)
print(root)

root = bisection(5, 8, function)
print(root)

root = bisection(8, 10, function)
print(root)


# # Teoria
#
# * A equação de Schrodinger independente do tempo possui estados estacionários, uma propriedade de ondas estacionárias é que o número de nodos (pontos em que a função de onda cruza com o eixo x) está relacionado com a energia. Por exemplo: para o estado fundamental o número de nodos é igual a 0, para o primeiro estado excitado o número de nodos é 1, para o segundo estado excitado há 2 números de nodos e assim por diante. Logo, com o inutito de encontrar as autoenergias da equação de Schrodinger independente do tempo em diferentes potenciais, utilizaremos o método de Shooting, que consiste em transforma a equação que originalmente é um problema de condições de contorno em um problema de valor inicial. Essa mudança será feita utilizando o método de diferenças finitas, onde é feita uma propagação da função para cada valor de E, e as energias E serão encontradas através do método de Newton.
#
# ## Propagação da função
# * Com o intuito de fazer com que a nossa equação de Schrodinger independente do tempo possa ser resolvida por condições iniciais, uma vez que as condições de contorno são: $\Psi(0) = 0 $ e $\Psi(L) = 0$, usaremos o método de diferenças finitas para a derivada de segunda ordem, logo temos: $\newline$
#
# * Equação de Schrodinger: - $\frac{\hbar ^2}{2 m} \frac{\partial ^2}{\partial x^2} \Psi(x) + V(x) \Psi(x) = E \Psi(x)$ $\newline$
#
# * A segunda derivada discreta é dada por: f''(x) = $\frac{f(x + h) + f(x - h) - 2 f(x)}{h^2}$  $\newline$
#
# * Aplicando na equação de Schrodinger: $\newline$
#
# * $ \frac{\partial ^2}{\partial x^2} \Psi(x) = \frac{\Psi(x - a) - 2* \Psi(x) + \Psi(x + a)}{a^2} $ Ou, $\newline$
#
# * $ \frac{\partial ^2}{\partial x^2} \Psi(x) = \frac{\Psi_{i-1} - 2 \Psi_i + \Psi_{i + 1}}{a^2}$ Logo: $\newline$
#
# * $ - \frac{\hbar ^2}{2 m}$ $\frac{\Psi_{i-1} - 2 \Psi_i + \Psi_{i + 1}}{a^2}$ + V(x) $\Psi_i $ = E $\Psi_i $, Manipulando de forma a isolar $\Psi_{i + 1}$ e utilizando unidades atômicas m = $\hbar$ = 1: $\newline$
#
# * $ \Psi_{i+1} = 2 [a^2 (V(x) - E) + 1] \Psi_i - \Psi_{i-1} $
#
# ## Método de Newton-Raphson
#
# * O método de Newton-Raphson melhora o método de bisseção usando o gradiente da curva, apartir de um valor inicial é calculado o valor da reta tangente da função nesse ponto e a interseção dela com o eixo das abcissas(eixo x), repetindo esse processo cria-se um método interativo afim de encontrar a raiz da função. $\newline$
# * Considere uma função y = f(x), a reta tangente ao ponto $f(x_n)$ é: $\newline$
# $y = f(x_n) + f'(x_n)(x - x_n)$ $\newline$
# Utilizando a raiz da função (y = 0) e $x_{n + 1}$ é dado por: $\newline$
# $x_{n + 1} = x_n - \frac{f(x_n)}{f'(x_n)}$ $\newline$
# O critério de parada é feito definindo uma tolerância de forma que esta seja menor que o intervalo utilizado na derivada, enquanto o módulo da função for maior que a tolerância, aplica-se o método de newton até chegar a um valor adequado.
#
# ## Normalização
# * A normalização será feita obdecendo: $\int_{- \infty}^{+ \infty} \mid \Psi \mid^2 \,dx$ = A $\newline$
# * Pelo método do trapézio a integral é dada por: $\int_{x}^{x + \delta x} f(s) \,ds = \frac{1}{2} \delta x [f(x) + f(x + \delta x)]$ $\newline$
# * Para o nosso caso: A = $\frac{1}{2} \delta x [f(x)^2 + f(x + 1)^2]$ $\newline$
# * Logo para normalizar a função de onda, devemos multiplicar o resultado por 1 / $\sqrt A$ $\newline$
#
# ## Aplicação do Código
#
# * Chute um valor inicial de E.
# * Propague a função de onda para E em todo o intervalo 0 a L.
# * Verifique se $ \mid \Psi(L) \mid $ < e, e = $10^{-6}$.
# * Se não, volte ao primeiro passo.

# ## Questão 3
# * Calcule as energias da equações de Schrodinger independente do tempo para um poço parabolico centrado no meio do eixo x.

#constantes

L = 10
N = 100
a = L /(N - 1)
psi1 = a
psi0 = 0
eps = 10**(-6)
x = list(np.linspace(0, L, N))

#Potencial

def potencial(x):
    pot = []
    for i in x:
        pot.append(0.5*(i - L/2)**2)
    return(pot)

# Plot do Potencial

plt.plot(x, potencial(x))
plt.ylim(-1, +13)
plt.grid()

#propagação da função

def func(x, potencial, E):
    vetor = []
    psi1 = a
    psi0 = 0
    for i in range(len(x)):
        psi2 = 2*((a**2)* (potencial(x)[i] - E) + 1)*psi1 - psi0
        psi0 = psi1
        psi1 = psi2
        vetor.append(psi2)

    return(vetor)

#chute das raizes

x1 = np.linspace(0,5,1000)
y = func(x, potencial, x1)

plt.figure(figsize = (13,5))

plt.subplot(121)
plt.plot(x1, y[-1])

plt.subplot(122)
plt.plot(x1,y[-1])
plt.plot(x1, np.zeros(len(y[-1])), 'b--')
plt.grid()
plt.ylim(-10*10**3,10*10**3)
plt.xlim(0,5)
plt.show()

#derivada da função em função da energia

def derivada(x, potencial, E):
    h = 1*10**(-3)
    derivada = (func(x, potencial, E + h)[-1] - func(x , potencial, E - h)[-1])/(2*h)
    return derivada

#estimativa da raiz pelo método de newton

def NewtonRaphson(x, func, derivada, potencial, E, maxn = 30, tolerancia = 1*10**(-6)):
    '''
    Cálcula a raiz de uma função através do método de Newton-Raphson
    Input:
        x = Lista contendo interaveis do espaço x
        E = chute inicial
        func = função a ser utilizada
        derivada = derivada da função
        tolerância = fator de parada
        maxn = máximo de pontos
    Output:
        Raiz aproximada da função
    '''
    inteiro = 0
    while abs(func(x, potencial, E)[-1]) > tolerancia and inteiro < maxn:
        inteiro = inteiro + 1
        E = E - (func(x, potencial, E)[-1])/derivada(x, potencial, E)
    return(E)

#raizes

roots = []
for i in range(10):
    E0 = (i + 0.5)
    root = NewtonRaphson(x, func, derivada, potencial, E0)
    roots.append(root)
    print(str(100*(i+1)/10)+"%", end='\r')
roots = np.round(np.array(roots),3)
print(roots)

#plot das energias
plt.figure(figsize = (10,8))

plt.subplot(331)
plt.plot(x, func(x, potencial, roots[0]))

plt.subplot(332)
plt.plot(x, func(x, potencial, roots[1]))

plt.subplot(333)
plt.plot(x, func(x, potencial, roots[2]))

plt.subplot(334)
plt.plot(x, func(x, potencial, roots[3]))

plt.subplot(335)
plt.plot(x, func(x, potencial, roots[4]))

plt.subplot(336)
plt.plot(x, func(x, potencial, roots[5]))

plt.subplot(337)
plt.plot(x, func(x, potencial, roots[6]))

plt.subplot(338)
plt.plot(x, func(x, potencial, roots[7]))

plt.subplot(339)
plt.plot(x, func(x, potencial, roots[8]))

plt.show()

# Função para normalizar a função de onda

def normalization(x, func):
    '''
    Função para normalizar uma função de onda através do método dos trapézios

    INPUT:
        func = lista contendo os valores da função de onda para cada energia
        x = lista contendo os intevareis do espaço x
        N = Número de discretizações

    OUTPUT:
        Inverso da constante de normalização, respeitando as condições para a função de onda.


    '''
    A = 0
    deltax = (x[-1] - x[0])/len(x)
    for i in range(len(x)-1):
        A = A + deltax*(1/2)*((func[i])**2 + (func[i + 1])**2)
    return 1/(np.sqrt(A))

#Plot das energias normalizadas

ey0 = normalization(x, func(x, potencial, roots[0]))*np.array(func(x, potencial, roots[0]))
ey1 = normalization(x, func(x, potencial, roots[1]))*np.array(func(x, potencial, roots[1]))
ey2 = normalization(x, func(x, potencial, roots[2]))*np.array(func(x, potencial, roots[2]))
ey3 = normalization(x, func(x, potencial, roots[3]))*np.array(func(x, potencial, roots[3]))
plt.plot(x, ey0)
plt.plot(x, ey1)
plt.plot(x, ey2)
plt.plot(x, ey3)
plt.ylim(-1,1)
plt.xlim(0,10)
plt.grid()
plt.show()


# # Questão 4

# ## a) Faça o mesmo que a questão 3 para um Poço Quadrado com:
# *  Largura do poço = 4
# *  Largura total do poço + barreiras 10
# *  Altura (em energia) do poço = 10
# *  Discretizar espaço em 100 pontos
#

# Potencial do Poço finito:
#
# V(x) =   $\ \begin{array}{lll}\Psi_I:V_0, se x \geq a/2 \\ \Psi_{II}: 0, se -a/2 < x < a/2 \\ \Psi_{III}: V_0, se x \neq -a/2 \end{array}$

#Potencial
def pquadrado(x):
    v = []
    for i in range(len(x)):
        if 3 < x[i] < 7:
            v.append(0)
        else:
            v.append(10)
    return(v)

# Plot do Potencial

plt.plot(x, pquadrado(x))
plt.ylim(-1, +13)
plt.grid()

#chute das raizes

x2 = np.linspace(0,5,1000)
y1 = func(x, pquadrado, x1)

plt.figure(figsize = (13,5))

plt.subplot(121)
plt.plot(x2, y1[-1])

plt.subplot(122)
plt.plot(x2,y1[-1])
plt.plot(x2, np.zeros(len(y1[-1])), 'b--')
plt.grid()
plt.ylim(-10*10**5,10*10**5)
plt.xlim(0,7)
plt.show()

#raizes

rootss = []
for i in range(10):
    E0 = (i + 0.25)
    root = NewtonRaphson(x, func, derivada, pquadrado, E0)
    rootss.append(root)
    print(str(100*(i+1)/10)+"%", end='\r')
rootss = np.round(np.array(rootss),3)
print(rootss)

#plot das energias
plt.figure(figsize = (10,8))

plt.subplot(331)
plt.plot(x, func(x, pquadrado, rootss[0]))

plt.subplot(332)
plt.plot(x, func(x, pquadrado, rootss[2]))

plt.subplot(333)
plt.plot(x, func(x, pquadrado, rootss[3]))

plt.subplot(334)
plt.plot(x, func(x, pquadrado, rootss[4]))

plt.subplot(335)
plt.plot(x, func(x, pquadrado, rootss[5]))

plt.subplot(336)
plt.plot(x, func(x, pquadrado, rootss[7]))

plt.show()

#Plot das energias normalizadas

eiy0 = normalization(x, func(x, pquadrado, rootss[0]))*np.array(func(x, pquadrado, rootss[0]))
eiy1 = normalization(x, func(x, pquadrado, rootss[2]))*np.array(func(x, pquadrado, rootss[2]))
eiy2 = normalization(x, func(x, pquadrado, rootss[3]))*np.array(func(x, pquadrado, rootss[3]))
eiy3 = normalization(x, func(x, pquadrado, rootss[4]))*np.array(func(x, pquadrado, rootss[4]))
eiy3 = normalization(x, func(x, pquadrado, rootss[5]))*np.array(func(x, pquadrado, rootss[5]))
eiy3 = normalization(x, func(x, pquadrado, rootss[7]))*np.array(func(x, pquadrado, rootss[7]))
plt.plot(x, eiy0)
plt.plot(x, eiy1)
plt.plot(x, eiy2)
plt.plot(x, eiy3)
plt.ylim(-1,1)
plt.xlim(0,10)
plt.grid()
plt.show()


# ## b) Faça o mesmo que a questão 3 para um Poço Quadrado Duplo com:
# *  Largura de cada poço = 4
# *  Separação entre os poços = 1
# *  Largura total do poço + barreiras 20
# *  Altura (em energia) do poço = 2
# *  Discretizar espaço em 500 pontos

L = 20
N = 800
a = L /(N - 1)
psi1 = a
psi0 = 0
eps = 10**(-6)
xw = np.linspace(0,L,N)

#Potencial
def pduplo(xw):
    vd = []
    for i in range(len(xw)):
        if 5.5 < xw[i] < 9.5:
            vd.append(0)
        elif 10.5 < xw[i] < 14.5:
            vd.append(0)

        else:
            vd.append(2)
    return(vd)

#plot do potencial
plt.plot(xw, pduplo(xw))
plt.grid()
plt.show()

#chute das raizes

x3 = np.linspace(0,5,1000)
y2 = func(xw, pduplo, x3)

plt.figure(figsize = (14,12))

plt.subplot(221)
plt.plot(x3, y2[-1])

plt.subplot(222)
plt.plot(x3,y2[-1])
plt.plot(x3, np.zeros(len(y2[-1])), 'b--')
plt.grid()
plt.ylim(-1.1,1.1)
plt.xlim(0,0.3)

plt.subplot(223)
plt.plot(x3, y2[-1])
plt.plot(x3, np.zeros(len(y2[-1])), 'b--')
plt.grid()
plt.xlim(0,7)
plt.ylim(-1.1,1.1)

plt.show()

E0 = [0.16, 0.21, 0.5, 0.8, 1.5, 1.7]
raiz = []
for i in E0:
    raiz.append(NewtonRaphson(xw, func, derivada, pduplo, i))
raiz = np.round(np.array(raiz),3)
print(raiz)

#plot das energias
plt.figure(figsize = (10,8))

plt.subplot(331)
plt.plot(xw, func(xw, pduplo, raiz[0]))

plt.subplot(332)
plt.plot(xw, func(xw, pduplo, raiz[1]))

plt.subplot(333)
plt.plot(xw, func(xw, pduplo, raiz[2]))

plt.subplot(334)
plt.plot(xw, func(xw, pduplo, raiz[3]))

plt.subplot(335)
plt.plot(xw, func(xw, pduplo, raiz[4]))

plt.subplot(336)
plt.plot(xw, func(xw, pduplo, raiz[5]))

plt.show()

#Plot das energias normalizadas

eity0 = normalization(xw, func(xw, pduplo, raiz[0]))*np.array(func(xw, pduplo, raiz[0]))
eity1 = normalization(xw, func(xw, pduplo, raiz[1]))*np.array(func(xw, pduplo, raiz[1]))
eity2 = normalization(xw, func(xw, pduplo, raiz[2]))*np.array(func(xw, pduplo, raiz[2]))
eity3 = normalization(xw, func(xw, pduplo, raiz[3]))*np.array(func(xw, pduplo, raiz[3]))
eity3 = normalization(xw, func(xw, pduplo, raiz[4]))*np.array(func(xw, pduplo, raiz[4]))
eity3 = normalization(xw, func(xw, pduplo, raiz[5]))*np.array(func(xw, pduplo, raiz[5]))
plt.plot(xw, eity0)
plt.plot(xw, eity1)
plt.plot(xw, eity2)
plt.plot(xw, eity3)
plt.ylim(-1,1)
plt.xlim(0,20)
plt.grid()
plt.show()
