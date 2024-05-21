import numpy as np

def centering(X):
    # Centrar les dades
    return X - X.mean(axis=1, keepdims=True)

def whitening(X):
    # Blanqueig de les dades
    cov = np.dot(X, X.T) / X.shape[1]
    U, S, _ = np.linalg.svd(cov)
    D = np.diag(1.0 / np.sqrt(S))
    V = np.dot(D, U.T)
    X_white = np.dot(V, X)
    return X_white

def decorrelation(W):
    # Decorrelació de la matriu de barrejament (W)
    D, P = np.linalg.eigh(np.dot(W, W.T))
    return np.dot(np.dot(P, np.diag(1.0 / np.sqrt(D))), P.T)  # Fa un return de la W

def ica(X, max_iter=1000, tol=1e-5):
    """Implementació de l'algorisme ICA"""
    # Inicialització aleatòria de la matriu de barrejament
    m, n = X.shape
    W = np.random.rand(m, m)

    # Normalització de la matriu de barrejament
    W = decorrelation(W)

    # Iteracions de l'algorisme ICA
    for _ in range(max_iter):
        # Càlcul de les fonts estimades
        S = np.dot(W, X)

        # Funció de contrast: g(x) = tanh(x)
        g = np.tanh(S)

        # Derivada de la funció de contrast: g'(x) = 1 - tanh^2(x)
        g_prime = 1 - np.square(np.tanh(S))

        # Actualització de la matriu de barrejament utilitzant l'aproximació de la regla delta
        delta_W = np.dot(g, X.T) - np.dot(g_prime, np.ones((n, m))) * W
        W += delta_W

        # Normalització i decorrelació de la matriu de barrejament
        W = decorrelation(W)

        # Comprovació de la convergència
        if np.linalg.norm(delta_W) < tol:
            break

    # Recuperació de les fonts estimades
    S = np.dot(W, X)

    return S, W

# Exemple d'ús
# Suposem que 'X' és una matriu on cada fila representa una mostra d'àudio
# i cada columna representa una pista d'àudio en un fitxer multipista
# 'X' ha de tenir dimensions (m, n), on 'm' és el nombre de pistes i 'n' el nombre de mostres
# 'S' contindrà les fonts estimades, i 'W' serà la matriu de barrejament estimada
# 'S', 'W' = ica(X)

