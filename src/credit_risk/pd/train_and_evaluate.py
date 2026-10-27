"""Compatibilidad con el punto de entrada previo; la lógica vive en experiments."""

from credit_risk.pd.experiments import ejecutar_experimento_completo, main
from credit_risk.pd.neural_network import PDNet, PDNeuralNetwork

__all__ = ["PDNet", "PDNeuralNetwork", "ejecutar_experimento_completo", "main"]


if __name__ == "__main__":
    main()
