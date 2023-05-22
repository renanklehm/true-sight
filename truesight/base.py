from abc import ABC, abstractmethod

class StatisticalForecaster(ABC):
    @abstractmethod
    def fit(self, y):
        pass
    
    @abstractmethod
    def predict(self, y):
        pass