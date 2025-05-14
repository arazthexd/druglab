from typing import Any

class BasePreparation:
    def prepare(self, obj) -> Any:
        raise NotImplementedError()
    