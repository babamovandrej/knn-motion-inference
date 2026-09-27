from collections.abc import Mapping, Sequence

MAX_REPORTED_ERRORS = 5


class FeatureValidationError(ValueError):
    pass


class FeatureValidator:
    def __init__(
        self,
        feature_names: Sequence[str],
    ) -> None:
        self._expected = frozenset(feature_names)

    def _problems(
        self,
        sample: Mapping[str, float],
    ) -> list[str]:
        keys = set(sample)
        missing = sorted(self._expected - keys)
        extra = sorted(keys - self._expected)

        return [
            f"{len(names)} {kind} (e.g. {names[:3]})"
            for kind, names in (("missing", missing), ("unexpected", extra))
            if names
        ]

    def validate(
        self,
        samples: Sequence[Mapping[str, float]],
    ) -> None:
        errors = [
            f"sample {index}: {', '.join(problems)}"
            for index, sample in enumerate(samples)
            if (problems := self._problems(sample))
        ]

        if not errors:
            return

        message = "; ".join(errors[:MAX_REPORTED_ERRORS])
        remaining = len(errors) - MAX_REPORTED_ERRORS

        if remaining > 0:
            message += f"; and {remaining} more"

        raise FeatureValidationError(message)
