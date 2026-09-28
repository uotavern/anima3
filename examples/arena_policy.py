"""Run: --decision-factory examples.arena_policy:create --model-label my-policy --version v1.
Replace the decision here with your model/RL policy. Keep the supplied option IDs.
"""
from anima3.decision import Decision


class MyPolicy:
    name = 'example'

    def choose(self, scene, question, options):
        # scene contains ordinary client observations. options are currently legal actions.
        # Do not execute arbitrary model-generated code or read opponent private state.
        choice = next(iter(options))
        return Decision(choice, {key: float(key == choice) for key in options}, 1.0, 0.0, self.name)


def create():
    return MyPolicy()
