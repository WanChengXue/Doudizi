from rosea import control_types
from rosea.signal_controller.controller import BaseController


class MultiPointIndependentController(BaseController):
    def step(self, action):
        if action is None:
            return
        action = getattr(self._action_converter, f"convert_{control_types[self._cfg.get('control_type', 'delta_categorical')]['controller']}")(action)
        self.assign(action)

    def reset(self, logic_type, **kwargs):
        for tls_id, unit in self:
            if logic_type == 'default' and kwargs.get('default_states', None):
                unit.reset(**unit.get_logic('user_defined', **kwargs['default_states'][tls_id]))
            else:
                unit.reset(**unit.get_logic(logic_type))
