"""Bounded scripted ablation; never presented as on-policy neural experience."""
from .schema import ACTION_INDEX


class MeleeProbe:
    def __init__(self, mode, resources=False):
        if mode not in ('magic', 'punch', 'stun'):
            raise ValueError(mode)
        self.mode = mode
        self.resources = resources
        self.next_resource = 0.0
        self.stun_ready = False
        self.stun_pending = False
        self.next_stun = 0.0
        self.next_attack = 0.0
        self.opening = set()

    def feedback(self, journal, now):
        for j in journal:
            if j.cliloc == 1019011:
                self.stun_ready, self.stun_pending = True, False
            elif j.cliloc in (1019012, 1004013, 1004008, 1004031):
                self.stun_ready, self.stun_pending = False, False
            if j.cliloc in (1004013, 1004010):
                self.next_stun = now + 10.0

    def resource_packet(self, view, now, inventory, *, busy, hands_free):
        if (not self.resources or busy or not hands_free or not view.fresh
                or not view.ready or view.paralyzed or now < self.next_resource):
            return []
        graphic = 0xF08 if view.stamina_max < 30 else (0xF0B if view.stamina < 20 else None)
        item = next((i for i in inventory if i.graphic == graphic and i.amount > 0), None)
        if item is None:
            return []
        self.next_resource = now + 5.0
        return [{'type': 'Use', 'serial': item.serial}]

    def packets(self, view, now, opponent, *, busy, hands_free):
        if (busy or not view.fresh or not view.ready or view.paralyzed
                or not view.opponent_visible or not view.line_of_sight
                or view.distance > 1 or not hands_free or self.mode == 'magic'):
            return []
        packets = []
        if (self.mode == 'stun' and not self.stun_ready and not self.stun_pending
                and now >= self.next_stun and view.stamina >= 15):
            packets.append({'type': 'StunRequest'})
            # Toggle only once until the server acknowledges it; lost replies
            # must not cause repeated toggles that cancel the prepared stun.
            self.stun_pending = True
        if now >= self.next_attack:
            packets.append({'type': 'Attack', 'serial': opponent})
            self.next_attack = now + 2.0
        return packets

    def choose(self, view, mask):
        def legal(name):
            return mask[ACTION_INDEX[name]]
        for name, needed in (
            ('cure_potion', view.poisoned), ('cure', view.poisoned),
            ('heal_potion', view.hp < 65), ('greater_heal', view.hp < 55),
            ('heal', view.hp < 35),
        ):
            if needed and legal(name):
                return name
        for name in ('weaken', 'clumsy'):
            if name not in self.opening and legal(name):
                self.opening.add(name)
                return name
        reach = 4 if self.mode == 'magic' else 1
        if view.distance > reach and legal('approach'):
            return 'approach'
        if view.hp < 30 and legal('retreat'):
            return 'retreat'
        # An Mani is full strength within 1 tile, half at 2, quarter beyond 2.
        if view.distance <= 1 and legal('harm'):
            return 'harm'
        for name in ('energy_bolt', 'magic_arrow', 'meditate', 'hold'):
            if legal(name):
                return name
        return 'hold'
